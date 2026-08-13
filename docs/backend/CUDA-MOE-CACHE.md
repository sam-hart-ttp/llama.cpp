# CUDA MoE streaming (experimental)

This research extension runs CPU-resident routed experts on CUDA without making
the complete expert tensor resident in VRAM. It has two independent paths:

- **decode cache**: frequently routed experts persist in a bounded,
  layer-partitioned CUDA LRU; cache hits execute on CUDA while the ordinary CPU
  `MUL_MAT_ID` kernel computes misses;
- **prefill streamer**: selected experts pass through two pinned-host and two
  CUDA staging slots, overlapping the next expert upload with current-expert
  computation.

Both paths are opportunistic. Unsupported shapes, allocation failures,
contention, invalidated weights, CUDA errors, or unsuccessful dispatch return to
the stock CPU operation. Streaming is disabled by default and requires an
explicit nonzero budget.

## Quick start

Keep routed experts on the CPU and request explicit budgets:

```sh
GGML_CUDA_MOE_CACHE_RESERVE_MB=512 \
./build/bin/llama-cli \
    -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -ngl 99 -cmoe -fa on \
    --moe-cache-mib 512 \
    --moe-prefetch-mib 16 \
    --moe-cache-stats
```

`--moe-cache-mib` is a per-selected-device upper bound for persistent decode
slabs and their dispatch scratch. `--moe-prefetch-mib` is a per-device upper
bound on the pinned-host staging allocation; the implementation currently needs
exactly two expert-sized host slots and two expert-sized CUDA slots. A zero value
disables that path.

The common argument parser disables CPU weight repacking whenever either path
is requested. The streamer needs canonical, address-stable CPU expert tensors.
Direct library users must likewise load the model with extra/repacked buffer
types disabled; context parameters are applied after model loading and cannot
undo an already repacked tensor.

## Placement and memory

The extension sees only expert tensors that remain on a host buffer with
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. It does not move experts out of VRAM.
Use `-cmoe`, `-ncmoe N`, or tensor overrides to select CPU placement.

The decode budget is resolved on first eligible use:

```text
usable = min(requested MiB, free CUDA memory - reserve MiB)
```

The default reserve is 512 MiB. On a 4 GiB discrete GPU, start with a 128–512
MiB cache and inspect normal CUDA/KV allocations before increasing it. The CUDA
allocator may trim all streaming storage once and retry if a normal allocation
runs out of memory; that device then remains disabled for the session.

On Jetson unified-memory systems, “CPU-resident” and CUDA memory draw from the
same physical DRAM pool. Streaming can still reduce the CUDA virtual/device
working set and avoid keeping every expert in an accelerator allocation, but it
does not reduce the checkpoint's total DRAM requirement. The practical gain is
capacity management and transfer scheduling, not creation of additional RAM.

## Decode cache

For a one-token `MUL_MAT_ID` node, the CPU thread first partitions routed rows
into hits and misses. Hit slots are pinned by a reader count. Only after the
complete CUDA dispatch is accepted are those rows omitted from CPU work. CPU
workers then compute misses concurrently with the CUDA matvec. Collection copies
hit rows back to the normal host output.

Misses are never synchronously uploaded for the current operation. Repeated
misses enqueue bounded low-priority fills for later tokens. Defaults are:

- admit after the second observed miss;
- at most 8 admissions per node;
- 128 queued jobs and 512 MiB of queued source ranges per device;
- fair per-layer targets with shared slack: an idle layer reserves no physical
  slots, while a newly active under-target layer may reclaim from a borrower;
- protected-share LRU replacement, with 8 fresh misses before an at-target
  layer may recycle its own entry or take a true surplus entry;
- a minimum viable pool of 8 expert slices;
- one fill stream per device, with host transfers serialized within a session.

Pools are separated by expert-slice size and quantization type. Shape discovery
waits until the graph's set of expert tensors has repeated, then divides the
budget between observed shapes. Layer-to-device selection is deterministic,
capacity weighted, and sticky while that assignment remains usable.

Within each pool, tensor names containing `blk.N` share a partition for layer
`N`; unusual names without a parseable layer receive a per-tensor partition.
The target is `slots / partitions`, with deterministic distribution of any
remainder. Free slots remain globally borrowable. Once full, an under-target
layer reclaims the oldest entry belonging to a partition strictly over target.
An at-target layer selects the older of its own LRU entry and any true surplus
entry elsewhere, so borrowed slack can migrate without reducing an established
share. This prevents a sequential scan in one transformer layer from sweeping
the hot set of every other layer while still allowing temporarily idle capacity
to be used.

### Eligibility

The decode path currently requires:

- CUDA compute capability 7.0 or newer;
- regular CPU `MUL_MAT_ID` with F32 activations;
- a host weight tensor whose name contains `_exps`;
- a supported quantized CUDA MMVQ weight type;
- 1 to 64 routed rows and at most one token by default;
- an expert slice of at least 256 KiB;
- budget for at least 8 slices plus dispatch scratch.

The lower CC bound deliberately includes Xavier (SM 7.2), the laptop GTX 1650
(SM 7.5), and Thor (SM 11.0). This is a build-support statement, not a promise
that every toolkit version supports all three targets.

## Prefill streamer

For a multi-token `MUL_MAT_ID`, the prompt path:

1. validates every route and groups rows by expert;
2. copies the first selected expert from mmap/RAM into pinned host slot 0 and
   uploads it to CUDA slot 0;
3. quantizes at most 64 activation rows and runs CUDA MMVQ for that expert;
4. starts expert `N+1` in the other slot while expert `N` computes;
5. writes each result back to its original `(token, route)` row;
6. falls back to the complete stock CPU node if any stage fails.

CUDA events prevent a host slot from being overwritten before its H2D copy
finishes and prevent a CUDA weight slot from being reused before its compute
finishes. One session serializes decode dispatch and prefill on each device.

The path measures overlap. After 32 opportunities, a device self-disables
prefill if no next-expert transfer has ever completed before current-expert
compute. This avoids repeatedly paying a streaming overhead that the machine
cannot hide. Disablement affects prefill only; decode caching remains available.

This is selected-expert streaming within one graph operation. It is not the
scheduler-wide whole-tensor lookahead proposed by the separate llama.cpp
prefetch branch.

## Correctness and lifetime

The principal safety rule is: **a CPU row may be skipped only while a valid,
pinned CUDA slot owns that exact expert generation**.

- Dispatch rejection restores all planned hit rows before CPU workers cross
  their existing barrier.
- Collection failure recomputes every skipped row with the stock CPU helper.
- Fill publication checks slot state, key, and generation under the session
  mutex; a cancelled or invalidated upload cannot publish stale contents.
- Eviction requires `readers == 0`; a node keeps its slots pinned until `end`.
- Public host-buffer writes and teardown invalidate overlapping slots, queued
  work, in-flight source reads, and demand records before mutation proceeds.
- Scheduler destruction stops admission, drains active scopes/nodes, joins fill
  workers, and only then releases CUDA and pinned-host storage.
- Backend unload unregisters the provider so later CPU calls cannot jump through
  code from an unloaded CUDA module.

CUDA and CPU quantization/arithmetic are not bit identical. Numerical tests use
normalized squared error, and a near-tie token can still change even when logits
are quality-equivalent.

## Public configuration

Programs using the common parser expose:

| Option | Meaning |
| --- | --- |
| `--moe-cache-mib N` | Persistent decode-cache upper bound per selected CUDA device; `0` disables it |
| `--moe-prefetch-mib N` | Pinned-host prompt staging upper bound per selected CUDA device; `0` disables it |
| `--moe-cache-stats` | Print statistics every five seconds and at teardown without enabling general trace logging |
| `--moe-cache-stats-interval N` | Override the periodic interval in milliseconds; implies `--moe-cache-stats`, and `0` selects teardown-only reporting |

Direct library users can set the experimental fields in
`llama_context_params`: `moe_cache_mib`, `moe_prefetch_mib`, and
`moe_cache_stats_interval_ms`. The statistics field uses `-1` to disable
requested reports, `0` for teardown only, and a positive millisecond interval
for periodic plus teardown reports.

Backend developer controls are read when a scheduler session is created:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | `512` | Free CUDA memory excluded from decode slabs |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | `256` | Minimum eligible expert-slice size |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | `1` | Largest token count using persistent decode cache |
| `GGML_CUDA_MOE_CACHE_MIN_SLOTS` | `8` | Minimum slices in a pool |
| `GGML_CUDA_MOE_CACHE_INSERTS` | `8` | Maximum admissions planned by one node |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` | `2` | Misses required for first admission |
| `GGML_CUDA_MOE_CACHE_THROTTLE` | `8` | Fresh misses required before replacement |
| `GGML_CUDA_MOE_CACHE_QUEUE` | `128` | Queued fill-job bound |
| `GGML_CUDA_MOE_CACHE_QUEUE_MB` | `512` | Queued source-byte bound |
| `GGML_CUDA_MOE_CACHE_STATS_INTERVAL_MS` | unset | Milliseconds between requested statistics; `0` selects teardown only |
| `GGML_CUDA_MOE_CACHE_NDEV` | all | Maximum scheduler-selected CUDA devices |
| `GGML_CUDA_MOE_CACHE_SERIAL_FILL` | `1` | Serialize fill transfers inside a session |
| `GGML_CUDA_MOE_CACHE_MIN_CC` | `700` | Minimum CC as `major*100 + minor*10` |
| `GGML_CUDA_MOE_CACHE_FAIL` | empty | Test-only injection: `dispatch`, `collect`, `insert`, `slab`, `prefill`, or `all` |

The legacy backend variables `GGML_CUDA_MOE_CACHE_MODE=auto|on|off`,
`GGML_CUDA_MOE_CACHE`, `GGML_CUDA_MOE_CACHE_BUDGET_MB`, and
`GGML_CUDA_MOE_PREFETCH_BUDGET_MB` remain available for experiments. The old
`GGML_CUDA_MOE_CACHE_STATS` collection-count interval is also retained for
backend tests and compatibility, but new callers should use the time-based
variable. Prefer the CLI or library parameters because raw environment
variables do not control model repacking.

Keep `GGML_OP_OFFLOAD_MIN_BATCH` above the decode batch size. Generic operation
offload can otherwise move the entire `MUL_MAT_ID` to CUDA before the CPU hybrid
path sees it.

Statistics report `partitions`, `local`, and `reclaim`: the number of known
layer partitions, same-layer evictions, and fair-share cross-layer reclaims.
Ordinary hits refresh a per-partition LRU. Cross-layer selection compares only
the oldest unpinned entry from partitions that are strictly over target.

## Platform builds

The implementation was originally built and tested on the development laptop
with CUDA 11.2, GCC 9, and SM 7.5. A follow-up Windows validation used CUDA
13.3.73, MSVC 19.50, and an RTX PRO 500 Blackwell GPU with compute capability
12.0. Recommended deployment toolchains are:

| Machine | CUDA architecture | Practical toolchain |
| --- | ---: | --- |
| GTX 1650 laptop | `75` | CUDA 12.x/13.x container or the verified host CUDA 11.2 build |
| RTX PRO 500 Blackwell laptop | `120a` | CUDA 13.3.73 with Visual Studio 2026/MSVC 19.50 |
| Jetson AGX Xavier | `72` | The CUDA release supplied by its JetPack image; CUDA 13 no longer targets every pre-Turing architecture |
| Jetson AGX Thor | `110` | Thor JetPack/CUDA 13 image |

On the verified Windows build, use static libraries. The default shared-DLL
layout did not link the cross-backend `ggml_moe_cache` symbol on this branch:

```powershell
cmake -S . -B build-cuda-static -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_STATIC=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda-static --target test-moe-cache llama-cli llama-bench -j 8
```

CMake normalizes architecture `120` to `120a` for this GPU. If the Visual
Studio developer shell does not expose `nvcc`, set `CUDAToolkit_ROOT` and
`CMAKE_CUDA_COMPILER` to the CUDA installation and its `bin\nvcc.exe` path.

Example out-of-tree builds:

```sh
cmake -S . -B build-sm75 -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=ON
cmake --build build-sm75 -j --target test-moe-cache llama-cli

cmake -S . -B build-xavier -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=72 -DLLAMA_BUILD_TESTS=ON

cmake -S . -B build-thor -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=110 -DLLAMA_BUILD_TESTS=ON
```

Build Xavier and Thor natively or in containers based on their matching JetPack
images. A single CUDA 13 binary is not the portable answer for both generations.

The existing `.devops/cuda.Dockerfile` is also usable for a discrete-GPU or
Thor CUDA 13 build. Substitute an image tag that exists in the NVIDIA registry
for the target platform:

```sh
docker build -f .devops/cuda.Dockerfile \
    --build-arg CUDA_VERSION=<available-13.x.y-tag> \
    --build-arg CUDA_DOCKER_ARCH=75 \
    --target light -t llama-moe-sm75:cuda13 .
```

Use `CUDA_DOCKER_ARCH=110` for Thor. Do not use this recipe for Xavier unless
the chosen base image and toolkit explicitly retain SM 7.2 support; its JetPack
container is the supported baseline.

## Validation and benchmarking

The focused GPU regression exercises numerical equivalence, hit/miss
partitioning, dispatch and collection fallbacks, allocation failure, bounded
admission, fair-share borrowing/reclamation, resistance to cross-layer scan
thrashing, small pools, invalidation during fill, concurrent sessions, nested
scope isolation, lifecycle teardown, shape routing, library-level prefill
enablement, and backend reload:

```sh
CUDA_VISIBLE_DEVICES=0 ./build-sm75/bin/test-moe-cache
```

The built CLI should advertise all three controls before a model run:

```sh
./build-sm75/bin/llama-cli --help | rg 'moe-(cache|prefetch)'
```

Cache measurements require a long decode warmup: graph-shape discovery, repeated
demand, and asynchronous fills make the first tokens deliberately cold. Compare
matched placements and repacking policies, and record final hit/fill/fallback
counters. A useful four-arm experiment is:

1. both budgets zero (stock CPU experts);
2. decode cache only;
3. prefill streamer only;
4. both paths.

Use identical model, prompt, context, threads, expert placement, and CUDA-layer
placement in all arms. Report prompt and generation throughput separately. A high
decode hit rate is not sufficient evidence of a speedup: PCIe transfers, result
copies, CPU memory bandwidth, routing locality, and GPU MMVQ speed all matter.

## Current limitations

- CUDA only; other backends register no provider.
- CPU-resident quantized expert `MUL_MAT_ID` only; no fused gate/up/activation
  pipeline or device-resident output handoff.
- No persistence or learned hot-set across scheduler/process lifetimes.
- No cache-aware automatic model placement.
- Prefill uses selected-expert operation-level staging, not cross-layer router
  prediction.
- Direct raw-pointer mutation bypasses invalidation; use backend tensor/buffer
  mutation APIs.
- Separate contexts reserve separate sessions and budgets.
- The public context fields are experimental and change the C ABI of this
  research branch.
- Correctness is tested on synthetic Q4_0 experts; real Qwen3.6 throughput and
  quality still require the model checkpoint on each target machine.
