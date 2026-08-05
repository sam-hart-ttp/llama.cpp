# Concrete llama.cpp extension: design and audit report

## Outcome

This branch now contains a portable CUDA/ggml implementation of the useful
common denominator from Swiftlet and AirLLM: leave the full MoE checkpoint in
ordinary host storage, but move only routed expert slices through a bounded
accelerator working set. It reuses llama.cpp's model loaders, GGUF/mmap,
quantized CPU kernels, CUDA MMVQ, scheduler, KV cache, sampling, and model-family
graphs rather than creating a third inference runtime.

The implementation is a research-quality vertical slice. Its synthetic CUDA
regression is comprehensive and passes on the GTX 1650 development laptop. A
local Qwen3.6-35B-A3B Q4_K_M GGUF also completes an end-to-end decode benchmark;
the measured result remains machine- and placement-specific rather than a
production default.

## Architecture

### Integration boundary

`ggml_backend_sched` owns an opaque provider session. The CUDA backend registers
one private provider table; CPU `MUL_MAT_ID` calls it only while its scheduler's
thread-local scope is active. This keeps ordinary CPU graphs and non-CUDA builds
inert, prevents one context from accidentally consuming another context's
cache, and gives teardown one clear owner.

Public configuration flows from common CLI parameters to experimental
`llama_context_params`, then to
`ggml_backend_sched_set_moe_streaming()`. An inert session is still constructed
when the environment says “off”, because the library API is applied after
scheduler construction and must be able to activate it.

### Decode

Each expert tensor shape receives a device slab divided into fixed expert-size
slots. A key is `(host tensor base, expert index)`. The plan operation, under one
session mutex:

1. pins valid hits and refreshes their LRU position;
2. counts misses without removing them from CPU work;
3. after admission thresholds, reserves a free slot, a layer-local LRU victim,
   or an over-target donor from another layer;
4. increments its generation, marks it copying, and enqueues a bounded job.

The fill worker copies mmap/RAM into reusable pinned memory, uploads on a
low-priority stream, synchronizes, then publishes only if key, state, and
generation still match. Thus current-token latency never waits for a demand
fill; a miss may become a hit on later tokens.

For planned hits, CUDA dispatch uploads compact slot IDs and F32 activation
rows, quantizes activations to Q8_1, and calls a small wrapper around llama.cpp's
existing MMVQ dispatcher. CPU workers compute misses concurrently. Results are
downloaded into the ordinary host `MUL_MAT_ID` output.

Pools use fair-share targets per transformer layer with shared slack. An idle
layer consumes no slots, so active layers can borrow the whole pool. When a new
layer becomes active it reclaims only from a partition above target. Once at
target, a layer chooses the older of its own LRU entry and any genuine surplus
entry, allowing borrowed slack to migrate without reducing an established
share. This stops one layer's sequential expert scan from becoming a cache-wide
scan.

### Prefill

The prompt path sorts routes by expert and handles at most 64 rows per MMVQ
chunk. Two pinned-host expert slots and two CUDA expert slots alternate. A copy
stream publishes “ready”; a compute stream waits, quantizes activations, runs
MMVQ, downloads results, then publishes “consumed”. Upload of the next expert is
issued during current-expert compute.

This granularity is important. The active llama.cpp prefetch branch copies whole
weight tensors one scheduler split ahead. This extension instead knows the
router's selected experts and transfers only those expert slices inside the
operation. It preserves mmap as the durable source and needs only two
expert-sized pinned buffers.

### Memory pressure and invalidation

Decode slabs obey an explicit per-device cap and free-memory reserve. The normal
CUDA allocator may ask the provider to trim a device and retry an OOM. Prefill
staging is also bounded, but its budget is a threshold rather than a pool to
fill: two expert slices must fit.

Every public host-weight buffer/tensor mutation path sends an overlapping range
notification before mutation or release. Invalidation cancels queued work,
waits for active nodes and in-flight fill-source reads, resets slots and demand
records, then returns. This is necessary because cache jobs retain raw pointers
into mmap/host allocations.

## Correctness assessment

### Strong properties

- A hit is never removed from CPU work until the entire CUDA dispatch is
  accepted.
- Dispatch failure restores all rows before the existing CPU worker barrier.
- Collection failure recomputes all skipped rows with the unmodified CPU helper.
- Reader pins prevent LRU eviction until a node ends.
- Established layer shares cannot be reduced by another at-target layer;
  cross-layer reclamation requires the donor to be strictly over target.
- Fill generations prevent cancellation/reuse ABA at a slot.
- Prefill success is whole-node success; any partial failure synchronizes both
  streams and returns to complete-node CPU computation.
- Provider lifetime follows backend load/unload and scheduler construction/
  destruction.
- The CPU partition and prefill grouping/chunking algorithms have a no-`sorry`
  Lean proof that their additive route multiplicity is preserved.

### Bugs found while implementing

1. **Conditional extra barrier deadlock — fixed.** The first prompt integration
   added a barrier reached only by thread 0 after provider success. GPU tests
   hung. The final design stores a shared `-1` sentinel and uses the kernel's
   single pre-existing barrier, after which every worker returns together.
2. **Library enable-after-off failure — fixed.** Session creation returned null
   for environment-disabled streaming, so later context parameters had nothing
   to configure. Disabled sessions are now inert objects that the public API can
   revive; the prefill test exercises this route.
3. **Legacy 64-slot gate defeated the new 8-slot policy — fixed.** Allocation
   still rejected `slot_count < 64` before the configurable retry loop. It now
   checks `min_slots`, with a one-MiB regression whose pool is below 64 slots.
4. **TLA+ completion-expression precedence — fixed.** TLC exposed an undefined
   branch in the first prefetch model; parentheses now make
   `completedCorrect'` a single Boolean assignment.
5. **Cross-layer global-LRU thrashing — fixed.** A sequential expert scan in one
   layer could evict useful entries from every other layer. Victim eligibility
   is now fair-share partitioned by layer, with shared free capacity and
   over-target reclamation. A four-layer/16-slot GPU regression distinguishes
   the new policy from the old cache-wide LRU.

### Remaining correctness limitations

- The GPU and CPU paths are numerically close, not bit identical. This can alter
  a near-tie token even when normalized error is small.
- Direct raw-pointer writes bypass backend invalidation. They are outside the
  supported mutation contract.
- Dynamic backend unload must not race arbitrary calls already executing inside
  the unloaded module; this is the existing backend lifecycle precondition.
- The library context fields configure execution after model load and therefore
  cannot undo an already chosen repacked CPU weight buffer.
- Only quantized MMVQ types are accepted; F16/F32 and any fused expert operation
  fall back.

## Completeness assessment

Implemented:

- explicit CLI and library budgets, disabled by default;
- CPU expert placement compatibility and repack avoidance through common CLI;
- scheduler-scoped provider lifecycle and nested-scope isolation;
- bounded persistent decode cache with admission, layer-aware fair-share LRU,
  generations, reader pins, multi-shape and multi-device routing;
- selected-expert double-buffered prompt streaming;
- CUDA allocator-pressure trim;
- host weight invalidation across public mutation/free paths;
- fallback/error injection and statistics;
- focused GPU, CLI, TLA+, and Lean validation;
- platform/build/benchmark documentation.

Not implemented:

- cache-aware automatic placement or fit accounting;
- direct chaining of gate/up/down expert tensors or fused SwiGLU;
- CUDA-resident result handoff into the next graph operation;
- learned/persisted hot sets, predictive router lookahead, or disk-range
  coalescing;
- equivalent HIP/Vulkan/Metal providers;
- real-model benchmark automation or validated production defaults.

The largest performance gap is the host round trip for every hit result and the
operation-level separation of gate/up/down expert matrices. Correctness is
complete enough for experimentation; performance architecture is intentionally
conservative.

## Comparison with the two audited repositories

| Dimension | Swiftlet | AirLLM | This llama.cpp extension |
| --- | --- | --- | --- |
| Model semantics | Reimplemented Swift/Metal Qwen runtime | Transformers model/hooks | Existing llama.cpp graph/model code |
| Durable source | Fixed-stride `.qpack` expert files | Per-layer safetensors | GGUF mmap/host tensors |
| Streaming unit | One packed routed expert blob | Usually whole module; Kimi-specific expert hooks | One quantized matrix's selected expert slice |
| Decode reuse | Persistent LFU/recency cache | Generally reloads modules | Bounded generational, layer-partitioned LRU |
| Prefill | Batch-union selected experts | Layer streaming/prefetch | Sorted selected experts, 2-slot overlap |
| Accelerator | Apple Metal/unified memory | Primarily CUDA/PyTorch | CUDA through ggml; SM 7.0+ |
| Correctness oracle | CPU/Metal/MLX fixtures | Transformers itself | Stock CPU `MUL_MAT_ID` fallback/differential test |
| Portability burden | Custom kernels per Apple/model family | Python dependency/hook compatibility | One ggml provider plus existing model support |
| Main risk | Custom math and shared recurrence | Hook/materialization lifecycle | Concurrent slot lifetime and transfer profitability |

Relative to Swiftlet, llama.cpp gives up the ideal coalesced gate/up/down expert
blob and direct Metal-style zero-copy path, but gains Linux/NVIDIA support and a
large existing quantized model ecosystem. Relative to AirLLM, it gives up broad
Transformers plug-in flexibility, but avoids Python hooks, repeated safetensors
materialization, and dependence on remote model code.

## Comparison with the local llama.cpp references

### `reference/moe-cache-pr`

The persistent decode design is deliberately close to the reworked reference:
hybrid CPU misses/CUDA hits, per-scheduler sessions, generations, reader pins,
bounded workers, safe invalidation, and allocator trim. This branch changes the
product assumptions:

- explicit budget default rather than automatically active behavior;
- SM 7.0 minimum, 512 MiB reserve, 256 KiB expert minimum, and 8-slot pool
  minimum for the GTX 1650/Xavier class;
- public context parameters in addition to backend environment controls;
- a separate prompt path and runtime profitability disablement.

### `reference/prefetch`

That branch adds scheduler-level asynchronous whole-weight copy one graph split
ahead and backend hooks for copy streams/events. It is generic across operations
but needs the source tensor in a copyable/pinned arrangement and transfers
weights even when only a sparse subset of experts will execute. This extension
does no graph lookahead: it waits until router IDs exist, stages only selected
expert slices, and overlaps consecutive experts inside one `MUL_MAT_ID`.

The approaches can eventually compose: scheduler lookahead for dense weights,
selected-expert staging for sparse tensors, and a persistent decode hot set.

## Platform implications

- **GTX 1650, 4 GiB**: useful primarily as a bounded expert coprocessor. Keep
  dense placement/KV within the remaining VRAM, use a modest decode budget, and
  expect PCIe and host-RAM bandwidth to dominate. The checkpoint must still fit
  the laptop's 30 GiB RAM/mmap working environment.
- **Jetson AGX Xavier, 32 GiB unified DRAM**: much more model capacity than the
  laptop and no discrete PCIe host/device topology, but old Volta-class compute
  and memory bandwidth make throughput uncertain. Use its matching JetPack
  CUDA, not a CUDA 13-only build.
- **Jetson AGX Thor, 128 GiB unified DRAM**: Qwen3.6-35B-A3B already fits, as the
  user's daily deployment demonstrates. Streaming matters more for larger MoE
  checkpoints, for keeping more context/other services resident, or as a
  selective cache over CPU/unified allocations. SM 11.0 and much larger memory
  substantially change the profitable cache/prefill regime.

The memory technique generalizes cleanly; profitable policy does not. Each
machine needs matched off/cache/prefetch/both measurements before enabling it by
default.

## Verification summary

- Full CUDA target builds with host CUDA 11.2/GCC 9/SM 7.5 (upstream CUDA 11
  warning noise remains).
- Focused GPU regression covers normal and injected failure paths, concurrency,
  lifecycle, invalidation, routing, small capacity, and prefill.
- `MoECache.cfg`: exhaustive TLC pass, 8,100 distinct states, depth 14.
- `MoEPrefetch.cfg`: exhaustive TLC pass, 113 distinct states, depth 13.
- `MoEPartitionPolicy.cfg`: exhaustive TLC pass, 3,629 distinct states, depth
  22; the cache-wide-LRU mutation violates `ProtectedSharesRemain` at depth 7.
- Four deliberately mutated cache hunts each produce the intended invariant
  counterexample: pinned eviction, stale publication, early CPU skip, and
  cache-wide cross-layer eviction.
- `MoEPartition.lean` compiles with no `sorry`; it now also proves that a full
  equal-share pool with an under-target requester has an over-target donor and
  that reclaiming one slot preserves the donor target.
- Connected Maxima independently reduced the production remainder-quota sum to
  `p*q+r` and the two-partition donor margin to `quota-requester` under the Lean
  assumptions.

See `../../../audit/README.md` for the original two-repository audit and
`docs/backend/CUDA-MOE-CACHE.md` for operational instructions.
