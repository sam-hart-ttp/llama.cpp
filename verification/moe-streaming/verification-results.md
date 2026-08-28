# Verification results

Date: 2026-08-04; revalidated 2026-08-06 and 2026-08-17. Source base: upstream
llama.cpp `1c3c9674d` plus this extension, since committed on
`moe-streaming-research` as `7cc3ffbc7` (implementation) and `dcacef205`
(verification models), then merged with upstream `b10289` in `9b23d90c5` and
carried forward to upstream `7077abbe1` in `21c898a7b`.

The 2026-08-17 pass revalidated the build and runtime tests after that merge.
The TLA+, Lean, and Maxima results were not re-run because the merge left
`verification/` byte-identical.

## Build and runtime tests

The CUDA target built on the development laptop with CUDA 11.2, GCC 9, and
`CMAKE_CUDA_ARCHITECTURES=75`. Upstream CUDA 11 emits a large number of template
warnings, but the target links successfully.

The focused GPU regression checks:

- hit numerical equivalence;
- dispatch, collection, fill-insert, and slab-allocation fallback;
- pre-census, resident-slot, and in-flight fill invalidation;
- concurrent sessions, repeated lifecycle, and nested/dormant scope isolation;
- multi-shape routing, multi-device override when a second GPU exists, admission
  and eviction throttling;
- layer-partitioned borrowing/reclamation and resistance to cross-layer LRU
  scan thrashing;
- a pool below the legacy 64-slot threshold;
- throttled statistics reporting at a configured interval;
- selected-expert prefill and library-level enable-after-environment-off.

The final executable is run as:

```sh
CUDA_VISIBLE_DEVICES=0 ./build-moe-cuda9/bin/test-moe-cache
```

All 17 runnable cases passed. Multi-device routing was skipped because the
laptop has one CUDA GPU; backend unload/reload was skipped because this
configuration links the CUDA backend statically.

The `llama-completion` target also builds, and `llama-completion --help`
exposes `--moe-cache-mib`, `--moe-prefetch-mib`, `--moe-cache-stats`, and
`--moe-cache-stats-interval`. Earlier revisions of this file named `llama-cli`.
That target is no longer buildable in this configuration: upstream PR #17824
rebuilt the CLI on the in-process server stack and PR #18670 gated it behind
`LLAMA_BUILD_SERVER`, which this build sets to `OFF`. The options are
registered in `common/arg.cpp`, so every tool built from common args still
carries them.

The common argument-parser target builds. Running the complete existing parser
suite offline eventually reaches a pre-existing network download test and exits
with `cannot make GET request`; the new option assertions execute before that
external failure.

### Follow-up Windows validation

On 2026-08-13, the branch built on Windows with Visual Studio 2026/MSVC
19.50.35723.0, CUDA 13.3.73, and `CMAKE_CUDA_ARCHITECTURES=120` (normalized by
CMake to `120a`) for an RTX PRO 500 Blackwell laptop GPU (compute capability
12.0, 6112 MiB VRAM). The build used `GGML_STATIC=ON` and
`BUILD_SHARED_LIBS=OFF`; the default shared-DLL layout did not link the
cross-backend `ggml_moe_cache` symbol on this branch.

The static `llama-cli`, `llama-bench`, and `test-moe-cache` targets linked
successfully. The focused test reported 16 passing cases. Multi-device routing
was skipped because only one CUDA device is present, and backend unload/reload
was skipped because the backend is statically linked.

The official Qwen3.6-35B-A3B Q4_K_M GGUF then loaded successfully with
`--moe-cache-mib 256` and automatic placement. The model loader reported a
19.06 GiB CPU-mapped model buffer and a 3.78 GiB CUDA model buffer, with all 41
layers scheduled across the hybrid placement. A one-token smoke prompt measured
11.1 tokens/s on the initial prompt and 16.5 tokens/s after prompt-cache reuse.
This was a load and execution smoke test, not a cache throughput benchmark; the
cache needs a longer matched decode run to produce useful hit and speed data.

These checks are collected in `merge-gate.sh` in this directory, which is the
script to run after every merge from `upstream/master`. It adds two checks that
this file previously carried only as prose: the count of upstream
`TAG_MUL_MAT_ID_CUDA_GRAPHS` sites, and a decode hit-rate floor. The hit-rate
floor is the only check that can detect scheduler placement drift, because
`test-moe-cache` drives the provider API directly and stays green even if
expert `MUL_MAT_ID` nodes stop reaching the CPU backend handler.

### Windows laptop capacity ladder

On 2026-08-28, after merging the current GitHub branch, the static Windows CUDA
build was refreshed on the Dell Pro Max 14 with CUDA 13.3.73 and MSVC. The
`llama-completion`, `llama-bench`, `test-moe-cache`, and `test-arg-parser`
targets linked. The focused `test-moe-cache` run reported 17 OK cases, with the
multi-device and backend-unload cases skipped as expected on a one-GPU static
build.

The machine used for the ladder exposed an RTX PRO 500 Blackwell laptop GPU
(6113 MiB VRAM, compute capability 12.0) to WSL, with about 15 GiB RAM visible
inside WSL. Logs, CSV summaries, and GPU-memory samples were kept as ignored
local files under `verification/moe-streaming/run-logs/`; they are not part of
this branch.

The existing Qwen3.6-35B-A3B Q4_K_M GGUF was used first to map the decode-cache
budget. Runs used `llama-completion`, `-n 256`, `-c 512`, `-t 6`, `-ngl 99`,
`-cmoe`, `--no-repack`, and `-fa on`. All cache-size rows exited 0. Throughput
varied enough that the table should be read as a capacity and hit-rate result,
not a confidence interval.

| Model | Cache request | Decode hit rate | Slots used | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B Q4_K_M | 128 MiB | 6.6% | 223/223 | 2748 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 256 MiB | 16.8% | 451/451 | 2876 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 768 MiB | 38.4% | 1361/1361 | 3388 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 1024 MiB | 38.4% | 1816/1816 | 3644 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 1792 MiB | 53.3% | 3181/3181 | 4412 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 2048 MiB | 53.2% | 3608/3608 | 4652 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 3072 MiB | 53.3% | 3608/3608 | 4652 MiB |

For this prompt and context, Qwen3.6 reached its observed effective decode-cache
plateau at about a 2048 MiB request: larger requests did not increase slots or
peak GPU memory. Separate prefill-budget probes showed that a 1 MiB prefill
budget did not activate useful reporting, while 2 MiB and larger budgets ran 240
prefill nodes over 9600 rows with `hidden=6513/6513`. On this laptop that prompt
path remained slower than the baseline, so the prefill result is a viability
check, not evidence of a speedup.

The first larger capacity target was the split GGUF
`bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF`, quantized as IQ2_XS. The
two shards were downloaded directly into `models/`, verified against Hugging
Face API sizes, and occupied 61.110 GiB total. This avoided an extra Hugging
Face cache copy. Free space on the C: volume after download was about 115 GiB.

A lowest-risk smoke run used `-ngl 0`, `-n 1`, `-c 256`, `-cmoe`, `--no-repack`,
and both MoE streaming budgets set to zero. It exited 0, loaded in 21.99 s, and
measured 0.35 prompt tokens/s with a 626 MiB GPU-memory peak. Further one-token
smokes with cache disabled all exited 0 for `-ngl` values 8, 16, 24, 32, 48, 64,
80, and 99. The `-ngl 99` smoke peaked at 3802 MiB, leaving room for a bounded
expert cache.

The A22B viability ladder used the same short prompt, `-c 256`, `-t 6`, `-cmoe`,
`--no-repack`, and `-fa on`. A 128 MiB CUDA reserve was used for the tradeoff
runs that request larger caches. All rows below exited 0.

| Model | Tokens | `-ngl` | Cache request | Decode tokens/s | Decode hit rate | Slots used | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-235B-A22B IQ2_XS | 64 | 99 | 0 MiB | 1.44 | n/a | n/a | 3802 MiB |
| Qwen3-235B-A22B IQ2_XS | 64 | 99 | 1536 MiB | 1.47 | 9.4% | 518/518 | 4654 MiB |
| Qwen3-235B-A22B IQ2_XS | 64 | 80 | 3072 MiB | 1.46 | 15.3% | 799/799 | 4652 MiB |
| Qwen3-235B-A22B IQ2_XS | 64 | 0 | 4096 MiB | 1.26 | 39.1% | 2461/2461 | 4652 MiB |
| Qwen3-235B-A22B IQ2_XS | 256 | 80 | 3072 MiB | 1.37 | 14.2% | 799/799 | 4652 MiB |
| Qwen3-235B-A22B IQ2_XS | 256 | 0 | 4096 MiB | 1.29 | 38.8% | 2461/2461 | 4652 MiB |

The A22B model is therefore viable on this laptop with mmap-backed CPU-resident
experts and a bounded CUDA expert cache. The largest observed expert-cache
working set was 2461 slots at `-ngl 0`, but that was not the fastest setting:
keeping dense layers on the GPU while accepting fewer expert slots gave better
throughput in the sustained runs. The useful tuning variable is the balance
between dense-layer offload and expert-cache capacity, not simply the largest
cache request.

## TLA+

TLC 2.20 used 12 workers and exhaustive breadth-first search.

| Module/configuration | Expected | Result |
| --- | --- | --- |
| `MoECache.tla` / `MoECache.cfg` | pass | **Pass**: 35,221 generated, 8,100 distinct, depth 14 |
| `MoEPrefetch.tla` / `MoEPrefetch.cfg` | pass | **Pass**: 287 generated, 113 distinct, depth 13 |
| `MoEPrefetchFallbackHunt.cfg` | pass | **Pass**: 287 generated, 113 distinct, depth 13 |
| `MoEPrefetchDisableHunt.cfg` | pass | **Pass**: 313 generated, 121 distinct, depth 13 |
| `MoEPartitionPolicy.tla` / `MoEPartitionPolicy.cfg` | pass | **Pass**: 14,635 generated, 3,629 distinct, depth 22 |
| `MoEPartitionGlobalLRUHunt.cfg` | find injected bug | **Expected violation**: `ProtectedSharesRemain`, depth 7 |
| `MoECachePinnedEvictionHunt.cfg` | find injected bug | **Expected violation**: `PinnedExpertIsStable`, depth 7 |
| `MoECacheStalePublishHunt.cfg` | find injected bug | **Expected violation**: `PublishedGenerationIsCurrent`, depth 6 |
| `MoECacheEarlySkipHunt.cfg` | find injected bug | **Expected violation**: `SkippedRowsAreBacked`, 76 generated / 44 distinct, depth 9 |

The four failing configurations intentionally relax one implementation guard;
they are mutation tests of invariant sensitivity, not reported implementation
bugs. SANY accepts `MoECache.tla`, `MoEPrefetch.tla`, `MoEPartitionPolicy.tla`,
and the timebox `Trace.tla`.

Hunt-row depths are detection-time values; they vary with TLC version and
worker scheduling, while the violated invariant and the exhaustive-run state
counts are the stable signals. A 2026-08-06 rerun with a current TLC nightly
reproduced every row with identical state counts. The same pass repaired the
trace harness (string constants in `Trace.cfg`, weak fairness in `TraceSpec`,
and the environment-variable `JSON` override): as originally committed it
compared model values against JSON strings, so TLC accepted any log after
matching zero events. TLC now validates the hand-written smoke trace
`traces/example-decode.ndjson` end to end (5 states, 4 distinct, all three
events consumed), and rejects the same trace with one post-state field
falsified: the corrupted event matches no action and TLC stops at a deadlock.
That negative case is what distinguishes a working validator from the
previous always-green one.

## Lean 4

```sh
/home/optimist/.elan/bin/lean \
    verification/moe-streaming/lean/MoEPartition.lean
```

Lean 4.28.0 exits 0 with no proof warnings and no `sorry`; a 2026-08-06 rerun
on Lean 4.32.2 also exits 0. The launcher reports only that it cannot query a
newer release in the offline environment. The proof establishes that hit/miss
partitioning preserves the additive multiset of route contributions, expert
grouping preserves that expert's contribution, and every bounded prefix cut
plus remainder preserves it. It also proves the two-partition fair-share
lemma: a full pool and under-target requester imply an over-target donor, whose
target survives a one-slot reclaim. It abstracts from IEEE-754 rounding and
CUDA kernel correctness. The file imports only `Std`, so it checks from a bare
`elan` toolchain with no lakefile or dependency fetch.

## Mathematical scope

Connected Maxima additionally checked the fair-share arithmetic. It reduced
`r*(q+1)+(p-r)*q-(p*q+r)` to zero, confirming deterministic remainder quotas
sum to the slot count, and solved the two-partition full-pool equation as
`donor=2*quota-requester`, whose over-target margin is `quota-requester`.

## Hardware gap

The Qwen3.6-35B-A3B Q4_K_M GGUF used for the measurements below was present on
the laptop when they were taken. It is not on this workspace as of 2026-08-17,
so the `merge-gate.sh` hit-rate floor reported SKIP on that revalidation and
the figures below have not been re-measured since. Before the layer-partitioning
change, a matched 256-token decode measured
24.22 token/s with streaming off and 25.16 token/s with a requested 768 MiB
cache (627 MiB effective after scratch/reserve, 1,115 slots, 33.0% hit rate).
A later matched A/B used 1,179 slots (663 MiB slab), six CPU threads, a fixed
33-token prompt, and 255 timed decode runs. The temporary cache-wide-LRU control
measured 25.53 token/s, 34.0% hits, and 13,206 cross-layer evictions. The final
partition-aware policy measured 25.57 token/s, 34.6% hits, 13,577 local
evictions, and only 65 legal surplus reclaims. This single short run is not a
confidence interval, but it demonstrates that isolation need not cost
throughput. Xavier/Thor are not attached to this workspace, so SM 7.2/SM 11.0
builds and target-specific performance remain deployment validation tasks.
