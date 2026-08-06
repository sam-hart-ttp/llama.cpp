# Verification results

Date: 2026-08-04; revalidated 2026-08-06. Source base: upstream llama.cpp
`1c3c9674d` plus this extension, since committed on `moe-streaming-research`
as `7cc3ffbc7` (implementation) and `dcacef205` (verification models), then
merged with upstream `b10289` in `9b23d90c5`.

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
- selected-expert prefill and library-level enable-after-environment-off.

The final executable is run as:

```sh
CUDA_VISIBLE_DEVICES=0 ./build-moe-cuda9/bin/test-moe-cache
```

All runnable cases passed. Multi-device routing was skipped because the laptop
has one CUDA GPU; backend unload/reload was skipped because this configuration
links the CUDA backend statically.

The CPU `llama-cli` target also builds, and `llama-cli --help` exposes
`--moe-cache-mib`, `--moe-prefetch-mib`, and `--moe-cache-stats`.

The common argument-parser target builds. Running the complete existing parser
suite offline eventually reaches a pre-existing network download test and exits
with `cannot make GET request`; the new option assertions execute before that
external failure.

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

A local Qwen3.6-35B-A3B Q4_K_M GGUF is now present and runs on the GTX 1650
laptop. Before the layer-partitioning change, a matched 256-token decode measured
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
