# CUDA MoE streaming modeling brief

## 1. System overview

The target is the experimental CUDA MoE streaming extension in this llama.cpp
fork. Its core is about 2,900 lines of C++/CUDA plus a 110-line CPU integration
path. It is **Category B (concurrent runtime)**: a CPU graph thread, CPU worker
threads, CUDA streams, and background fill workers transfer ownership of expert
slots under a mutex, reader pins, generations, events, and condition variables.
The decode algorithm partitions one `MUL_MAT_ID` node into CUDA cache hits and
CPU misses; prefill streams selected experts through two alternating slots.

The implementation derives the persistent cache from local branch
`reference/moe-cache-pr` (`708c18c2b`, reworked by `a01eb2646`) and the overlap
idea from `reference/prefetch` (`449735fe0` through `6e3d2ef73`). It differs from
both: budgets are explicit and disabled by default, the minimum target is SM
7.0, prompt streaming is selected-expert/double-buffered rather than
scheduler-wide whole-weight lookahead, and prefill self-disables when overlap is
not observed.

## 2. Bug families

### Family 1: pin, eviction, and slot-generation ownership

**Mechanism**: a planned hit holds a local slot identity while the fill worker
and later nodes may recycle that slot; correctness needs both a reader pin and a
matching publication generation.

**Evidence**:

- `moe-cache.cu:1840-1866` increments `readers` while holding `session.mu`.
- `moe-cache.cu:588-627` and `moe-cache.cu:1903-1952` reject pinned eviction
  and increment the generation before reuse.
- `moe-cache.cu:875-894` publishes only when slot state, key, and generation
  still match the job.
- `moe-cache.cu:2229-2248` releases pins only after dispatch/collection ends.

**Affected paths**: `moe_cache_plan`, `moe_cache_worker`,
`moe_cache_slot_reset`, `moe_cache_end`.

**Suggested model**: split reserve, asynchronous publication, dispatch, collect,
and release. Track slot state, expert, generation, published generation, reader
count, and the node's pinned slot.

**Priority**: High. A failure is silent wrong-expert computation or use of
overwritten device memory.

### Family 2: hybrid hit/miss fallback completeness

**Mechanism**: CPU rows are removed speculatively, so dispatch and collection
failure paths must restore exactly every omitted route and no successful row may
be computed zero times.

**Evidence**:

- `ggml-cpu.c:1676-1703` partitions hit and miss rows.
- `ggml-cpu.c:1706-1717` restores all hits before the existing worker barrier if
  dispatch is rejected.
- `ggml-cpu.c:1798-1811` recomputes every skipped hit if collection fails.
- An earlier implementation added a second barrier on only one thread and
  deadlocked the test; the current sentinel uses the kernel's existing barrier
  at `ggml-cpu.c:1730-1734`.

**Affected paths**: CPU `MUL_MAT_ID`, `moe_cache_dispatch`,
`moe_cache_collect`.

**Suggested model**: keep a CPU obligation until CUDA dispatch is accepted;
make dispatch rejection and collection failure distinct actions; require every
completed node to have a correct result.

**Priority**: High. The failure modes are deadlock or corrupt logits.

### Family 3: host-weight invalidation versus asynchronous readers

**Mechanism**: cached and queued entries use raw ranges into mmap/host buffers;
mutation or destruction must cancel queued work and wait for active source
copies before memory changes.

**Evidence**:

- `ggml-backend.cpp:94-103` limits notifications to host weight buffers.
- Public clear/reset/set/memset/copy/free paths notify before mutation.
- `moe-cache.cu:2687-2710` cancels queued jobs and waits for overlapping active
  nodes or the fill worker's in-flight source.
- `moe-cache.cu:2712-2783` resets overlapping slots and discovery records.

**Affected paths**: backend buffer/tensor mutation API, fill worker,
`moe_cache_invalidate_session`.

**Suggested model**: split invalidation start and finish around the reader-drain
window; allow cancellation between reserve and publish; require a cancelled job
never to publish an old generation.

**Priority**: High. This is the reclamation/grace-period analogue for mmap
weight ranges.

### Family 4: alternating prefetch-slot reuse and partial failure

**Mechanism**: pinned host slot `N mod 2` and CUDA slot `N mod 2` are reused by
different streams. Host reuse needs copy completion; device reuse needs compute
consumption. Any partial failure must abandon all partial output and recompute
the complete CPU node.

**Evidence**:

- `moe-cache.cu:2522-2551` waits for ready/consumed events before slot reuse.
- `moe-cache.cu:2562-2650` orders ready, compute, result, and consumed events.
- `moe-cache.cu:2672-2679` synchronizes both streams before returning failure.
- `ggml-cpu.c:1599-1613` accepts the streamed result only on a whole-node true
  return.

**Affected paths**: `moe_cache_prefill`, CPU prompt branch.

**Suggested model**: two slots with free/copying/ready/computing states; group
completion; nondeterministic pipeline failure; all-or-nothing CPU fallback.

**Priority**: High. Premature reuse computes with the wrong expert; partial
acceptance leaves missing route rows.

### Family 5: performance disablement must preserve correctness

**Mechanism**: prefill self-disable is driven by measured event overlap, but it
must transition only to complete CPU fallback rather than leaving partial staged
state or incomplete output.

**Evidence**: `moe-cache.cu:2657-2669` disables after 32 opportunities with no
fully hidden transfer; later calls reject at `moe-cache.cu:2395-2423` and the CPU
path executes normally.

**Affected paths**: prefill statistics/update, entry eligibility, CPU fallback.

**Suggested model**: opportunities, fully-hidden count, disabled flag, fallback
obligation, and completion set.

**Priority**: Medium. The policy is performance-only, but a bad transition would
be a correctness failure.

### Family 6: cross-layer admission and eviction isolation

**Mechanism**: a single cache-wide LRU lets a sequential expert scan in one
transformer layer evict the hot entries of every other layer. The improved
policy assigns each observed layer a fair-share target while retaining shared
free capacity. A full-pool admission must either replace the requester's own LRU
entry or reclaim from a different partition strictly over target.

**Evidence**:

- `moe-cache.cu:557-627` caches deterministic targets and searches per-layer
  recency lists for legal same-layer or over-target candidates.
- `moe-cache.cu:1880-1945` selects the low admission threshold below target,
  throttles replacement at target, and records local versus reclaim eviction.
- `test-moe-cache.cpp:1305-1399` fills a 16-slot pool from one layer, reclaims
  four shares, churns one layer, and requires all other layers to remain hot.

**Affected paths**: `moe_cache_plan`, `moe_cache_lru_candidate`, pool discovery,
slot reset, and invalidation metadata cleanup.

**Suggested model**: abstract away exact recency ordering but retain free-slot
borrowing, threshold changes, resident counts, local replacement, over-target
reclamation, and a cache-wide-victim mutation. Once a partition has reached its
target, require that other partitions cannot push it below target.

**Priority**: Medium. This does not change logits, but persistent cross-layer
thrashing can erase the feature's throughput benefit.

## 3. Modeling recommendations

### 3.1 Model

- Model Families 1–3 together in `MoECache.tla` because the interesting states
  compose: pinning, cancellation, stale worker publication, dispatch, and
  collection.
- Model Families 4–5 in `MoEPrefetch.tla` with exactly two explicit slots and
  separate copy/compute completion actions.
- Model Family 6 separately in `MoEPartitionPolicy.tla`; it is an admission
  policy abstraction rather than an asynchronous slot-lifetime protocol.
- Preserve the real mutex atomicity: slot lookup/reserve/publication happen as
  individual locked actions; CUDA transfer and compute completion remain
  interleavable between them.
- Inject three implementation mutations as hunt toggles: pinned eviction, stale
  publication after cancellation, and removal of a CPU row before dispatch has
  accepted it.

### 3.2 Do not model

- Quantized CUDA arithmetic and floating-point rounding: differential GPU tests
  and Lean's additive routing abstraction are more suitable.
- Exact LRU ordering, weighted multi-device placement, and byte-budget
  arithmetic. The Family 6 model covers victim eligibility and quota isolation,
  but nondeterministically chooses within the legal LRU candidate set.
- CUDA's hardware memory model: synchronization is through API stream/event
  ordering and host mutexes; model their documented completion boundaries.
- Model loading/repacking policy and CLI parsing: unit-testable configuration,
  not a concurrent protocol.
- Raw-pointer mutation outside the backend API: explicitly outside llama.cpp's
  normal synchronization contract.

## 4. Proposed extensions

| Extension | Variables | Purpose | Family |
| --- | --- | --- | --- |
| Generational slots | `generation`, `publishedGeneration`, `jobGeneration` | Reject cancelled/stale fill publication | 1, 3 |
| Reader ownership | `readers`, `pinned`, `slotExpert` | Prevent eviction/reuse while a node uses a slot | 1 |
| Hybrid obligations | `cpuRequired`, `gpuAccepted`, `resultCorrect` | Prove all dispatch/collect fallback branches | 2 |
| Invalidation window | `invalidating`, job and slot state | Model cancel, drain, and reclamation order | 3 |
| Two-stage pipeline | `stageState`, `stageExpert`, `currentExpert`, `outputs` | Model alternating host/device slot ownership | 4 |
| Adaptive disable | `opportunities`, `fullyHidden`, `disabled`, `fallbackRequired` | Keep the performance policy semantics-neutral | 5 |
| Layer fair shares | `owner`, `demand`, `protected`, resident counts | Preserve hot sets while allowing borrowing/reclamation | 6 |

## 5. Proposed invariants

| Invariant | Type | Description | Targets |
| --- | --- | --- | --- |
| `TypeOK` | Safety/structural | Every variable remains in its finite domain | all |
| `NoReadersOfUnpublishedSlots` | Safety | Reader pins exist only on valid slots | 1 |
| `PublishedGenerationIsCurrent` | Safety | A valid slot was published for its current generation | 1, 3 |
| `PinnedExpertIsStable` | Safety | A pinned slot remains valid and owns the node's expert | 1 |
| `SkippedRowsAreBacked` | Safety | Removing CPU work requires accepted CUDA ownership | 2 |
| `CompletedResultsAreCorrect` | Safety | Every completed decode node has a valid CPU/GPU result | 2 |
| `ComputeOwnsReadyExpert` | Safety | Prefill compute owns a staged expert | 4 |
| `NoDuplicateStaging` | Safety | The two live slots do not alias one expert lifecycle | 4 |
| `HiddenTransfersAreOpportunities` | Safety | Measurement bookkeeping cannot overcount hidden copies | 5 |
| `CompletionIsTotal` | Safety | Successful prefill produced every selected expert group | 4 |
| `DisabledRequiresFallbackOrCompletion` | Safety | Self-disable cannot strand a partial node | 5 |
| `CapacityAccounting` | Safety/structural | Every cache slot is either free or owned exactly once | 6 |
| `ProtectedSharesRemain` | Safety | Once a partition reaches target, other demand cannot push it below target | 6 |
| `UnderQuotaHasDonor` | Safety/arithmetic | A full pool with an under-target requester has an over-target donor | 6 |
| `ReadyDemandCanProgress` | Safety/enabledness | Admission thresholds and victim rules cannot strand ready demand | 6 |

## 6. Findings pending verification

### 6.1 Model-checkable

| ID | Description | Expected invariant violation | Family |
| --- | --- | --- | --- |
| MC-1 | Evict/reassign a slot while a node holds its reader pin | `PinnedExpertIsStable` | 1 |
| MC-2 | Publish a fill after invalidation increments the generation | `PublishedGenerationIsCurrent` | 3 |
| MC-3 | Omit a miss row before CUDA dispatch accepts it | `SkippedRowsAreBacked` | 2 |
| MC-4 | Fail either prefetch stream after partial work | `CompletionIsTotal` unless CPU fallback completes | 4 |
| MC-5 | Disable after no observed overlap | `DisabledRequiresFallbackOrCompletion` | 5 |
| MC-6 | Restore cache-wide victim selection without quota eligibility | `ProtectedSharesRemain` | 6 |

### 6.2 Test-verifiable

| ID | Description | Suggested test |
| --- | --- | --- |
| TV-1 | CPU/CUDA quantized results remain equivalent | Synthetic quantized experts and NMSE comparison |
| TV-2 | Dispatch/collect/allocation errors restore stock output | Fault injection at every provider stage |
| TV-3 | Invalidation is not a false-positive model artifact | Mutate a live weight while fills are queued/in flight |
| TV-4 | A configured 8-slot pool is not blocked by a legacy 64-slot check | One-MiB small-pool regression |
| TV-5 | Public scheduler API can enable an environment-disabled session | Configure prefill after scheduler creation |
| TV-6 | One layer's expert scan does not evict other layers' fair shares | Four-layer pool: borrow, reclaim, churn, then require immediate hot hits |

### 6.3 Code-review-only

| ID | Description | Suggested action |
| --- | --- | --- |
| CR-1 | Dynamic backend unload is not synchronized with arbitrary concurrent API calls | Retain llama.cpp's backend lifecycle precondition and document it |
| CR-2 | Direct library fields cannot undo weight repacking after load | Keep explicit documentation and consider future model-level params |
| CR-3 | Performance profitability varies by PCIe/UMA topology | Benchmark per target; do not encode as a safety rule |

## 7. Reference pointers

- Implementation report: `analysis-report.md`
- Decode model: `spec/MoECache.tla`
- Prefill model: `spec/MoEPrefetch.tla`
- Admission/partition model: `spec/MoEPartitionPolicy.tla`
- Lean proof: `lean/MoEPartition.lean`
- Prior repository audit: `../../../audit/`
- Persistent-cache reference: local branch `reference/moe-cache-pr`
- Scheduler-prefetch reference: local branch `reference/prefetch`
