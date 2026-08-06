# MoE cache trace instrumentation specification

This is the contract for a future test-only NDJSON harness consumed by
`spec/Trace.tla`. It is deliberately not enabled in normal llama.cpp builds:
logging inside the session mutex or CUDA completion path would perturb the
interleavings and timings the extension is trying to measure.

## 1. Trace event schema

Each event is one JSON object on one line:

```json
{
  "event": "PublishFill",
  "thread": "fill0",
  "start": 1042,
  "end": 1047,
  "expert": "e1",
  "slot": "s1",
  "state": {
    "slotState": {"s1": "valid", "s2": "free"},
    "slotExpert": {"s1": "e1", "s2": "none"},
    "generation": {"s1": 1, "s2": 0},
    "publishedGeneration": {"s1": 1, "s2": -1},
    "readers": {"s1": 0, "s2": 0},
    "jobExpert": {"s1": "none", "s2": "none"},
    "jobGeneration": {"s1": -1, "s2": -1},
    "nodeState": "idle",
    "nodeExpert": "none",
    "pinned": "none",
    "cpuRequired": false,
    "gpuAccepted": false,
    "resultCorrect": false,
    "completedCorrect": true,
    "invalidating": []
  }
}
```

`start` and `end` are readings from one monotonic process-wide counter or clock
placed immediately around the modeled boundary. The thread label is stable for
the trace (`cpu0`, `fill0`, `invalidate0`, and so on). For a compact trace,
instrument only two modeled slots and map encountered cache keys to stable
symbols `e1`, `e2`; reject/stop capture before a third distinct key instead of
silently aliasing it.

The state object is a shadow of the TLA+ variables, not a dump of CUDA pointers.
Capture it while holding `session.mu` when the corresponding action mutates slot
state. CPU obligation fields are test-harness shadow fields updated at the
documented CPU integration boundaries. Every captured field is checked by
`ValidatePostState`; none is informational-only.

## 2. Action-to-code mapping

| Spec action | Code location | Trigger point | Event | Extra fields |
| --- | --- | --- | --- | --- |
| `BeginHit(e,s)` | `ggml/src/ggml-cuda/moe-cache.cu:1851-1861` | Around valid lookup + reader increment, still under `session.mu` | `BeginHit` | `expert`, `slot` |
| `ObserveMiss(e)` | `moe-cache.cu:1862-1901` and `ggml-cpu.c:1676-1703` | After miss is retained as a CPU row | `ObserveMiss` | `expert` |
| `Admit(e,s)` | `moe-cache.cu:1903-1981` | Around slot generation/reservation and successful queue insertion | `Admit` | `expert`, `slot` |
| `CancelFill(s)` | `moe-cache.cu:1297-1318`, called at `2691-2693` | After a copying/queued slot is cancelled under `session.mu` | `CancelFill` | `slot` |
| `PublishFill(s)` | `moe-cache.cu:875-894` | Around the generation/key check and valid publication | `PublishFill` | `slot` |
| `DispatchAccept` | `moe-cache.cu:2145-2153`, `ggml-cpu.c:1705-1721` | After CUDA work is completely accepted, before the CPU worker barrier | `DispatchAccept` | none |
| `DispatchReject` | `ggml-cpu.c:1706-1717` | After all hit mappings have been restored, before the barrier | `DispatchReject` | none |
| `CpuComplete` | `ggml-cpu.c:1736-1795` | After all modeled CPU obligations for the node finish | `CpuComplete` | none |
| `CollectSuccess` | `moe-cache.cu:2177-2197` | After stream synchronization and result copies | `CollectSuccess` | none |
| `CollectFailure` | `ggml-cpu.c:1798-1811` | After every skipped row has been recomputed | `CollectFailure` | none |
| `EndNode` | `moe-cache.cu:2229-2248` | After reader pins and active-source references are released | `EndNode` | none |
| `StartInvalidate(e)` | `moe-cache.cu:2687-2693` | Immediately after queued cancellation, before waiting for readers | `StartInvalidate` | `expert` |
| `FinishInvalidate(e)` | `moe-cache.cu:2695-2783` | After reader/in-flight drain and overlapping slot reset | `FinishInvalidate` | `expert` |

The `CollectFailure` model transition represents “CUDA collection failed, then
the CPU recomputation obligation was restored”; record it after the recompute,
not at the first false return. This keeps the trace event aligned with the
model's post-state.

## 3. Special considerations

- Use a fixed-size, preallocated per-thread trace ring. Do not allocate, take a
  logging mutex, format JSON, or perform I/O inside `session.mu`/CUDA paths.
  Serialize rings to NDJSON after the test scheduler is destroyed.
- Capture `start` just before the state-changing critical boundary and `end`
  immediately after its post-state snapshot. Long CUDA transfer time belongs
  between dispatch and collect events, not inside either event's interval.
- A fill worker event and CPU event may overlap. `Trace.tla`'s `ViableThreads`
  preserves non-overlapping real-time order and lets TLC explore both orders for
  overlaps.
- Backend pointers, CUDA events, LRU links, byte counts, and performance counters
  are excluded because they do not affect the modeled ownership invariants.
- Produce scenario traces from cache hit, dispatch fallback, collection
  fallback, fill invalidation, and concurrent-session tests. Keep each trace to
  50–300 events and validate many short traces rather than one long branchy one.
- Prefill uses a different variable set. If implementation trace validation is
  later required for it, create `PrefetchTrace.tla` rather than weakening the
  decode cache's strong post-state validator.
