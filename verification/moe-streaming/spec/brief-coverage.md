# Modeling-brief coverage audit

This audit was filled from the actual final configuration files, not from the
intended model design.

## Table 1: bug families

| Brief family | Hunt cfg file | Family-relevant invariants enabled | If skipped, why? |
| --- | --- | --- | --- |
| 1. Pin/eviction/generation ownership | `MoECachePinnedEvictionHunt.cfg` | `NoReadersOfUnpublishedSlots`, `PinnedExpertIsStable` | — |
| 2. Hybrid fallback completeness | `MoECacheEarlySkipHunt.cfg` | `SkippedRowsAreBacked`, `CompletedResultsAreCorrect` | — |
| 3. Host invalidation vs async readers | `MoECacheStalePublishHunt.cfg` | `PublishedGenerationIsCurrent` | — |
| 4. Prefetch slot reuse/partial failure | `MoEPrefetchFallbackHunt.cfg` | `ComputeOwnsReadyExpert`, `NoDuplicateStaging`, `CompletionIsTotal` | — |
| 5. Performance disablement | `MoEPrefetchDisableHunt.cfg` | `HiddenTransfersAreOpportunities`, `CompletionIsTotal`, `DisabledRequiresFallbackOrCompletion` | — |
| 6. Cross-layer admission/eviction isolation | `MoEPartitionGlobalLRUHunt.cfg` | `ProtectedSharesRemain` | — |

## Table 2: proposed invariants

The compact models combine base and MC operators in `MoECache.tla` and
`MoEPrefetch.tla`; “wired” therefore means the invariant is defined in the
executable model module and named by a final cfg.

| Brief invariant | Defined at | Wired? | Enabled in hunt cfg(s) | If skipped, why? |
| --- | --- | --- | --- | --- |
| `TypeOK` (cache) | `MoECache.tla:236` | yes | all three cache hunts | — |
| `NoReadersOfUnpublishedSlots` | `MoECache.tla:249` | yes | `MoECachePinnedEvictionHunt.cfg` | — |
| `PublishedGenerationIsCurrent` | `MoECache.tla:252` | yes | `MoECacheStalePublishHunt.cfg` | — |
| `PinnedExpertIsStable` | `MoECache.tla:256` | yes | `MoECachePinnedEvictionHunt.cfg` | — |
| `SkippedRowsAreBacked` | `MoECache.tla:262` | yes | `MoECacheEarlySkipHunt.cfg` | — |
| `CompletedResultsAreCorrect` | `MoECache.tla:269` | yes | `MoECacheEarlySkipHunt.cfg` | — |
| `TypeOK` (prefetch) | `MoEPrefetch.tla:139` | yes | both prefetch hunts | — |
| `ComputeOwnsReadyExpert` | `MoEPrefetch.tla:148` | yes | `MoEPrefetchFallbackHunt.cfg` | — |
| `NoDuplicateStaging` | `MoEPrefetch.tla:153` | yes | `MoEPrefetchFallbackHunt.cfg` | — |
| `HiddenTransfersAreOpportunities` | `MoEPrefetch.tla:158` | yes | `MoEPrefetchDisableHunt.cfg` | — |
| `CompletionIsTotal` | `MoEPrefetch.tla:160` | yes | both prefetch hunts | — |
| `DisabledRequiresFallbackOrCompletion` | `MoEPrefetch.tla:162` | yes | `MoEPrefetchDisableHunt.cfg` | — |
| `CapacityAccounting` | `MoEPartitionPolicy.tla:113` | yes | `MoEPartitionPolicy.cfg`, `MoEPartitionGlobalLRUHunt.cfg` | — |
| `ProtectedSharesRemain` | `MoEPartitionPolicy.tla:121` | yes | `MoEPartitionPolicy.cfg`, `MoEPartitionGlobalLRUHunt.cfg` | — |
| `UnderQuotaHasDonor` | `MoEPartitionPolicy.tla:127` | yes | `MoEPartitionPolicy.cfg` | — |
| `ReadyDemandCanProgress` | `MoEPartitionPolicy.tla:134` | yes | `MoEPartitionPolicy.cfg` | — |

## Table 3: model-checkable findings

| Finding ID | Trigger mechanism | Expected violated invariant | Hunt cfg |
| --- | --- | --- | --- |
| MC-1 | `AllowPinnedEviction=TRUE` permits reserve/reassign under a reader | `PinnedExpertIsStable` | `MoECachePinnedEvictionHunt.cfg` |
| MC-2 | `AllowStalePublish=TRUE` retains a cancelled job and permits old-generation publication | `PublishedGenerationIsCurrent` | `MoECacheStalePublishHunt.cfg` |
| MC-3 | `AllowEarlySkip=TRUE` permits dispatch acceptance for an unpinned miss | `SkippedRowsAreBacked` | `MoECacheEarlySkipHunt.cfg` |
| MC-4 | Nondeterministic `PipelineFailure` after partial staging | `CompletionIsTotal` unless `CPUFallback` completes | `MoEPrefetchFallbackHunt.cfg` |
| MC-5 | `SelfDisable` after an overlap opportunity with no hidden transfer | `DisabledRequiresFallbackOrCompletion` | `MoEPrefetchDisableHunt.cfg` |
| MC-6 | `AllowGlobalEviction=TRUE` permits an at-target layer to select another layer's protected slot | `ProtectedSharesRemain` | `MoEPartitionGlobalLRUHunt.cfg` |

## Coverage summary

Families: **6/6 covered**. Proposed safety/structural invariants: **16/16
enabled in at least one final cfg**. Model-checkable findings: **6/6 targeted**.
Hunt cfg files: **6**. The four cache/policy hunts deliberately mutate a guard
and are expected to fail with their named invariant; the two prefetch hunts
explore real failure/disable actions and are expected to pass.
