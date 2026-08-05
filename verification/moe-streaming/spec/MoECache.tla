----------------------------- MODULE MoECache -----------------------------
EXTENDS Naturals, Integers, FiniteSets

CONSTANTS Slots, Experts, MaxGeneration,
          AllowPinnedEviction, AllowStalePublish, AllowEarlySkip

None == "none"
SlotStates == {"free", "copying", "valid"}
NodeStates == {"idle", "plannedHit", "plannedMiss", "gpu", "cpu", "result"}

VARIABLES slotState, slotExpert, generation, publishedGeneration, readers,
          jobExpert, jobGeneration, nodeState, nodeExpert, pinned,
          cpuRequired, gpuAccepted, resultCorrect, completedCorrect,
          invalidating

vars == <<slotState, slotExpert, generation, publishedGeneration, readers,
          jobExpert, jobGeneration, nodeState, nodeExpert, pinned,
          cpuRequired, gpuAccepted, resultCorrect, completedCorrect,
          invalidating>>

Init ==
    /\ slotState = [s \in Slots |-> "free"]
    /\ slotExpert = [s \in Slots |-> None]
    /\ generation = [s \in Slots |-> 0]
    /\ publishedGeneration = [s \in Slots |-> -1]
    /\ readers = [s \in Slots |-> 0]
    /\ jobExpert = [s \in Slots |-> None]
    /\ jobGeneration = [s \in Slots |-> -1]
    /\ nodeState = "idle"
    /\ nodeExpert = None
    /\ pinned = None
    /\ cpuRequired = FALSE
    /\ gpuAccepted = FALSE
    /\ resultCorrect = FALSE
    /\ completedCorrect = TRUE
    /\ invalidating = {}

\* Cache lookup and reader pin: moe-cache.cu:1660-1703.
BeginHit(e, s) ==
    /\ nodeState = "idle"
    /\ e \in Experts \ invalidating
    /\ s \in Slots
    /\ slotState[s] = "valid"
    /\ slotExpert[s] = e
    /\ nodeState' = "plannedHit"
    /\ nodeExpert' = e
    /\ pinned' = s
    /\ readers' = [readers EXCEPT ![s] = @ + 1]
    /\ cpuRequired' = TRUE
    /\ gpuAccepted' = FALSE
    /\ resultCorrect' = FALSE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    jobExpert, jobGeneration, completedCorrect, invalidating>>

\* Miss remains a CPU obligation: moe-cache.cu:1682-1724 and
\* ggml-cpu.c:1675-1702.
ObserveMiss(e) ==
    /\ nodeState = "idle"
    /\ e \in Experts \ invalidating
    /\ nodeState' = "plannedMiss"
    /\ nodeExpert' = e
    /\ pinned' = None
    /\ cpuRequired' = TRUE
    /\ gpuAccepted' = FALSE
    /\ resultCorrect' = FALSE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, completedCorrect,
                    invalidating>>

\* Reserve/evict, increment generation, then enqueue: moe-cache.cu:1725-1777.
Admit(e, s) ==
    /\ e \in Experts \ invalidating
    /\ s \in Slots
    /\ slotState[s] \in {"free", "valid"}
    /\ readers[s] = 0 \/ AllowPinnedEviction
    /\ generation[s] < MaxGeneration
    /\ slotState' = [slotState EXCEPT ![s] = "copying"]
    /\ slotExpert' = [slotExpert EXCEPT ![s] = e]
    /\ generation' = [generation EXCEPT ![s] = @ + 1]
    /\ publishedGeneration' = [publishedGeneration EXCEPT ![s] = -1]
    /\ readers' = [readers EXCEPT ![s] = 0]
    /\ jobExpert' = [jobExpert EXCEPT ![s] = e]
    /\ jobGeneration' = [jobGeneration EXCEPT ![s] = generation'[s]]
    /\ UNCHANGED <<nodeState, nodeExpert, pinned, cpuRequired, gpuAccepted,
                    resultCorrect, completedCorrect, invalidating>>

\* Invalidation cancels queued/copying state: moe-cache.cu:2482-2521.
CancelFill(s) ==
    /\ s \in Slots
    /\ slotState[s] = "copying"
    /\ generation[s] < MaxGeneration
    /\ slotState' = [slotState EXCEPT ![s] = "free"]
    /\ slotExpert' = [slotExpert EXCEPT ![s] = None]
    /\ generation' = [generation EXCEPT ![s] = @ + 1]
    /\ publishedGeneration' = [publishedGeneration EXCEPT ![s] = -1]
    /\ readers' = [readers EXCEPT ![s] = 0]
    /\ jobExpert' = IF AllowStalePublish THEN jobExpert
                    ELSE [jobExpert EXCEPT ![s] = None]
    /\ jobGeneration' = IF AllowStalePublish THEN jobGeneration
                        ELSE [jobGeneration EXCEPT ![s] = -1]
    /\ UNCHANGED <<nodeState, nodeExpert, pinned, cpuRequired, gpuAccepted,
                    resultCorrect, completedCorrect, invalidating>>

\* Worker publication checks state, key, and generation under session->mu:
\* moe-cache.cu:749-770.
PublishFill(s) ==
    /\ s \in Slots
    /\ jobExpert[s] \in Experts \ invalidating
    /\ (slotState[s] = "copying" \/ AllowStalePublish)
    /\ (jobGeneration[s] = generation[s] \/ AllowStalePublish)
    /\ slotState' = [slotState EXCEPT ![s] = "valid"]
    /\ slotExpert' = [slotExpert EXCEPT ![s] = jobExpert[s]]
    /\ publishedGeneration' = [publishedGeneration EXCEPT ![s] = jobGeneration[s]]
    /\ jobExpert' = [jobExpert EXCEPT ![s] = None]
    /\ jobGeneration' = [jobGeneration EXCEPT ![s] = -1]
    /\ UNCHANGED <<generation, readers, nodeState, nodeExpert, pinned,
                    cpuRequired, gpuAccepted, resultCorrect,
                    completedCorrect, invalidating>>

\* CPU rows are removed only after full CUDA acceptance:
\* moe-cache.cu:1786-1948 and ggml-cpu.c:1705-1717.
DispatchAccept ==
    /\ nodeState = "plannedHit" \/
       (AllowEarlySkip /\ nodeState = "plannedMiss")
    /\ nodeState' = "gpu"
    /\ cpuRequired' = FALSE
    /\ gpuAccepted' = TRUE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeExpert, pinned,
                    resultCorrect, completedCorrect, invalidating>>

\* Reinsert all planned rows before the worker barrier: ggml-cpu.c:1705-1717.
DispatchReject ==
    /\ nodeState = "plannedHit"
    /\ nodeState' = "cpu"
    /\ cpuRequired' = TRUE
    /\ gpuAccepted' = FALSE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeExpert, pinned,
                    resultCorrect, completedCorrect, invalidating>>

\* Stock CPU miss/fallback execution: ggml-cpu.c:1730-1795.
CpuComplete ==
    /\ nodeState \in {"plannedMiss", "cpu"}
    /\ cpuRequired
    /\ nodeState' = "result"
    /\ resultCorrect' = TRUE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeExpert, pinned,
                    cpuRequired, gpuAccepted, completedCorrect, invalidating>>

\* Synchronized result download: moe-cache.cu:1951-2005.
CollectSuccess ==
    /\ nodeState = "gpu"
    /\ pinned \in Slots
    /\ slotState[pinned] = "valid"
    /\ slotExpert[pinned] = nodeExpert
    /\ nodeState' = "result"
    /\ resultCorrect' = TRUE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeExpert, pinned,
                    cpuRequired, gpuAccepted, completedCorrect, invalidating>>

\* Collection failure restores the CPU obligation: ggml-cpu.c:1797-1811.
CollectFailure ==
    /\ nodeState = "gpu"
    /\ nodeState' = "cpu"
    /\ cpuRequired' = TRUE
    /\ gpuAccepted' = FALSE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeExpert, pinned,
                    resultCorrect, completedCorrect, invalidating>>

\* Release every reader pin and active-source reference: moe-cache.cu:2008-2047.
EndNode ==
    /\ nodeState = "result"
    /\ nodeState' = "idle"
    /\ nodeExpert' = None
    /\ readers' = IF pinned \in Slots
                  THEN [readers EXCEPT ![pinned] = @ - 1]
                  ELSE readers
    /\ pinned' = None
    /\ cpuRequired' = FALSE
    /\ gpuAccepted' = FALSE
    /\ completedCorrect' = completedCorrect /\ resultCorrect
    /\ resultCorrect' = FALSE
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    jobExpert, jobGeneration, invalidating>>

\* Public write notification starts while source lifetime is still valid:
\* ggml-backend.cpp:94-103 and moe-cache.cu:2482-2505.
StartInvalidate(e) ==
    /\ e \in Experts \ invalidating
    /\ invalidating' = invalidating \cup {e}
    /\ UNCHANGED <<slotState, slotExpert, generation, publishedGeneration,
                    readers, jobExpert, jobGeneration, nodeState, nodeExpert,
                    pinned, cpuRequired, gpuAccepted, resultCorrect,
                    completedCorrect>>

\* Wait for readers/in-flight source copies, then erase overlapping state:
\* moe-cache.cu:2490-2541.
FinishInvalidate(e) ==
    /\ e \in invalidating
    /\ \A s \in Slots : slotExpert[s] = e => readers[s] = 0
    /\ ~(pinned \in Slots /\ nodeExpert = e)
    /\ slotState' = [s \in Slots |-> IF slotExpert[s] = e THEN "free" ELSE slotState[s]]
    /\ slotExpert' = [s \in Slots |-> IF slotExpert[s] = e THEN None ELSE slotExpert[s]]
    /\ generation' = [s \in Slots |->
            IF slotExpert[s] = e /\ generation[s] < MaxGeneration
            THEN generation[s] + 1 ELSE generation[s]]
    /\ publishedGeneration' = [s \in Slots |->
            IF slotExpert[s] = e THEN -1 ELSE publishedGeneration[s]]
    /\ jobExpert' = [s \in Slots |-> IF jobExpert[s] = e THEN None ELSE jobExpert[s]]
    /\ jobGeneration' = [s \in Slots |-> IF jobExpert[s] = e THEN -1 ELSE jobGeneration[s]]
    /\ invalidating' = invalidating \ {e}
    /\ UNCHANGED <<readers, nodeState, nodeExpert, pinned, cpuRequired,
                    gpuAccepted, resultCorrect, completedCorrect>>

Next ==
    \/ \E e \in Experts, s \in Slots : BeginHit(e, s)
    \/ \E e \in Experts : ObserveMiss(e)
    \/ \E e \in Experts, s \in Slots : Admit(e, s)
    \/ \E s \in Slots : CancelFill(s)
    \/ \E s \in Slots : PublishFill(s)
    \/ DispatchAccept
    \/ DispatchReject
    \/ CpuComplete
    \/ CollectSuccess
    \/ CollectFailure
    \/ EndNode
    \/ \E e \in Experts : StartInvalidate(e)
    \/ \E e \in Experts : FinishInvalidate(e)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ slotState \in [Slots -> SlotStates]
    /\ slotExpert \in [Slots -> (Experts \cup {None})]
    /\ generation \in [Slots -> 0..MaxGeneration]
    /\ publishedGeneration \in [Slots -> (-1..MaxGeneration)]
    /\ readers \in [Slots -> Nat]
    /\ jobExpert \in [Slots -> (Experts \cup {None})]
    /\ jobGeneration \in [Slots -> (-1..MaxGeneration)]
    /\ nodeState \in NodeStates
    /\ nodeExpert \in Experts \cup {None}
    /\ pinned \in Slots \cup {None}
    /\ invalidating \subseteq Experts

NoReadersOfUnpublishedSlots ==
    \A s \in Slots : readers[s] > 0 => slotState[s] = "valid"

PublishedGenerationIsCurrent ==
    \A s \in Slots : slotState[s] = "valid" =>
        publishedGeneration[s] = generation[s]

PinnedExpertIsStable ==
    pinned \in Slots =>
        /\ readers[pinned] > 0
        /\ slotState[pinned] = "valid"
        /\ slotExpert[pinned] = nodeExpert

SkippedRowsAreBacked ==
    ~cpuRequired =>
        nodeState \in {"idle", "gpu", "result"} /\
        (nodeState = "gpu" =>
            gpuAccepted /\ pinned \in Slots /\
            slotState[pinned] = "valid" /\ slotExpert[pinned] = nodeExpert)

CompletedResultsAreCorrect == completedCorrect

=============================================================================
