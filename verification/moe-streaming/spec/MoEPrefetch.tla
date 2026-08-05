---------------------------- MODULE MoEPrefetch ----------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS ExpertCount, DisableThreshold

Experts == 1..ExpertCount
Slots == 0..1
None == 0
StageStates == {"free", "copying", "ready", "computing"}

VARIABLES stageState, stageExpert, nextExpert, currentExpert, outputs,
          fallbackRequired, disabled, opportunities, fullyHidden,
          completedCorrect

vars == <<stageState, stageExpert, nextExpert, currentExpert, outputs,
          fallbackRequired, disabled, opportunities, fullyHidden,
          completedCorrect>>

Init ==
    /\ stageState = [s \in Slots |-> "free"]
    /\ stageExpert = [s \in Slots |-> None]
    /\ nextExpert = 1
    /\ currentExpert = None
    /\ outputs = {}
    /\ fallbackRequired = FALSE
    /\ disabled = FALSE
    /\ opportunities = 0
    /\ fullyHidden = 0
    /\ completedCorrect = FALSE

\* Select the alternating staging slot and begin the next expert copy:
\* moe-cache.cu:2317-2345, 2408-2414.
StageNext ==
    /\ ~disabled
    /\ ~completedCorrect
    /\ ~fallbackRequired
    /\ nextExpert \in Experts
    /\ LET s == (nextExpert - 1) % 2 IN
       /\ stageState[s] = "free"
       /\ stageState' = [stageState EXCEPT ![s] = "copying"]
       /\ stageExpert' = [stageExpert EXCEPT ![s] = nextExpert]
    /\ nextExpert' = nextExpert + 1
    /\ UNCHANGED <<currentExpert, outputs, fallbackRequired, disabled,
                    opportunities, fullyHidden, completedCorrect>>

\* expert-ready CUDA event publication: moe-cache.cu:2336-2344.
CopyComplete(s) ==
    /\ s \in Slots
    /\ stageState[s] = "copying"
    /\ stageState' = [stageState EXCEPT ![s] = "ready"]
    /\ UNCHANGED <<stageExpert, nextExpert, currentExpert, outputs,
                    fallbackRequired, disabled, opportunities, fullyHidden,
                    completedCorrect>>

\* Compute stream waits on the selected ready event: moe-cache.cu:2357-2363.
StartCompute(s) ==
    /\ s \in Slots
    /\ currentExpert = None
    /\ stageState[s] = "ready"
    /\ stageExpert[s] \in Experts
    /\ stageExpert[s] \notin outputs
    /\ stageState' = [stageState EXCEPT ![s] = "computing"]
    /\ currentExpert' = stageExpert[s]
    /\ UNCHANGED <<stageExpert, nextExpert, outputs, fallbackRequired,
                    disabled, opportunities, fullyHidden, completedCorrect>>

\* Chunked MMVQ, result copy, overlap query, and slot-consumed publication:
\* moe-cache.cu:2365-2445.
FinishCompute(s) ==
    /\ s \in Slots
    /\ stageState[s] = "computing"
    /\ currentExpert = stageExpert[s]
    /\ LET overlap == \E other \in Slots \ {s} :
                           stageState[other] \in {"copying", "ready"}
           hidden == \E other \in Slots \ {s} : stageState[other] = "ready"
           newOutputs == outputs \cup {currentExpert}
       IN
       /\ stageState' = [stageState EXCEPT ![s] = "free"]
       /\ stageExpert' = [stageExpert EXCEPT ![s] = None]
       /\ outputs' = newOutputs
       /\ opportunities' = opportunities + IF overlap THEN 1 ELSE 0
       /\ fullyHidden' = fullyHidden + IF hidden THEN 1 ELSE 0
       /\ completedCorrect' = (completedCorrect \/ newOutputs = Experts)
    /\ currentExpert' = None
    /\ UNCHANGED <<nextExpert, fallbackRequired, disabled>>

\* Any partial pipeline error synchronizes both streams and returns false:
\* moe-cache.cu:2467-2478; ggml-cpu.c:1601-1612 then runs the stock node.
PipelineFailure ==
    /\ ~fallbackRequired
    /\ ~completedCorrect
    /\ (currentExpert # None \/ \E s \in Slots : stageState[s] # "free")
    /\ stageState' = [s \in Slots |-> "free"]
    /\ stageExpert' = [s \in Slots |-> None]
    /\ currentExpert' = None
    /\ fallbackRequired' = TRUE
    /\ UNCHANGED <<nextExpert, outputs, disabled, opportunities,
                    fullyHidden, completedCorrect>>

\* Disable after 32 measured opportunities with no fully hidden upload:
\* moe-cache.cu:2452-2463.
SelfDisable ==
    /\ ~disabled
    /\ ~completedCorrect
    /\ opportunities >= DisableThreshold
    /\ fullyHidden = 0
    /\ currentExpert = None
    /\ disabled' = TRUE
    /\ fallbackRequired' = TRUE
    /\ stageState' = [s \in Slots |-> "free"]
    /\ stageExpert' = [s \in Slots |-> None]
    /\ UNCHANGED <<nextExpert, currentExpert, outputs, opportunities,
                    fullyHidden, completedCorrect>>

\* Provider false means complete-node CPU fallback: ggml-cpu.c:1601-1612,
\* 1652-1795.
CPUFallback ==
    /\ fallbackRequired
    /\ outputs' = Experts
    /\ completedCorrect' = TRUE
    /\ fallbackRequired' = FALSE
    /\ UNCHANGED <<stageState, stageExpert, nextExpert, currentExpert,
                    disabled, opportunities, fullyHidden>>

Done == completedCorrect /\ UNCHANGED vars

Next ==
    \/ StageNext
    \/ \E s \in Slots : CopyComplete(s)
    \/ \E s \in Slots : StartCompute(s)
    \/ \E s \in Slots : FinishCompute(s)
    \/ PipelineFailure
    \/ SelfDisable
    \/ CPUFallback
    \/ Done

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ stageState \in [Slots -> StageStates]
    /\ stageExpert \in [Slots -> (Experts \cup {None})]
    /\ nextExpert \in 1..(ExpertCount + 1)
    /\ currentExpert \in Experts \cup {None}
    /\ outputs \subseteq Experts
    /\ opportunities \in Nat
    /\ fullyHidden \in Nat

ComputeOwnsReadyExpert ==
    currentExpert # None =>
        \E s \in Slots :
            stageState[s] = "computing" /\ stageExpert[s] = currentExpert

NoDuplicateStaging ==
    \A s1, s2 \in Slots :
        s1 # s2 /\ stageState[s1] # "free" /\ stageState[s2] # "free" =>
            stageExpert[s1] # stageExpert[s2]

HiddenTransfersAreOpportunities == fullyHidden <= opportunities

CompletionIsTotal == completedCorrect => outputs = Experts

DisabledRequiresFallbackOrCompletion ==
    disabled => fallbackRequired \/ completedCorrect

=============================================================================
