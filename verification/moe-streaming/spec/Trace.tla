------------------------------- MODULE Trace -------------------------------
EXTENDS MoECache, Json, IOUtils, Sequences, TLC

\* Category-B timebox trace validator. Each NDJSON event contains a tight
\* [start,end] interval around one modeled atomic boundary. TLC preserves
\* happens-before between disjoint intervals and explores both orders for
\* overlapping intervals.

CONSTANT TraceThreads

JsonFile ==
    IF "JSON" \in DOMAIN IOEnv THEN IOEnv.JSON
    ELSE "../traces/trace.ndjson"

TraceLog == ndJsonDeserialize(JsonFile)

IsThreadEvent(event, thread) == event.thread = thread

ThreadEvents(thread) ==
    SelectSeq(TraceLog, LAMBDA event : IsThreadEvent(event, thread))

traces == [thread \in TraceThreads |-> ThreadEvents(thread)]

VARIABLE pc
traceVars == <<pc>>

ThreadsWithEvents ==
    {thread \in TraceThreads : pc[thread] <= Len(traces[thread])}

NextEvent(thread) == traces[thread][pc[thread]]

ViableThreads ==
    {thread \in ThreadsWithEvents :
        ~\E other \in ThreadsWithEvents :
            /\ other # thread
            /\ NextEvent(other).end < NextEvent(thread).start}

SeqSet(seq) == {seq[index] : index \in DOMAIN seq}

\* Strong post-state validation. The instrumentation contract captures every
\* variable in the compact cache model immediately after the action boundary.
ValidatePostState(event) ==
    /\ slotState' = event.state.slotState
    /\ slotExpert' = event.state.slotExpert
    /\ generation' = event.state.generation
    /\ publishedGeneration' = event.state.publishedGeneration
    /\ readers' = event.state.readers
    /\ jobExpert' = event.state.jobExpert
    /\ jobGeneration' = event.state.jobGeneration
    /\ nodeState' = event.state.nodeState
    /\ nodeExpert' = event.state.nodeExpert
    /\ pinned' = event.state.pinned
    /\ cpuRequired' = event.state.cpuRequired
    /\ gpuAccepted' = event.state.gpuAccepted
    /\ resultCorrect' = event.state.resultCorrect
    /\ completedCorrect' = event.state.completedCorrect
    /\ invalidating' = SeqSet(event.state.invalidating)

MatchEvent(event) ==
    \/ /\ event.event = "BeginHit"
       /\ BeginHit(event.expert, event.slot)
       /\ ValidatePostState(event)
    \/ /\ event.event = "ObserveMiss"
       /\ ObserveMiss(event.expert)
       /\ ValidatePostState(event)
    \/ /\ event.event = "Admit"
       /\ Admit(event.expert, event.slot)
       /\ ValidatePostState(event)
    \/ /\ event.event = "CancelFill"
       /\ CancelFill(event.slot)
       /\ ValidatePostState(event)
    \/ /\ event.event = "PublishFill"
       /\ PublishFill(event.slot)
       /\ ValidatePostState(event)
    \/ /\ event.event = "DispatchAccept"
       /\ DispatchAccept
       /\ ValidatePostState(event)
    \/ /\ event.event = "DispatchReject"
       /\ DispatchReject
       /\ ValidatePostState(event)
    \/ /\ event.event = "CpuComplete"
       /\ CpuComplete
       /\ ValidatePostState(event)
    \/ /\ event.event = "CollectSuccess"
       /\ CollectSuccess
       /\ ValidatePostState(event)
    \/ /\ event.event = "CollectFailure"
       /\ CollectFailure
       /\ ValidatePostState(event)
    \/ /\ event.event = "EndNode"
       /\ EndNode
       /\ ValidatePostState(event)
    \/ /\ event.event = "StartInvalidate"
       /\ StartInvalidate(event.expert)
       /\ ValidatePostState(event)
    \/ /\ event.event = "FinishInvalidate"
       /\ FinishInvalidate(event.expert)
       /\ ValidatePostState(event)

TraceInit ==
    /\ Init
    /\ pc = [thread \in TraceThreads |-> 1]

TraceNext ==
    \/ /\ ThreadsWithEvents # {}
       /\ \E thread \in ViableThreads :
            /\ MatchEvent(NextEvent(thread))
            /\ pc' = [pc EXCEPT ![thread] = @ + 1]
    \/ /\ ThreadsWithEvents = {}
       /\ UNCHANGED <<vars, pc>>

TraceSpec == TraceInit /\ [][TraceNext]_<<vars, pc>>

TraceFullyConsumed == <>(ThreadsWithEvents = {})

=============================================================================
