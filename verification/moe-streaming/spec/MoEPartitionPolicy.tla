----------------------- MODULE MoEPartitionPolicy -----------------------
EXTENDS Naturals, FiniteSets

CONSTANTS Partitions, Slots, Empty, AdmitAfter, ReadmitAfter,
          AllowGlobalEviction

ASSUME /\ Partitions # {}
       /\ Slots # {}
       /\ Empty \notin Partitions
       /\ AdmitAfter \in Nat \ {0}
       /\ ReadmitAfter \in Nat \ {0}
       /\ AdmitAfter <= ReadmitAfter
       /\ Cardinality(Slots) >= Cardinality(Partitions)
       /\ Cardinality(Slots) % Cardinality(Partitions) = 0

FairQuota == Cardinality(Slots) \div Cardinality(Partitions)

VARIABLES owner, epoch, demand, protected

vars == <<owner, epoch, demand, protected>>

ResidentCount(partition, owners) ==
    Cardinality({slot \in Slots : owners[slot] = partition})

FreeSlots(owners) == {slot \in Slots : owners[slot] = Empty}

Full(owners) == FreeSlots(owners) = {}

Threshold(partition) ==
    IF ~Full(owner) \/ ResidentCount(partition, owner) < FairQuota
    THEN AdmitAfter
    ELSE ReadmitAfter

Ready(partition) == demand[partition] + 1 >= Threshold(partition)

Bump(value) == (value + 1) % 3

ProtectedAfter(owners) ==
    protected \cup
        {partition \in Partitions :
            ResidentCount(partition, owners) >= FairQuota}

Init ==
    /\ owner = [slot \in Slots |-> Empty]
    /\ epoch = [slot \in Slots |-> 0]
    /\ demand = [partition \in Partitions |-> 0]
    /\ protected = {}

RecordMiss(partition) ==
    /\ demand[partition] + 1 < Threshold(partition)
    /\ demand' = [demand EXCEPT ![partition] = @ + 1]
    /\ UNCHANGED <<owner, epoch, protected>>

AdmitInto(partition, slot) ==
    LET nextOwner == [owner EXCEPT ![slot] = partition]
    IN  /\ owner' = nextOwner
        /\ epoch' = [epoch EXCEPT ![slot] = Bump(@)]
        /\ demand' = [demand EXCEPT ![partition] = 0]
        /\ protected' = ProtectedAfter(nextOwner)

AdmitFree(partition) ==
    /\ Ready(partition)
    /\ ~Full(owner)
    /\ \E slot \in FreeSlots(owner) : AdmitInto(partition, slot)

AdmitLocal(partition) ==
    /\ Ready(partition)
    /\ Full(owner)
    /\ ResidentCount(partition, owner) >= FairQuota
    /\ \E slot \in Slots :
        /\ owner[slot] = partition
        /\ AdmitInto(partition, slot)

AdmitReclaim(partition) ==
    /\ Ready(partition)
    /\ Full(owner)
    /\ \E slot \in Slots :
        /\ owner[slot] # partition
        /\ owner[slot] # Empty
        /\ ResidentCount(owner[slot], owner) > FairQuota
        /\ AdmitInto(partition, slot)

\* Mutation used by the hunt configuration.  It represents a cache-wide LRU
\* victim chosen without the layer/quota eligibility filters.
AdmitGlobalBug(partition) ==
    /\ AllowGlobalEviction
    /\ Ready(partition)
    /\ Full(owner)
    /\ \E slot \in Slots :
        /\ owner[slot] # partition
        /\ owner[slot] # Empty
        /\ AdmitInto(partition, slot)

Demand(partition) ==
    \/ RecordMiss(partition)
    \/ AdmitFree(partition)
    \/ AdmitLocal(partition)
    \/ AdmitReclaim(partition)
    \/ AdmitGlobalBug(partition)

Next == \E partition \in Partitions : Demand(partition)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ owner \in [Slots -> Partitions \cup {Empty}]
    /\ epoch \in [Slots -> 0..2]
    /\ demand \in [Partitions -> 0..(ReadmitAfter - 1)]
    /\ protected \subseteq Partitions

CapacityAccounting ==
    Cardinality(FreeSlots(owner)) +
        Cardinality({slot \in Slots : owner[slot] # Empty}) =
    Cardinality(Slots)

\* Once a layer has obtained its fair share, demand from other layers cannot
\* push it below that share.  Local replacement preserves its resident count;
\* cross-layer reclamation is legal only from a donor strictly over quota.
ProtectedSharesRemain ==
    \A partition \in protected :
        ResidentCount(partition, owner) >= FairQuota

\* The arithmetic fact on which reclamation relies: in a full, exactly
\* partitioned pool, an under-quota requester implies an over-quota donor.
UnderQuotaHasDonor ==
    \A partition \in Partitions :
        Full(owner) /\ ResidentCount(partition, owner) < FairQuota
        => \E donor \in Partitions :
            ResidentCount(donor, owner) > FairQuota

\* Admission thresholds must not create a policy deadlock.
ReadyDemandCanProgress ==
    \A partition \in Partitions :
        Ready(partition) =>
            ENABLED (AdmitFree(partition) \/
                     AdmitLocal(partition) \/
                     AdmitReclaim(partition) \/
                     AdmitGlobalBug(partition))

=============================================================================
