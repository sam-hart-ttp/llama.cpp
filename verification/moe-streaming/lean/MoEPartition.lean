import Std

/-!
# Functional core of hybrid MoE execution

The implementation partitions routed rows into cache hits (computed on CUDA)
and misses (left for the ordinary CPU `MUL_MAT_ID` kernel).  During prefill it
also groups rows by expert and processes each group in bounded chunks.  These
proofs isolate the two facts on which numerical completeness depends:

* partitioning cannot lose or duplicate a routed contribution;
* grouping and cutting a group into a bounded prefix and remainder preserve
  the same contribution.

`Nat` is used as a free commutative additive model.  The result therefore
captures routing multiplicity without depending on floating-point rounding or
on a particular quantized matrix multiplication implementation.  The final
lemmas capture the arithmetic core of fair-share cache reclamation for the
two-partition bounded model.
-/

namespace MoEStreaming

def routeSum (weight : α → Nat) : List α → Nat
  | []      => 0
  | x :: xs => weight x + routeSum weight xs

theorem routeSum_append (weight : α → Nat) (xs ys : List α) :
    routeSum weight (xs ++ ys) = routeSum weight xs + routeSum weight ys := by
  induction xs with
  | nil => simp [routeSum]
  | cons x xs ih => simp [routeSum, ih, Nat.add_assoc]

def partitionHits (isHit : α → Bool) : List α → List α × List α
  | [] => ([], [])
  | x :: xs =>
      let rest := partitionHits isHit xs
      if isHit x then (x :: rest.1, rest.2) else (rest.1, x :: rest.2)

theorem partitionHits_conserves
    (isHit : α → Bool) (weight : α → Nat) (xs : List α) :
    let parts := partitionHits isHit xs
    routeSum weight xs = routeSum weight parts.1 + routeSum weight parts.2 := by
  induction xs with
  | nil => simp [partitionHits, routeSum]
  | cons x xs ih =>
      simp only [partitionHits]
      split <;>
        simp [routeSum, ih, Nat.add_assoc, Nat.add_left_comm]

def hybridSum (isHit : α → Bool) (weight : α → Nat) (xs : List α) : Nat :=
  let parts := partitionHits isHit xs
  routeSum weight parts.1 + routeSum weight parts.2

theorem gpuHits_plus_cpuMisses_eq_baseline
    (isHit : α → Bool) (weight : α → Nat) (xs : List α) :
    hybridSum isHit weight xs = routeSum weight xs := by
  exact (partitionHits_conserves isHit weight xs).symm

structure Route where
  expert : Nat
  row    : Nat
  value  : Nat
deriving DecidableEq, Repr

def expertGroup (expert : Nat) : List Route → List Route
  | [] => []
  | route :: routes =>
      if route.expert == expert then
        route :: expertGroup expert routes
      else
        expertGroup expert routes

def expertContribution (expert : Nat) : List Route → Nat
  | [] => 0
  | route :: routes =>
      (if route.expert == expert then route.value else 0) +
        expertContribution expert routes

theorem expertGroup_preserves_contribution (expert : Nat) (routes : List Route) :
    routeSum Route.value (expertGroup expert routes) =
      expertContribution expert routes := by
  induction routes with
  | nil => simp [expertGroup, expertContribution, routeSum]
  | cons route routes ih =>
      by_cases h : route.expert = expert <;>
        simp [expertGroup, expertContribution, routeSum, ih, h]

def cut : Nat → List α → List α × List α
  | 0, xs => ([], xs)
  | _ + 1, [] => ([], [])
  | n + 1, x :: xs =>
      let rest := cut n xs
      (x :: rest.1, rest.2)

theorem cut_reassembles (limit : Nat) (xs : List α) :
    (cut limit xs).1 ++ (cut limit xs).2 = xs := by
  induction limit generalizing xs with
  | zero => simp [cut]
  | succ limit ih =>
      cases xs with
      | nil => simp [cut]
      | cons x xs => simp [cut, ih]

theorem cut_conserves
    (limit : Nat) (weight : α → Nat) (xs : List α) :
    routeSum weight (cut limit xs).1 + routeSum weight (cut limit xs).2 =
      routeSum weight xs := by
  rw [← routeSum_append]
  rw [cut_reassembles]

theorem expertChunk_preserves_contribution
    (expert limit : Nat) (routes : List Route) :
    routeSum Route.value (cut limit (expertGroup expert routes)).1 +
        routeSum Route.value (cut limit (expertGroup expert routes)).2 =
      expertContribution expert routes := by
  rw [cut_conserves]
  exact expertGroup_preserves_contribution expert routes

theorem donor_over_quota_of_full
    (quota requester donor : Nat)
    (full : requester + donor = quota + quota)
    (requester_under : requester < quota) :
    quota < donor := by
  omega

theorem reclaim_preserves_donor_quota
    (quota donor : Nat)
    (donor_over : quota < donor) :
    quota ≤ donor - 1 := by
  omega

theorem reclaim_restores_requester_and_preserves_donor
    (quota requester donor : Nat)
    (full : requester + donor = quota + quota)
    (requester_one_short : requester + 1 = quota) :
    requester + 1 = quota ∧ quota ≤ donor - 1 := by
  constructor
  · exact requester_one_short
  · apply reclaim_preserves_donor_quota
    apply donor_over_quota_of_full quota requester donor full
    omega

end MoEStreaming
