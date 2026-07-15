/-!
# hermes-edge — formal model of LONG-RUN stability

`EdgeAudit.lean` certifies single-turn arithmetic (compaction trigger,
deadline floors, gate table). It says nothing about what happens over a
multi-day session — exactly the gap its own item 18 admits ("multi-day soak
not yet run"). This file closes that gap by modelling a session as a
discrete dynamical system over turns and proving the condition under which
per-turn prefill cost stays bounded by a constant independent of session
length. The alternative — cost that grows with accumulated context — is the
failure reproduced live: a 93,551-token resident window whose checkpoint
miss re-prefills ALL of it (~29 min at 54 tok/s), with decode collapsed to
7.57 tok/s.

**Scope.** As in `EdgeAudit.lean`: these theorems certify the *design*
(the cost recurrence and its invariants), not the Python runtime or the
llama.cpp scheduler. Concrete `example`s are kernel-checked against the
numbers observed in the live run so model and measurement are cross-anchored.
Core Lean 4 v4.29.0, no mathlib; `lean EdgeStability.lean` must exit 0.

The three results that drive the redesign:

* **§2 Exclusivity.** A conversation that shares its server slot with ANY
  other traffic can be evicted and forced into a full re-prefill. Bounded
  per-turn cost requires *exclusive, persistent* slot tenancy — not a
  best-effort `cache_key` pin (never honored in the observed logs: every
  slot selection was `by LRU` / `by LCP`, never `by cache_key`).
* **§3 Bounded window.** Even a cold/miss turn is bounded iff the resident
  window is bounded. This is the working-set budget, re-justified below.
* **§4 Demote ≠ delete.** Bounding the window loses information ONLY when
  there is no archive. With a cold tier (session_search) the accessible
  information is invariant under compaction — so the budget the operator
  once rejected as "brain damage" is, with the archive present, lossless
  relocation. This is why the fix is *distribute*, not *cap*.
-/

namespace EdgeStability

/-! ## §1 Per-turn prefill cost model

A turn either restarts from a live checkpoint of its own prefix (cost =
the incremental delta since that checkpoint) or, on a checkpoint miss,
re-prefills the entire resident window. On hybrid/SWA attention a cross-slot
or evicted state cannot be restored — the miss is a genuine full re-prefill
(`forcing full prompt re-processing due to lack of cache data` in the logs).
-/

/-- Prefill cost: `delta` on a checkpoint hit, the full `window` on a miss. -/
def prefillCost (hit : Bool) (delta window : Nat) : Nat :=
  match hit with
  | true  => delta
  | false => window

theorem hit_cost_incremental (delta window : Nat) :
    prefillCost true delta window = delta := rfl

theorem miss_cost_full (delta window : Nat) :
    prefillCost false delta window = window := rfl

/-- A miss costs more the larger the resident window: unbounded context ⇒
unbounded miss cost. -/
theorem miss_cost_mono (delta w₁ w₂ : Nat) (h : w₁ ≤ w₂) :
    prefillCost false delta w₁ ≤ prefillCost false delta w₂ := by
  show w₁ ≤ w₂
  exact h

/-- Milliseconds to prefill `tokens` at `tps` tokens/second (shared with
`EdgeAudit.prefillMs`; redefined so this file stands alone). -/
def prefillMs (tokens tps : Nat) : Nat := tokens * 1000 / tps

/-! ## §2 Slot tenancy — the eviction that forces a full re-prefill

Model the server slots as the conversation id each slot currently holds.
A serve of conversation `c` is a *hit* iff a slot already holds `c`; a miss
otherwise (its state must be rebuilt). We show:

* One slot serving ONE conversation → steady-state hits (cost = Δ).
* One slot with ANY interleaved second conversation → a reachable miss.
* Two slots under LRU with a third context → a reachable miss (the observed
  main/aux/subagent thrash).

The invariant that actually matters is therefore *exclusive tenancy*, not
slot count: bounded per-turn cost needs the main conversation to own a slot
that no other traffic ever touches.
-/

/-- Single slot: `resident` is the conversation it holds. Serving `c`
overwrites it and hits iff it already held `c`. -/
def serve1 (resident c : Nat) : Nat × Bool := (c, resident == c)

/-- Exclusive tenancy: the same conversation (id 1) served repeatedly on one
slot. Only the cold first turn misses; every later turn hits. Concrete
schedule `1,1,1` → last serve hits. -/
def exclusiveFinalHit : Bool :=
  let r₁ := serve1 0 1
  let r₂ := serve1 r₁.1 1
  let r₃ := serve1 r₂.1 1
  r₃.2

theorem exclusive_tenancy_hits : exclusiveFinalHit = true := rfl

/-- Contention on a single slot: interleave conversation 2 between two turns
of conversation 1 (`1,2,1`). Conversation 1's return turn MISSES — one slot
is not enough; the slot must be exclusive. This is why `--parallel 1` alone
is insufficient if auxiliary traffic shares the endpoint. -/
def interleavedFinalHit : Bool :=
  let r₁ := serve1 0 1
  let r₂ := serve1 r₁.1 2
  let r₃ := serve1 r₂.1 1
  r₃.2

theorem interleaving_forces_miss : interleavedFinalHit = false := rfl

/-- Two slots holding conversation ids, with an LRU eviction target. -/
structure Two where
  s0 : Nat
  s1 : Nat
  lruIsS0 : Bool

/-- Serve `c`: hit if either slot holds it (flip LRU to the other); on a
miss, evict the LRU slot and load `c` there. -/
def serveTwo (st : Two) (c : Nat) : Two × Bool :=
  if st.s0 = c then ({ st with lruIsS0 := false }, true)
  else if st.s1 = c then ({ st with lruIsS0 := true }, true)
  else if st.lruIsS0 = true then ({ s0 := c, s1 := st.s1, lruIsS0 := false }, false)
  else ({ s0 := st.s0, s1 := c, lruIsS0 := true }, false)

/-- Two slots do NOT save you once a third context appears. Schedule
`1,2,3,1`: conversation 1 is evicted by 3 and its return turn (turn 4)
misses — a full re-prefill of the whole resident window. This is the
observed main(93K)/aux/subagent thrash: `--parallel 2` without a *honored*
pin is not stable. -/
def thrashFinalHit : Bool :=
  let st₀ : Two := ⟨0, 0, true⟩
  let r₁ := serveTwo st₀ 1
  let r₂ := serveTwo r₁.1 2
  let r₃ := serveTwo r₂.1 3
  let r₄ := serveTwo r₃.1 1
  r₄.2

theorem two_slots_thrash : thrashFinalHit = false := rfl

/-! ## §3 The stability theorem

Combine §1 and §2. Under **exclusive tenancy** every warm turn hits, so its
cost is the incremental delta Δ. Under a **bounded window** (working-set
budget `B`, `EdgeAudit.trigger_le_budget`) the resident window never exceeds
`B`, so even a cold or miss turn costs at most `B`. Hence with `Δ ≤ B`,
EVERY turn — cold, warm, or miss — costs at most `B`, a constant independent
of the turn index. Remove either hypothesis and the bound is a growing
window (unbounded budget) or a reachable full re-prefill (non-exclusive
tenancy).
-/

/-- Headline bound: per-turn prefill cost ≤ the working-set budget `B` for
every turn, given the resident window is bounded (`window ≤ B`) and the warm
delta is bounded (`delta ≤ B`). -/
theorem per_turn_cost_le_budget
    (hit : Bool) (delta window B : Nat)
    (hwin : window ≤ B) (hdelta : delta ≤ B) :
    prefillCost hit delta window ≤ B := by
  cases hit with
  | true  => exact hdelta
  | false => exact hwin

/-- The stable operating point: a cold turn pays the full bounded window,
a warm turn pays only Δ. -/
def stableCost (isCold : Bool) (delta B : Nat) : Nat :=
  if isCold = true then B else delta

theorem stable_cost_bounded (isCold : Bool) (delta B : Nat) (h : delta ≤ B) :
    stableCost isCold delta B ≤ B := by
  unfold stableCost
  split
  · exact Nat.le_refl B
  · exact h

/-! ### Anchored to the live run

The unstable run rode the window to 93,551 tokens; a checkpoint miss
re-prefills all of it, and at the run's measured 54 tok/s that is
~1,732,425 ms ≈ 28.9 minutes — per turn, recurring. Bounding the working
set to 32,768 tokens caps the worst miss at ~10.1 minutes, and exclusive
tenancy makes warm turns pay only Δ instead. -/
example : prefillCost false 200 93551 = 93551 := rfl
example : prefillMs 93551 54 = 1732425 := rfl
example : (1500000 : Nat) < prefillMs 93551 54 := by decide      -- > 25 min, per turn
example : prefillMs 32768 54 = 606814 := rfl
example : prefillMs 32768 54 < prefillMs 93551 54 := by decide   -- bounding wins
/-- With exclusive tenancy the warm turn re-prefills only the delta, e.g. a
2,000-token turn ≈ 37 s at 54 tok/s — two orders of magnitude below the
93,551-token miss it replaces. -/
example : prefillMs (prefillCost true 2000 93551) 54 = 37037 := rfl

/-! ## §4 Demote ≠ delete — why the budget is not brain damage

The operator's objection to a working-set budget was that capping context
destroys reasoning. That is true only when compaction DELETES. Split the
conversation's information into `resident` (in-window) and `archived`
(recoverable via session_search). Compaction with an archive *moves* tokens
resident→archived; the total accessible information is invariant. Without an
archive it is a strict loss. So the budget is safe precisely when the cold
tier is enabled — the two changes are a package, not alternatives.
-/

/-- Total accessible information: what is in the window plus what is
retrievable from the archive. -/
def accessible (resident archived : Nat) : Nat := resident + archived

/-- Compaction WITH an archive relocates `d` tokens resident→archived. The
accessible total is unchanged: information is demoted, not deleted. -/
theorem archive_preserves_accessible (r a d : Nat) (h : d ≤ r) :
    accessible (r - d) (a + d) = accessible r a := by
  unfold accessible; omega

/-- Compaction WITHOUT an archive (delete) strictly reduces accessible
information — the failure mode the operator rightly rejected. -/
theorem delete_loses_information (r d : Nat) (hd : 0 < d) (h : d ≤ r) :
    (r - d) < accessible r 0 := by
  unfold accessible; omega

/-! ## §5 Derived design requirements (fed back to the edge profile)

The theorems above are satisfied by exactly the following configuration —
each is a *requirement*, not a tuning choice:

* **R1 — exclusive tenancy** (`exclusive_tenancy_hits`,
  `interleaving_forces_miss`, `two_slots_thrash`): the main conversation must
  own a server slot no other traffic touches. Concretely: `--parallel 1` on
  the main-model server AND every auxiliary LLM call disabled or pointed at a
  SEPARATE endpoint. Correctness must not depend on `cache_key` (unhonored in
  the observed build).
* **R2 — bounded window** (`per_turn_cost_le_budget`, `trigger_le_budget`):
  `compression.budget_tokens` set, so the resident window — and therefore the
  worst-case miss — is a constant.
* **R3 — archive present** (`archive_preserves_accessible`,
  `delete_loses_information`): R2 is only sound with the cold tier
  (`session_search`) enabled, so a bounded window relocates rather than
  discards. `context.engine: mechanical` keeps the in-window digest itself
  non-lossy for user turns and the tool-call index.

`tests/edge/test_edge_stability.py` pins these numbers to the Python side
and asserts the shipped `docs/edge/config.yaml` + launch scripts satisfy
R1–R3. -/

end EdgeStability
