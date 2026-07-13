/-!
# hermes-edge — formal audit of the edge design arithmetic

Machine-checked (core Lean 4, no mathlib) models of the arithmetic and
decision logic added by the edge fork, each mapped to its Python source
and to the pytest that pins the implementation to this model.

**Scope — what a proof here means.** These theorems certify the *design*:
the arithmetic can't overflow its stated bounds and the gate's decision
table has no bad row. They do not certify the Python runtime (GIL, OS
scheduling, `threading.RLock` timing) — that layer is covered empirically
by `tests/agent/test_local_runtime.py` and
`tests/agent/test_compression_budget_warmup.py`. Concrete `example`s are
kernel-checked against the same numbers verified by running the Python
implementation, so model and implementation are cross-anchored bit-for-bit
on the profile's operating point.

Conventions: token counts and milliseconds are `Nat`; percentages are
rationals `n/d` under Python's floor semantics (`int(x * pct)` =
`x * n / d` in `Nat`).
-/

namespace EdgeAudit

/-! ## §1 Compaction trigger with an explicit budget

Python: `agent/context_compressor.py::_compute_threshold_tokens`
(budget branch). Pytest: `TestBudgetTokens` in
`tests/agent/test_compression_budget_warmup.py`.
-/

/-- `ContextCompressor` effective input window:
`context_length - max_tokens`, falling back to `context_length` when the
reservation swallows the window (Python guards `<= 0`; `Nat` subtraction
truncates identically). -/
def effectiveWindow (ctx maxTok : Nat) : Nat :=
  if ctx - maxTok = 0 then ctx else ctx - maxTok

/-- Budget-mode trigger: `max(1, max(1, min(budget, window)) * pct)`. -/
def budgetTrigger (budget window n d : Nat) : Nat :=
  max 1 (max 1 (min budget window) * n / d)

/-- The trigger never exceeds the window the provider will accept —
compaction always fires while the request can still be sent. -/
theorem trigger_le_window (budget window n d : Nat)
    (hd : 0 < d) (hn : n ≤ d) (hw : 1 ≤ window) :
    budgetTrigger budget window n d ≤ window := by
  have hmin : max 1 (min budget window) ≤ window := by
    have h1 : min budget window ≤ window := Nat.min_le_right _ _
    exact Nat.max_le.mpr ⟨hw, h1⟩
  have hmul : max 1 (min budget window) * n ≤ window * d :=
    Nat.mul_le_mul hmin hn
  have hdiv : max 1 (min budget window) * n / d ≤ window * d / d :=
    Nat.div_le_div_right hmul
  have hcancel : window * d / d = window := Nat.mul_div_cancel window hd
  exact Nat.max_le.mpr ⟨hw, by rw [hcancel] at hdiv; exact hdiv⟩

/-- The trigger never exceeds the budget itself — the operator's
working-set statement is a hard cap, so every planned re-prefill stays
inside the budgeted size. -/
theorem trigger_le_budget (budget window n d : Nat)
    (hd : 0 < d) (hn : n ≤ d) (hb : 1 ≤ budget) :
    budgetTrigger budget window n d ≤ budget := by
  have hmin : max 1 (min budget window) ≤ budget := by
    have h1 : min budget window ≤ budget := Nat.min_le_left _ _
    exact Nat.max_le.mpr ⟨hb, h1⟩
  have hmul : max 1 (min budget window) * n ≤ budget * d :=
    Nat.mul_le_mul hmin hn
  have hdiv : max 1 (min budget window) * n / d ≤ budget * d / d :=
    Nat.div_le_div_right hmul
  have hcancel : budget * d / d = budget := Nat.mul_div_cancel budget hd
  exact Nat.max_le.mpr ⟨hb, by rw [hcancel] at hdiv; exact hdiv⟩

/-- Profile operating point, bit-matching the Python run:
budget 32768, ctx 131072, max_tokens 8192, pct 1/2 → trigger 16384. -/
example : budgetTrigger 32768 (effectiveWindow 131072 8192) 1 2 = 16384 := rfl

/-- Upstream default for contrast (no budget): pct raised to 3/4 by the
small-context rule, `MINIMUM_CONTEXT_LENGTH` floor 64000, degenerate-window
guard at 85%. Python cross-check: 92160. -/
def upstreamTrigger (window n d minCtx : Nat) : Nat :=
  let pctValue := window * n / d
  let floored := max pctValue minCtx
  if window ≤ floored then max 1 (min (window * 85 / 100) (window - 1))
  else floored

example : upstreamTrigger (effectiveWindow 131072 8192) 3 4 64000 = 92160 := rfl

/-! ## §2 Prefill-aware deadlines

Python: `agent/local_runtime.py::prefill_floor_seconds` /
`local_aux_timeout_floor` / `bounded_local_stale_seconds`.
Pytest: `test_prefill_floor_math`, `test_floors_off_without_prefill_tps`.
-/

/-- Milliseconds a backend at `tps` tokens/second needs to prefill
`tokens`. -/
def prefillMs (tokens tps : Nat) : Nat := tokens * 1000 / tps

/-- Deadline floor: margin plus own prefill time. -/
def floorMs (marginMs tokens tps : Nat) : Nat := marginMs + prefillMs tokens tps

/-- The floor always covers the request's own prefill. -/
theorem floor_covers_prefill (m t s : Nat) : prefillMs t s ≤ floorMs m t s :=
  Nat.le_add_left _ _

/-- Larger requests never get shorter deadlines. -/
theorem floor_mono (m s t₁ t₂ : Nat) (h : t₁ ≤ t₂) :
    floorMs m t₁ s ≤ floorMs m t₂ s := by
  have : t₁ * 1000 ≤ t₂ * 1000 := Nat.mul_le_mul_right 1000 h
  exact Nat.add_le_add_left (Nat.div_le_div_right this) m

/-- Why a static deadline structurally loses: if `(deadline+1)·tps ≤
tokens·1000` then the deadline expires strictly before prefill completes —
at any setting, for any request past `deadline × tps` tokens. -/
theorem static_deadline_loses (deadlineMs tokens tps : Nat) (htps : 0 < tps)
    (h : (deadlineMs + 1) * tps ≤ tokens * 1000) :
    deadlineMs < prefillMs tokens tps :=
  Nat.lt_of_lt_of_le (Nat.lt_succ_self _)
    ((Nat.le_div_iff_mul_le htps).mpr h)

/-- The observed failure, exactly: a 30s deadline against a 16,419-token
tool payload at 50 tok/s (the measured +16,419-token turn). -/
example : (30000 : Nat) < prefillMs 16419 50 := by decide

/-- The bounded stale deadline (replaces upstream's `float("inf")` for
local endpoints): full prefill plus a decode budget. -/
def staleBoundMs (marginMs tokens tps decodeMs : Nat) : Nat :=
  floorMs marginMs tokens tps + decodeMs

/-- A wedged server is detected in bounded time, yet the bound still
covers a legitimate full prefill and full decode budget. -/
theorem stale_bound_covers_prefill (m t s dec : Nat) :
    prefillMs t s ≤ staleBoundMs m t s dec :=
  Nat.le_trans (floor_covers_prefill m t s) (Nat.le_add_right _ _)

theorem stale_bound_covers_decode (m t s dec : Nat) :
    dec ≤ staleBoundMs m t s dec :=
  Nat.le_add_left _ _

/-! ## §3 Endpoint-gate decision table

Python: `agent/local_runtime.py::EndpointGate.__enter__` +
`gate_openai_client`. Pytest: serialization/reentrancy/busy-skip/fail-open
tests in `tests/agent/test_local_runtime.py`.

The table models the decision *endpoints* (what holds once any wait
budget is exhausted); RLock timing itself is empirical territory.
-/

inductive Outcome
  | enter                     -- proceed holding the gate
  | skipNoHttp                -- deferrable task skipped BEFORE any HTTP
  | enterWithoutExclusivity   -- fail-open after bounded wait
  deriving DecidableEq

/-- Decision function: `skippable` = task is in `gate_skip_when_busy`,
`heldByOther` = another thread holds the gate past our wait budget,
`heldBySelf` = this thread already holds it (reentrant path). -/
def gateOutcome (skippable heldByOther heldBySelf : Bool) : Outcome :=
  if heldBySelf then .enter
  else if heldByOther then
    if skippable then .skipNoHttp else .enterWithoutExclusivity
  else .enter

/-- The main conversation (never skippable) is never dropped. -/
theorem main_never_skipped :
    ∀ o s : Bool, gateOutcome false o s ≠ Outcome.skipNoHttp := by decide

/-- Reentry from the same thread always enters — the streaming →
non-streaming fallback cannot self-deadlock. -/
theorem self_reentry_always_enters :
    ∀ sk o : Bool, gateOutcome sk o true = Outcome.enter := by decide

/-- A deferrable task meeting a busy gate never reaches the backend:
the skip happens before any HTTP request exists. -/
theorem busy_deferrable_sends_no_http :
    gateOutcome true true false = Outcome.skipNoHttp := by decide

/-- A free gate always admits the caller. -/
theorem free_gate_enters :
    ∀ sk : Bool, gateOutcome sk false false = Outcome.enter := by decide

/-- No input row blocks forever: every combination reaches a defined
outcome (totality, checked exhaustively). -/
theorem gate_total :
    ∀ sk o s : Bool,
      gateOutcome sk o s = Outcome.enter ∨
      gateOutcome sk o s = Outcome.skipNoHttp ∨
      gateOutcome sk o s = Outcome.enterWithoutExclusivity := by decide

/-! ## §4 Post-compaction re-prefill bound

Python: summary ceiling `min(budget_window * 5/100, 10000)` and tail
budget `trigger * ratio` in `agent/context_compressor.py`; warmup in
`run_agent.py::_maybe_schedule_compression_warmup`.
-/

/-- Summary (digest) token ceiling for a given budget window. -/
def digestCap (budgetWindow : Nat) : Nat := min (budgetWindow * 5 / 100) 10000

/-- Tail budget: `int(trigger * ratio)` with ratio `rn/rd`. -/
def tailCap (trigger rn rd : Nat) : Nat := trigger * rn / rd

/-- Everything the next call must re-prefill after compaction. -/
def rePrefillTokens (header digest tail : Nat) : Nat := header + digest + tail

/-- The bound is additive and monotone: components under their caps keep
the total under the cap sum. -/
theorem rePrefill_bounded (header digest tail H D T : Nat)
    (hh : header ≤ H) (hd : digest ≤ D) (ht : tail ≤ T) :
    rePrefillTokens header digest tail ≤ H + D + T := by
  unfold rePrefillTokens; omega

/-- Profile operating point, cross-checked against the Python run:
digest cap 1638 (= Python `max_summary_tokens`), tail cap 3276
(= Python `tail_token_budget`), and with a 3,000-token header the whole
post-compaction re-prefill is 7,914 tokens ≈ 158 s at 50 tok/s. -/
example : digestCap 32768 = 1638 := rfl
example : tailCap 16384 1 5 = 3276 := rfl
example : rePrefillTokens 3000 1638 3276 = 7914 := rfl
example : prefillMs 7914 50 = 158280 := rfl
example : prefillMs 7914 50 ≤ 180000 := by decide   -- ≤ 3 minutes
/-- Same layout on a 10 tok/s ultra-low box: ~13.2 minutes — the profile
must shrink `budget_tokens` on such hardware (see AUDIT.md scaling). -/
example : prefillMs 7914 10 = 791400 := rfl

/-! ## §5 Tool-output cap arithmetic

Python: `tool_output.max_bytes` (profile) — per-turn prefill cost of tool
results entering history.
-/

/-- Tokens for `bytes` at a bytes-per-token ratio `bn/bd`
(e.g. 35/10 = 3.5 B/tok for mixed prose, ~3.05 measured on the
50KB incident). -/
def bytesToTokens (bytes bn bd : Nat) : Nat := bytes * bd / bn

/-- The upstream 50KB default at the *measured* 3.05 B/tok ratio
reproduces the observed +16,393-token flood (user measured 16,419 —
ratio rounding; same class), and the 6KB edge cap holds a turn to ~34 s
at 50 tok/s. -/
example : bytesToTokens 50000 305 100 = 16393 := rfl
example : bytesToTokens 6000 35 10 = 1714 := rfl
example : prefillMs (bytesToTokens 6000 35 10) 50 = 34280 := rfl
example : prefillMs (bytesToTokens 6000 35 10) 50 ≤ 40000 := by decide

end EdgeAudit
