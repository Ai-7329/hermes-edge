# Edge-suitability audit

Question audited: **is this fork actually fit for genuinely low-spec edge
devices** — not "runs on a smaller PC", but bounded memory, bounded disk,
bounded wall-clock behavior, zero network dependence, unattended operation.

Method: every design claim is checked at three layers.
**Lean** (`docs/edge/audit/EdgeAudit.lean`, core Lean 4 v4.29.0, no mathlib —
`lean EdgeAudit.lean` must exit 0) proves the arithmetic and decision logic.
**pytest** pins the Python implementation to that model.
**Measurement** covers what neither can (RSS, disk, timings). Concrete Lean
`example`s and the Python runs are anchored bit-for-bit on the same numbers.

## Item-by-item

| # | Claim | Lean theorem | pytest | Measured |
|---|---|---|---|---|
| 1 | Compaction trigger never exceeds the provider window | `trigger_le_window` | `TestBudgetTokens` | — |
| 2 | Trigger never exceeds the operator budget (re-prefill hard cap) | `trigger_le_budget` | `test_budget_sets_threshold_directly` | — |
| 3 | Profile operating point: trigger 16384 (budget 32768, pct ½) | `example … = 16384 := rfl` | same test, same number | Python run: 16384 |
| 4 | Upstream default for contrast: trigger 92160 (~30 min re-prefill class @50 tok/s) | `example … = 92160 := rfl` | `test_no_budget_keeps_upstream_small_ctx_raise` | Python run: 92160 |
| 5 | Deadline floor always covers the request's own prefill | `floor_covers_prefill`, `floor_mono` | `test_prefill_floor_math` | — |
| 6 | A static deadline structurally loses past `deadline×tps` tokens | `static_deadline_loses`; concrete `30000 < prefillMs 16419 50` | timeout floor tests | observed failure reproduced |
| 7 | Wedged-server detection is bounded yet covers full prefill + decode | `stale_bound_covers_*` | `test_floors_*` | — |
| 8 | Main conversation can never be skipped by the gate | `main_never_skipped` (exhaustive) | gate serialization tests | — |
| 9 | Same-thread reentry cannot deadlock | `self_reentry_always_enters` | `test_gate_reentrant_same_thread` | — |
| 10 | Busy + deferrable ⇒ skip **before any HTTP** (cache never disturbed) | `busy_deferrable_sends_no_http` | `test_gated_client_skips_busy_aux_without_http` | — |
| 11 | Every gate input row terminates in a defined outcome | `gate_total` (exhaustive) | fail-open test | — |
| 12 | Post-compaction re-prefill ≤ header + digest cap + tail cap | `rePrefill_bounded` | budget/warmup tests | — |
| 13 | Profile numbers: digest cap 1638, tail cap 3276, re-prefill 7,914 tok ≈ 158 s @50 tok/s (warmup hides it in idle) | four `rfl` examples | `test_summary_ceiling_follows_budget` | Python run: 1638 |
| 14 | 50KB tool cap ≈ the observed 16.4K-token flood; 6KB cap ≈ 34 s/turn | `bytesToTokens` examples | — | incident: +16,419 tok |
| 15 | Agent process floor | — | — | import RSS **93 MB**, warm import 0.7 s (M1 Pro) |
| 16 | Core-only install is edge-sized | — | — | **64 packages / 195 MB** site-packages (kitchen-sink personal venv: 210 pkg / 831 MB — extras, not core) |
| 17 | Zero unconditional outbound on CLI launch | — | `test_update_check_on_startup.py` | surfaces enumerated below |
| 18 | Long-run hygiene | — | upstream suites | sessions auto-prune + log rotation on; **multi-day soak not yet run** |

## Offline surface enumeration (item 17)

Every outbound path reachable from a default CLI launch, each with its
real code gate (verified consumed, not aspirational):

| Surface | Code gate | Edge profile |
|---|---|---|
| Model-catalog manifest fetch (1h TTL) | `hermes_cli/model_catalog.py::get_catalog` early-return | `model_catalog.enabled: false` |
| Startup update check (git ls-remote / PyPI, bg thread) | `hermes_cli/main.py` (**fork switch** — upstream had none) | `updates.check_on_startup: false` |
| LSP server auto-download | `agent/lsp/manager.py` `enabled` gate | `lsp.enabled: false` |
| tirith scanner probe | security config | `security.tirith_enabled: false` |
| Whisper STT model load (RAM, local but heavy) | stt config | `stt.enabled: false` |
| web/browser/TTS/vision tool surfaces | not loaded at all | excluded from `platform_toolsets` |
| Title generation | `auxiliary.title_generation.enabled` (fork switch) | `false` |
| Curator / memory nudge / background review | config intervals | off / 0 |

## Scaling (what "low-spec" means in numbers)

Prefill speed is the one parameter everything keys off (`local_runtime.prefill_tps`).
Costs below are Lean-checked arithmetic on the profile shapes:

| prefill tok/s | 6KB tool turn | re-prefill after compaction | recommended `budget_tokens` |
|---|---|---|---|
| 50 (i5-class, 35B-A3B MoE) | 34 s | 7,914 tok ≈ 158 s (hidden by warmup) | 32768 |
| 25 (older desktop, 7-14B) | 69 s | ≈ 317 s | 16384 |
| 10 (N100/SBC-class, 4-8B) | 171 s | 4,228 tok ≈ 423 s at budget 8192 — header size becomes the dominant term; shrink toolsets further / keep `tool_search` on | 8192 |

## Honest limits (what this audit does NOT establish)

1. **Lean proves the model, pytest pins the implementation, neither proves
   the OS.** RLock timing, GIL scheduling and httpx socket behavior remain
   empirical (covered by tests, not proofs).
2. **Async auxiliary paths are not gated** (MoA/gateway aggregation — not
   loaded in the edge CLI profile, but a known boundary).
3. **Multi-day soak is not yet run.** Session pruning and log rotation are
   configured; RSS drift over weeks needs an on-device measurement.
4. **Process floor is ~100 MB-class Python.** Fine for 4 GB+ boxes next to
   a small model; genuinely MCU-class (<1 GB) targets are out of scope for
   this codebase.
5. **The model dominates the box.** 35B-A3B wants ~32 GB RAM; 8–16 GB edge
   boxes run 4–8B models and must re-measure `prefill_tps` — every
   mechanism scales with that one knob, which is the point of the design.
6. Deploy from an archive or shallow clone: the working tree is ~155 MB but
   full `.git` history adds ~500 MB.
