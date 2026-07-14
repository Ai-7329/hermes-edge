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
| 50 (i5-class, 35B-A3B MoE — **measured, see below**) | 34 s | 7,914 tok ≈ 158 s (hidden by warmup) | 32768 |
| 25 (older desktop, 7-14B) | 69 s | ≈ 317 s | 16384 |
| 10 (N100/SBC-class, 4-8B) | 171 s | 4,228 tok ≈ 423 s at budget 8192 — header size becomes the dominant term; shrink toolsets further / keep `tool_search` on | 8192 |

## Measured: 35B-A3B inside an 8 GB memory ceiling

Can the 19.6 GB model run on an 8 GB-class box at all? Measured, not
argued: aarch64 Linux VM, cgroup limit **7 GiB with swap denied**
(`--memory 7g --memory-swap 7g`, the ~8.2 GB VM plays the role of the
whole box), CPU-only, weights on a **native ext4 volume**, `--no-repack`,
default mmap. The OS page cache — charged against the same 7 GiB — is the
only expert-caching mechanism (colibri's hierarchy, degenerate form,
zero new code):

| Metric | Measured |
|---|---|
| Decode | **2.2–3.2 tok/s** (23-token and 7-token runs) |
| Prefill, amortized | **7.6 tok/s** (792-token prompt, 131.8 ms/tok) |
| Prefill, tiny prompt | 2.1 tok/s (12 tokens — no batch amortization) |
| Cold model start | minutes-class (~104 s for load+792-tok prefill, warm-ish cache) |
| OOM without `--no-repack` | confirmed (`oom_kill 1`) — repack materializes weights in RAM |

Operational conclusions for an 8 GB deployment:

1. **It runs, unattended-agent shaped.** Answer-in-minutes monitoring/
   automation is viable; interactive chat is not the use case. Profile:
   `prefill_tps: 7`, `budget_tokens: 4096`, minimal toolsets, tool_output
   ≤ 4 KB. A 4–8B dense model (fully resident, ~15–40 tok/s CPU) remains
   the pragmatic interactive choice on this class — both are now measured
   options, not guesses.
2. **`--no-repack` is mandatory** below weight size: ARM repack rebuilds
   tensors in anonymous RAM and the cgroup kills the process.
3. **The model must sit on a local filesystem.** A FUSE/network mount
   (virtiofs here; NFS on a factory floor) breaks mmap readahead — 4 KB
   page-fault reads, observed at ~270 KB/s effective: hours per load.
   Moving the same file to native ext4 turned it into the numbers above.
4. Page-cache LRU already captures the A3B hot-expert set well enough for
   the band above; colibri-style learned expert pinning is an
   *optimization* on top (fork candidate: per-expert mlock), not a
   prerequisite.

Caveats: virtualized I/O and aarch64 NEON (a real x86 edge box with local
NVMe differs in both directions); decode samples are short; multi-hour
steady-state churn unmeasured.

## Measured: the reference box (i5-14500 / 32 GB / 8 GB GPU)

Phase-1 calibration from VERIFY.md, run on the deployment target
(35B-A3B Q4_K_M, experts on CPU, `n_ctx` 66,560/slot × 2 slots — the
`--parallel 2` split works on the hybrid-attention model):

| Metric | Measured |
|---|---|
| Prefill, cold | 50–57 tok/s (stable: 19,491 tok / 389.7 s = 50.0; 17,446 / 343.6 s = 50.8) |
| Prefill, checkpoint-restore mixed | 40–47 tok/s (restore works on-device) |
| Decode, single slot | ~26 tok/s |
| Decode, 2 slots concurrent | 12–13 tok/s each |
| Decode vs context length | 22K → 17, 34K → 14, 37K → 13 tok/s |

Three design assumptions this confirms:

1. **Prefill ~50 tok/s is the world the fork was built for** — the
   README's founding number, reproduced on the target.
2. Decode ~26 sits at the computed memory-bandwidth ceiling
   (dual-channel DDR5 ÷ ~2 GB active weights/token ≈ 27), and
   **concurrency halves it** — which is why the endpoint gate serializes
   instead of letting requests share the box.
3. Decode decays with context (26 → 13 by 37K), so
   `compression.budget_tokens` is not only re-prefill economics: keeping
   the working set small **preserves generation speed** too.

Profile calibration from these numbers: `local_runtime.prefill_tps: 40`
(the checkpoint-restore-mixed floor, rounded down — deadlines derive
from the slowest realistic prefill, not the cold best case).

## Measured: the cost of leaving half the machine free

Coexistence claim (the agent as tenant, not landlord) quantified: same
35B-A3B, full-RAM CPU inference, 727-token prompt, measured on a machine
that was simultaneously running other work — which is the point:

| Threads | Prefill | Decode |
|---|---|---|
| 8 | 32.9 tok/s | 16.7 tok/s |
| 4 (half) | **37.8 tok/s (+15%)** | 14.0 tok/s (−16%) |

Halving the CPU allocation costs 16% of decode and *improved* prefill —
threads spilling onto efficiency cores and cross-process contention hurt
more than the lost cores helped. Decode's wall is memory bandwidth, not
core count, so a half-machine allocation is nearly free. `COEXIST=1` in
the launch scripts applies this (half threads + lower priority); RAM
shares automatically (mmap, no mlock — under pressure inference degrades
toward the measured 7-GiB paging floor instead of OOMing), and GPU
headroom is set by lowering `-ngl` until `nvidia-smi` shows the desired
free VRAM (CUDA time-slices compute between processes on its own).

## Live end-to-end run (real model, real server)

Executed on an M1 Pro 32GB against llama-server (35B-A3B MoE Q4_K_M,
Metal, measured prefill ~395 tok/s, decode ~31 tok/s), isolated
`HERMES_HOME`, the edge profile, one 4-turn chat session driven over a PTY
(passphrase set → recall → 20,000-line tool command → recall again):

| Observed | Evidence |
|---|---|
| Header 14,866 tok with 5 toolsets + repo AGENTS.md → **~5K tok** after toolset trim + clean cwd | server log prompt_n; `hermes prompt-size` breakdown |
| Turn 2 prefilled **+37 tokens in ~3 s** (vs 18 s full prefill on turn 1) | within-session prefix cache working end-to-end |
| Tool flood capped: `seq 1 20000` (~100KB raw) entered history bounded; correct answer ("20000") | `tool_output.max_bytes` live |
| Budget compaction fired at 6,500 tokens (13000 × 0.5), mechanical digest, **7→4 messages, −54%**, no LLM summary call, no livelock on the immediate ineffective retry | agent.log |
| Passphrase recalled correctly **after 2 compactions** | digest keeps user turns verbatim |
| Warmup ran post-compaction; **endpoint gate serialized it against the main call live** ("main waited 26.3s") | agent.log `local_runtime` lines |
| Offline switches held: no update-check cache, no catalog cache created | isolated home stayed clean |

Findings the run exposed (each now addressed or documented):

1. **Preflight-compaction warmup raced the imminent real call** — the gate
   made the collision safe (serialized, no cache damage) but the main call
   waited 26 s for a warmup whose bytes then diverged (post-compaction
   memory reload changes the system prompt). Fixed: warmup takes a 2 s
   grace and yields (skip-when-busy) to an imminent real call; it now fires
   usefully only in the idle post-response case.
2. **`-z/--oneshot` ignores `--continue`/`--resume`** (upstream design:
   every oneshot is a fresh session). Scripted multi-turn edge use needs
   the chat/gateway process, or a future oneshot-resume feature.
3. **Cross-session header reuse on hybrid-attention models depends on
   checkpoint density** — with `--ctx-checkpoints 4` the new session's
   header (a strict prefix of the old state) still fully re-prefilled.
   Use the fork default 32 (scripts do); dense checkpoints are what make
   restart-warm-starts real on SWA/hybrid models.
4. Reasoning-heavy models grow context fast from their own thinking
   (~4.9K generated tokens on the tool turn). On slow edge boxes consider
   server-side reasoning off/low; the compaction budget contains it either
   way.

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
