# On-device verification runbook

Deployment-side counterpart to `AUDIT.md`: the audit proved the design;
this checklist calibrates and proves **your machine**. Written for a
Windows x86 box (i5-class, 32 GB, small NVIDIA GPU) — adjust paths for
Linux. Expected effort: ~1 hour for phases 0–3, then a multi-day soak
that runs itself.

## Phase 0 — install

```bat
git clone https://github.com/Ai-7329/hermes-edge
cd hermes-edge
py -3.11 -m venv venv && venv\Scripts\pip install -e .
```

Copy `docs\edge\config.yaml` to `%USERPROFILE%\.hermes\config.yaml` and
fill the CHANGEMEs — leave `prefill_tps` for Phase 1. Model file goes on
a **local NTFS drive** (never a network share — see AUDIT: FUSE/NFS
breaks mmap readahead).

## Phase 1 — server calibration (record 4 numbers)

Launch with the template (defaults already include
`--ctx-checkpoints 32 --cache-ram 4096 --parallel 2`):

```bat
set MODEL_PATH=C:\models\your-model.gguf
set ALIAS=your-alias
scripts\llama-server-edge.bat
```

1. **prefill tok/s** — send one large prompt (≥500 tokens) via curl to
   `/v1/chat/completions`; read `timings.prompt_n / prompt_ms` from the
   response (or the server log `prompt processing` lines). Record the
   *amortized* value, write it into `local_runtime.prefill_tps`
   (round DOWN — a too-high value makes deadlines too tight).
2. **decode tok/s** — same response, `timings.predicted_per_second`.
3. **VRAM at full context** — `nvidia-smi` while a long prompt runs.
   If tight: `-ctk q8_0 -ctv q8_0` (needs `--flash-attn on`) or lower `-ngl`.
4. **parked-state size** — server log prints MiB per state when
   `--cache-ram` saves a slot; budget `--cache-ram` ≈ 2–3 states.

Also confirm in the startup log that repack is not materializing weights
(RAM-resident boxes may keep it; below-weight-size boxes MUST pass
`--no-repack` — see AUDIT).

## Phase 2 — functional battery (one chat session, 4 turns)

`hermes --cli` and type, in order:

1. `覚えておいて: 合言葉は「青い狐」。1文で了解して。`
2. `合言葉は何？1文で。` → **PASS**: correct recall, and the status bar
   turn time collapses vs turn 1 (prefix cache: only the delta prefills).
3. `ターミナルで seq 1 20000 を実行して、最後の数を教えて。` (approve the
   command) → **PASS**: answer `20000`, context grows by ~2K tokens, not
   ~30K (`tool_output.max_bytes` working).
4. `もう一度: 最初の合言葉は？` → if compaction fired (🗜️ in status bar),
   **PASS** = still recalls the passphrase (digest keeps user turns
   verbatim).

Offline check: after the session, `%USERPROFILE%\.hermes\` must contain
no `.update_check` and no `models_dev_cache.json`.

Log check (`logs\agent.log`): look for `endpoint gate:` lines (gate
active), `compression warmup:` (only if compaction fired), and the
absence of any auxiliary timeout errors.

## Phase 3 — coexistence

Re-launch with `set COEXIST=1` (half threads, below-normal priority).
Run your real workload (Python scripts etc.) alongside and repeat one
battery turn. **PASS**: your workload is not starved; decode drops by
roughly the measured ~16%, not by half. Record both tok/s values.

## Phase 4 — soak (the long-run proof)

Leave the server + a gateway/chat session up for ≥72 h of normal use.
Once a day, record:

| Metric | Command | Fail line |
|---|---|---|
| hermes RSS | `tasklist /fi "imagename eq python.exe"` | unbounded growth |
| server RSS | same for llama-server | > weights + cache-ram + slack |
| session DB | size of `.hermes\state.db` | growth with `sessions.auto_prune` on |
| agent log | size + `findstr /c:"Compression #" logs\agent.log` | livelock pattern (repeated ineffective compressions) |
| stale events | `findstr /c:"stale" logs\agent.log` | recurring stale kills on a healthy server |

**PASS** = no OOM, no livelock, no unbounded growth, agent still answers
on day 3 with turn times consistent with day 1.

## Phase 5 — feed the numbers back

Fill the measured row (prefill/decode/VRAM/state-size + soak verdict)
into `docs/edge/AUDIT.md`'s scaling table for your hardware class, and
correct `local_runtime.prefill_tps` / `compression.budget_tokens` in the
profile if the measured speeds moved them.
