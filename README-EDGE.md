# hermes-edge

A fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
for **long-running agents on low-spec, shared machines** — a single local
llama.cpp server, prefill measured in tens of tokens/second, and a box that
must keep doing other work while the agent runs. Target deployments: office
PCs, factory-floor machines, physical-AI prep — places where the agent is a
tenant, not the landlord.

This is not a chat-latency tweak. Upstream hermes assumes a cloud provider:
concurrency is free, prefill is instant, timeouts are generous-enough
constants. On a local backend every one of those assumptions inverts, and the
result is not "slow" but **structurally unstable**: livelocks, cache-eviction
storms, and timeouts that cannot be won at any setting.

## Failure anatomy → mechanism

Measured on i5-14500 / 32GB / 8GB GPU, 35B-A3B MoE (experts on CPU),
prefill ~50 tok/s, hybrid attention (prefix cache restores from checkpoints
only — any byte change at the head forces a full re-prefill):

| Observed failure | Root cause | Mechanism in this fork |
|---|---|---|
| 15,781-token fixed header → 5.3 min for "hello" | full header re-prefill on every cache miss | trimmed toolsets + bounded context files (measured live: 14.9K → ~5K tokens; `tool_search` defers MCP/plugin tools only) + slot pinning via `custom_providers[].extra_body.cache_key`; server keeps checkpoints (`--ctx-checkpoints`) and parks evicted states (`--cache-ram`) |
| auxiliary calls all die at 30s ("cancel task") | static cloud deadlines + queueing behind main's multi-minute prefill with no concurrency control | single-flight endpoint gate (`local_runtime.single_flight`) — deferrable tasks **skip before sending HTTP**; prefill-aware timeout floors from `local_runtime.prefill_tps` |
| aux interleaving wipes main's checkpoints → full re-prefill | requests race for the same server slot | the same gate (at most one in-flight request per local endpoint) + `cache_key` slot binding |
| +16,419 tokens of tool output in one turn | `tool_output.max_bytes: 50000` ≈ 16.4K tokens — a cloud-sized cap | cap sized to prefill reality (6KB ≈ 35–40s); mechanical digest engine dedups on compaction |
| compaction livelock (66 attempts / 0 successes, 9h54m) | LLM summarizer needs a full-window prefill at the exact moment the context is fullest | **mechanical context engine** — deterministic extraction, no LLM call, no timeout (`context.engine: mechanical`) |
| unattended approval timeout reported as "User denied" | mislabel caused models to retry the same command against nobody | honest `deny_timeout` reporting |
| compaction rewrite → next call re-prefills everything at the worst time | default trigger rides a 131K window to ~92K tokens (~30 min re-prefill) | `compression.budget_tokens` working-set cap (trigger = budget × threshold) + `local_runtime.compression_warmup` re-prefills in idle time between turns |

All mechanisms are **opt-in and inert for cloud providers**: with
`local_runtime` unset, non-local endpoints behave exactly as upstream.

## Quick start

1. Launch the server with its half of the contract:
   `scripts/llama-server-edge.sh` (or `.bat` on Windows). Flags that matter:
   `--parallel 2 --ctx-checkpoints 32 --cache-ram <MiB>`. Slot binding by
   request `cache_key` needs a build that supports it (e.g.
   llama-cpp-turboquant); upstream builds ignore the field harmlessly.
2. Copy `docs/edge/config.yaml` into
   `~/.hermes/profiles/<name>/config.yaml`, fix the `CHANGEME`s, and set
   the profile active.
3. Measure, then calibrate:
   - **prefill tok/s** (server log `prompt eval time`) → `local_runtime.prefill_tps`
   - **parked-state size** (server log on `--cache-ram` save) → size `--cache-ram`
   - **VRAM at full context** (`nvidia-smi`) → `-ngl` / `-ctk q8_0 -ctv q8_0`

## Config added by this fork

```yaml
local_runtime:
  single_flight: true        # serialize requests per local endpoint
  prefill_tps: 50            # measured; drives all timeout floors
  decode_budget: 900         # generation allowance in stale bounds
  timeout_margin: 30
  gate_wait_timeout: 900     # main: max gate wait, then fail-open
  busy_skip_wait: 15         # deferrable aux: give up fast, skip HTTP
  gate_skip_when_busy: [...] # tasks that skip instead of queue
  gate_hosts: []             # govern extra (non-auto-local) hosts
  compression_warmup: true   # idle-time re-prefill after compaction

compression:
  budget_tokens: 32768       # working-set cap for compaction policy

auxiliary:
  title_generation:
    enabled: false           # upstream had no off switch
```

`local_runtime.prefill_tps` also converts the upstream "local endpoints never
go stale" infinity into a computable bound (full prefill + decode budget), so
an **unattended** session detects a wedged server in bounded time instead of
hanging forever.

## Design rules this fork follows

1. **Fix causes, not symptoms.** A 30s timeout that always loses is not
   fixed by making it 120s; it is fixed by not sending droppable requests to
   a busy server at all, and by deriving deadlines from measured speed.
2. **The scheduler must not rely on luck.** Anything that touches the local
   server goes through one gate with explicit queue-vs-skip policy.
3. **Fail open, never wedge.** Exclusivity, warmup, floors — every mechanism
   degrades to upstream behavior on error, timeout, or missing config.
4. **Cloud paths untouched.** Every hook no-ops for non-local endpoints.

## Known limits / deferred

- **Async auxiliary paths are not gated** (sync paths are; async aux is used
  by MoA-style aggregation, unusual on one-box deployments).
- **Volatile system-prompt tail** (memories / profile / date sit before the
  template-rendered tools block, so a day rollover or memory flush
  invalidates the tools region across sessions). Deliberately deferred:
  upstream already builds the system prompt once per session and replays it
  verbatim, so the win is limited to cross-session prefix reuse, and the
  rewrite would cut deep into the prompt-assembly path. Revisit if
  cross-session warm starts become a measured bottleneck.
- Warmup byte-fidelity is best-effort (it mirrors the main loop's message
  scrub); divergence costs one wasted idle prefill, never a wrong prompt.

## Formal audit

`docs/edge/AUDIT.md` answers "is this actually edge-fit?" item by item:
the design arithmetic and gate decision table are machine-checked in Lean 4
(`docs/edge/audit/EdgeAudit.lean` — core only, `lean EdgeAudit.lean` exits 0),
each theorem is pinned to the implementation by a named pytest, and the
footprint/offline claims are measured (import RSS 93 MB, core install
64 packages / 195 MB, zero unconditional outbound on CLI launch).

## Relationship to upstream

MIT, © Nous Research (see `LICENSE`). Branch layout: `main` mirrors
upstream; `edge` carries this fork. Rebase `edge` onto upstream `main` to
update.

---

## クイックスタート (日本語)

低スペ共有機で hermes を長期運用するための fork。チャットを速くする改造では
なく、「他の作業と同居しながら数週間動き続ける」ための構造修正が本体。

1. `scripts/llama-server-edge.bat` (Windows) / `.sh` で server を起動する
   (`--parallel 2 --ctx-checkpoints 32 --cache-ram <MiB>` が要点。request
   `cache_key` での slot 固定は対応 build のみ、非対応でも無害)。
2. `docs/edge/config.yaml` を `~/.hermes/profiles/<名前>/config.yaml` に
   コピーし、CHANGEME (モデル alias / context_length / prefill_tps) を実測値で
   埋める。
3. 較正は実測から: server ログの prompt eval 速度 → `prefill_tps`、
   `--cache-ram` 保存ログの MiB → cache-ram 予算、`nvidia-smi` → KV 量子化判断。

機構はすべて opt-in で、cloud provider には一切影響しない。
