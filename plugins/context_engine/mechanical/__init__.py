"""Mechanical digest context engine — compaction without an LLM call.

Why this exists: the built-in compressor summarizes the middle window with
an auxiliary LLM at the moment the context is fullest.  On a single-slot
local backend (e.g. llama-server on consumer hardware) that summary request
is a full-window prefill issued at the worst possible time — it reliably
exceeds the auxiliary timeout, timeouts are classified as network failures
which always abort compaction (nothing is dropped), the failure cooldown is
only 30s, and the session livelocks: compress → timeout → pause → compress,
while the threshold stays exceeded forever and every cycle occupies the
backend for minutes.

This engine removes the failure mode instead of tuning around it: the
"summary" is a deterministic mechanical extraction over the message dicts
already in memory —

  * user turns verbatim        (contracts; they originate outside the model
                                and cannot be regenerated, so never paraphrase)
  * assistant report spine     (the model's own substantive reports, head-cut)
  * tool execution index       (deduplicated, with repeat counts and result
                                heads, so the model can SEE it already ran a
                                command and what came back — the anti-amnesia
                                core that stops re-running the same probes)
  * current position anchor    (last substantive report)

Pure function of the window: no network, no timeout, no cooldown, O(window)
string work in milliseconds.  Everything else — threshold arithmetic, tool
result pruning, head/tail protection, role alternation, orphaned tool-pair
cleanup, media stripping — is inherited unchanged from ContextCompressor.

Select with:

    context:
      engine: mechanical

Config: reads the same ``compression:`` section as the built-in engine
(threshold / protect_first_n / protect_last_n / target_ratio).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import (
    ContextCompressor,
    _content_text_for_contains,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

# First line of every digest this engine produces.  Used to recognise our
# own previous digest on re-compaction (self-merge) vs. a legacy LLM summary
# inherited from before the engine switch.
_GEN_MARKER = "# mechanical context digest"

_SEC_USER = "## User turns (verbatim, chronological)"
_SEC_SPINE = "## Assistant report spine (chronological)"
_SEC_TOOLS = "## Tool execution index (deduplicated, latest run last)"
_SEC_POSITION = "## Current position"
_SEC_LEGACY = "## Inherited summary (pre-mechanical, opaque)"
_ALL_SECTIONS = (_SEC_USER, _SEC_SPINE, _SEC_TOOLS, _SEC_POSITION, _SEC_LEGACY)

# Per-item and per-section shape limits.  These are content-shape constants,
# not tuning knobs: verbatim cap keeps a single pasted wall-of-text from
# eating the whole budget; section caps bound worst-case list growth before
# the global char budget applies.
_USER_VERBATIM_MAX = 400
_USER_HEAD_KEEP = 6          # oldest user turns kept when the section is elided
_USER_MAX = 60
_SPINE_ITEM_MAX = 250
_SPINE_MIN_CONTENT = 120     # assistant text shorter than this is chatter, not a report
_SPINE_MAX = 40
_TOOL_RESULT_HEAD = 120
_TOOL_ARG_HEAD = 160
_TOOLS_MAX = 60
_LEGACY_HEAD = 1500
_POSITION_HEAD = 300

_ELIDE_RE = re.compile(r"^- … (\d+) earlier item\(s\) elided")

# Rendering artifacts, not items: emitted for empty sections / missing
# position, and must never be parsed back as content on self-merge.
_NONE_PLACEHOLDER = "- (none in this window)"
_NO_POSITION_PLACEHOLDER = "(no substantive assistant report in this window)"


def _single_line(text: str, limit: int) -> str:
    """Collapse whitespace/newlines and head-cut to *limit* chars."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + f" … [+{len(flat) - limit} chars]"


def _tool_call_fields(tc: Any) -> Tuple[str, str, str]:
    """Extract (call_id, name, arguments_json) from a dict or namespace."""
    if isinstance(tc, dict):
        cid = tc.get("call_id") or tc.get("id") or ""
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            return cid, fn.get("name") or "unknown", fn.get("arguments") or ""
        return cid, getattr(fn, "name", "unknown"), getattr(fn, "arguments", "")
    cid = getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""
    fn = getattr(tc, "function", None)
    return (
        cid,
        (getattr(fn, "name", "") if fn else "") or "unknown",
        (getattr(fn, "arguments", "") if fn else "") or "",
    )


def _salient_arg(arguments_json: str) -> str:
    """Pick the one argument a human would use to identify the call."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (ValueError, TypeError):
        return _single_line(arguments_json, _TOOL_ARG_HEAD)
    if not isinstance(args, dict):
        return _single_line(str(args), _TOOL_ARG_HEAD)
    for key in ("command", "file_path", "path", "query", "pattern", "url", "code"):
        val = args.get(key)
        if val:
            return _single_line(str(val), _TOOL_ARG_HEAD)
    if args:
        key = next(iter(args))
        return _single_line(f"{key}={args[key]}", _TOOL_ARG_HEAD)
    return ""


def _summarize_tool_result(content: Any) -> str:
    """One-line outcome of a tool result message.

    Terminal-style JSON payloads ({output, exit_code, error}) get a
    structured head; anything else gets a plain head-cut.  The point is to
    preserve the *outcome class* (exit code, BLOCKED, traceback head, empty
    output) — that is what stops the model from re-running the same probe.
    """
    text = _content_text_for_contains(content).strip()
    if not text:
        return "(empty result)"
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and (
        "output" in payload or "exit_code" in payload or "error" in payload
    ):
        parts = []
        if payload.get("exit_code") is not None:
            parts.append(f"exit {payload['exit_code']}")
        err = payload.get("error")
        if err:
            parts.append(_single_line(str(err), _TOOL_RESULT_HEAD))
        out = payload.get("output")
        if out:
            out = str(out)
            if len(out) <= 80:
                parts.append(_single_line(out, 80))
            else:
                parts.append(f"output {len(out):,} chars: {_single_line(out, 60)}")
        elif not err:
            parts.append("empty output")
        return "; ".join(parts) if parts else "(empty result)"
    return _single_line(text, _TOOL_RESULT_HEAD)


class MechanicalDigestEngine(ContextCompressor):
    """ContextCompressor with the LLM summarizer replaced by pure extraction."""

    def __init__(self) -> None:
        comp: Dict[str, Any] = {}
        model_cfg: Dict[str, Any] = {}
        try:
            from cli import CLI_CONFIG
            comp = CLI_CONFIG.get("compression") or {}
            model_cfg = CLI_CONFIG.get("model") or {}
        except Exception:
            # Config unavailable (tests, embedded use) — built-in defaults
            # below match the built-in engine's defaults.
            pass
        cfg_ctx = model_cfg.get("context_length")
        try:
            cfg_ctx = int(cfg_ctx) if cfg_ctx else None
        except (TypeError, ValueError):
            cfg_ctx = None
        super().__init__(
            model=str(model_cfg.get("default") or "pending"),
            threshold_percent=float(comp.get("threshold", 0.50)),
            protect_first_n=int(comp.get("protect_first_n", 3)),
            protect_last_n=int(comp.get("protect_last_n", 20)),
            summary_target_ratio=float(comp.get("target_ratio", 0.20)),
            # No summary LLM is ever called, so summary-failure knobs are inert.
            config_context_length=cfg_ctx,
        )

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "mechanical"

    def is_available(self) -> bool:
        return True

    # -- Threshold ------------------------------------------------------------

    @staticmethod
    def _effective_threshold_percent(
        context_length: int, threshold_percent: float,
    ) -> float:
        """Skip the small-context threshold floor (75% under 512K).

        The floor guards against LLM-summarizer churn: at a 50% trigger the
        incompressible floor eats the reclaimed headroom and the session
        spends wall-clock re-summarizing.  Mechanical digestion costs
        milliseconds and compacts well below the trigger in one pass, so the
        configured threshold applies as-is.
        """
        return threshold_percent

    # -- The one override that matters ---------------------------------------

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """Deterministic digest of the compression window.  Never None.

        Returning a string unconditionally makes every summary-failure branch
        in ContextCompressor.compress() (abort-on-network-failure, static
        fallback, failure cooldown) dead code on this engine — the livelock
        cannot re-enter through this path.
        """
        try:
            body = self._build_digest(turns_to_summarize, focus_topic)
        except Exception:
            # A mechanical engine must not resurrect the failure path: fall
            # back to a counting stub rather than returning None.
            logger.exception("mechanical digest failed — emitting counting stub")
            body = (
                f"{_GEN_MARKER}\n"
                f"(digest builder error — see agent log)\n"
                f"{len(turns_to_summarize)} turn(s) compacted; raw turns remain "
                f"in the session DB."
            )
        body = redact_sensitive_text(body)
        self._previous_summary = body
        return self._with_summary_prefix(body)

    # -- Extraction ----------------------------------------------------------

    def _build_digest(
        self,
        turns: List[Dict[str, Any]],
        focus_topic: Optional[str],
    ) -> str:
        users: List[str] = []
        spine: List[str] = []
        position = ""
        # tool index: key -> [line, count]; re-inserted on repeat so ordering
        # reflects the LATEST run of each distinct call.
        tools: Dict[Tuple[str, str], List[Any]] = {}
        pending_calls: Dict[str, Tuple[str, str]] = {}

        for msg in turns:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                if self._is_context_summary_content(content):
                    continue
                text = _content_text_for_contains(content).strip()
                if not text:
                    continue
                flat = " ".join(text.split())
                if len(flat) > _USER_VERBATIM_MAX:
                    kept = flat[:_USER_VERBATIM_MAX].rstrip()
                    users.append(
                        f"- 「{kept}」 [+{len(flat) - len(kept)} chars in session DB]"
                    )
                else:
                    users.append(f"- 「{flat}」")
            elif role == "assistant":
                text = _content_text_for_contains(content).strip()
                if len(text) >= _SPINE_MIN_CONTENT:
                    spine.append(f"- {_single_line(text, _SPINE_ITEM_MAX)}")
                    position = _single_line(text, _POSITION_HEAD)
                for tc in msg.get("tool_calls") or []:
                    cid, tname, arg_json = _tool_call_fields(tc)
                    key = (tname, _salient_arg(arg_json))
                    if cid:
                        pending_calls[cid] = key
                    entry = tools.pop(key, None)
                    count = (entry[1] + 1) if entry else 1
                    label = f"`{key[1]}`" if key[1] else "(no args)"
                    tools[key] = [f"- {tname} {label} → (no result)", count]
            elif role == "tool":
                cid = msg.get("tool_call_id") or ""
                key = pending_calls.pop(cid, None)
                if key is None or key not in tools:
                    continue
                outcome = _summarize_tool_result(content)
                label = f"`{key[1]}`" if key[1] else "(no args)"
                count = tools[key][1]
                suffix = f"  (×{count} runs, latest shown)" if count > 1 else ""
                tools[key][0] = f"- {key[0]} {label} → {outcome}{suffix}"

        tool_lines = [entry[0] for entry in tools.values()]

        # Merge with our own previous digest (chronology preserved: previous
        # items first).  A non-mechanical previous summary (from before the
        # engine switch) is carried as an opaque, capped block instead.
        legacy_block = ""
        prev = (self._previous_summary or "").strip()
        if prev:
            if prev.startswith(_GEN_MARKER):
                prev_secs = self._parse_own_digest(prev)
                users = prev_secs.get(_SEC_USER, []) + users
                spine = prev_secs.get(_SEC_SPINE, []) + spine
                tool_lines = self._merge_tool_lines(
                    prev_secs.get(_SEC_TOOLS, []), tool_lines
                )
                legacy_block = "\n".join(prev_secs.get(_SEC_LEGACY, []))
                if not position:
                    prev_pos = prev_secs.get(_SEC_POSITION, [])
                    if prev_pos and prev_pos[0].removeprefix("- ") != _NO_POSITION_PLACEHOLDER:
                        position = prev_pos[0].removeprefix("- ")
            else:
                legacy_block = _single_line(prev, _LEGACY_HEAD)

        users = self._elide(users, _USER_MAX, focus_topic, head_keep=_USER_HEAD_KEEP)
        spine = self._elide(spine, _SPINE_MAX, focus_topic)
        tool_lines = self._elide(tool_lines, _TOOLS_MAX, focus_topic)

        return self._render(
            users, spine, tool_lines, position, legacy_block,
            n_turns=len(turns), focus_topic=focus_topic,
        )

    # -- Self-merge helpers ---------------------------------------------------

    @staticmethod
    def _parse_own_digest(body: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        current: Optional[str] = None
        for line in body.splitlines():
            stripped = line.rstrip()
            if stripped in _ALL_SECTIONS:
                current = stripped
                sections[current] = []
            elif current and stripped.startswith("- "):
                if stripped != _NONE_PLACEHOLDER:
                    sections[current].append(stripped)
            elif current == _SEC_LEGACY and stripped:
                sections[current].append(stripped)
        return sections

    @staticmethod
    def _merge_tool_lines(prev: List[str], new: List[str]) -> List[str]:
        """Dedup by the '- <tool> `<arg>` →' identity prefix, newest wins."""
        def ident(line: str) -> str:
            return line.split("→", 1)[0].strip()
        new_idents = {ident(l) for l in new}
        kept = [l for l in prev if ident(l) not in new_idents
                and not _ELIDE_RE.match(l)]
        return kept + new

    @staticmethod
    def _elide(
        items: List[str],
        cap: int,
        focus_topic: Optional[str],
        head_keep: int = 0,
    ) -> List[str]:
        """Cut a section to *cap* items, dropping oldest first.

        Items matching *focus_topic* are pinned.  An explicit elision marker
        replaces whatever was dropped — a digest must never silently truncate
        (the raw turns still exist in the session DB, and the marker says so).
        Counts from markers of previous passes are carried forward so the
        marker never under-reports what was dropped.
        """
        carried = 0
        kept_items = []
        for l in items:
            m = _ELIDE_RE.match(l)
            if m:
                carried += int(m.group(1))
            else:
                kept_items.append(l)
        items = kept_items
        if len(items) <= cap:
            if carried:
                items = [
                    f"- … {carried} earlier item(s) elided "
                    f"(raw remains in the session DB) …"
                ] + items
            return items
        topic = (focus_topic or "").casefold()
        pinned = [l for l in items if topic and topic in l.casefold()]
        head = items[:head_keep]
        # Pinning is a preference, cap is the constraint: keep only the newest
        # pinned lines that still fit after the head and one tail slot.  An
        # unbounded pin set makes this function a no-op on all-pinned sections,
        # and the _render budget loop then cannot make progress.
        max_pinned = max(cap - head_keep - 1, 0)
        if len(pinned) > max_pinned:
            pinned = pinned[-max_pinned:]
        tail_budget = cap - head_keep - len([l for l in pinned if l not in head])
        tail = items[head_keep:]
        kept_tail = tail[-max(tail_budget, 1):]
        merged = head + [l for l in pinned if l not in head and l not in kept_tail] + kept_tail
        n_dropped = len(items) - len(merged) + carried
        if n_dropped > 0:
            marker = (
                f"- … {n_dropped} earlier item(s) elided "
                f"(raw remains in the session DB) …"
            )
            merged.insert(len(head), marker)
        return merged

    # -- Rendering -----------------------------------------------------------

    def _render(
        self,
        users: List[str],
        spine: List[str],
        tool_lines: List[str],
        position: str,
        legacy_block: str,
        n_turns: int,
        focus_topic: Optional[str],
    ) -> str:
        # Char budget derives from the inherited summary token budget
        # (recomputed by update_model), so it tracks the real window size.
        budget = max(4000, min(24000, int(self.max_summary_tokens * 3)))

        def render_once() -> str:
            parts = [
                _GEN_MARKER,
                (
                    f"Deterministic extraction of {n_turns} compacted turn(s) — "
                    "no summarizer LLM involved. Quoted 「…」 text is verbatim; "
                    "everything else is head-cut, not paraphrased."
                    + (f" Focus: {_single_line(focus_topic, 120)}"
                       if focus_topic else "")
                ),
                "",
                _SEC_USER,
                *(users or [_NONE_PLACEHOLDER]),
                "",
                _SEC_SPINE,
                *(spine or [_NONE_PLACEHOLDER]),
                "",
                _SEC_TOOLS,
                *(tool_lines or [_NONE_PLACEHOLDER]),
            ]
            if legacy_block:
                parts += ["", _SEC_LEGACY, legacy_block]
            parts += [
                "",
                _SEC_POSITION,
                f"- {position or _NO_POSITION_PLACEHOLDER}",
                "",
                (
                    "Raw turns are NOT lost: the full transcript remains in the "
                    "session DB; this digest only replaces them in the live "
                    "context window."
                ),
            ]
            return "\n".join(parts)

        body = render_once()
        # Budget enforcement: shrink lists oldest-first (spine → tools → users)
        # before ever hard-cutting text.  Loop variant: each pass must strictly
        # shrink the section — a section that cannot shrink further (floor
        # reached modulo the elision marker) falls through to the hard cut.
        shrink_order = [
            (spine, 12, 0), (tool_lines, 20, 0), (users, 20, _USER_HEAD_KEEP),
        ]
        for lst, floor, head_keep in shrink_order:
            while len(body) > budget and len(lst) > floor:
                reduced = self._elide(lst, max(floor, len(lst) - 10),
                                      focus_topic, head_keep=head_keep)
                if len(reduced) >= len(lst):
                    break
                lst[:] = reduced
                body = render_once()
        if len(body) > budget:
            body = body[:budget].rstrip() + "\n… [digest hard-cut to budget]"
        return body


def register(ctx) -> None:
    ctx.register_context_engine(MechanicalDigestEngine())
