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

# Per-item and per-section shape limits.  Upper bounds only — the effective
# per-digest values scale DOWN with the char budget (see ``_shape()``) so a
# small summary budget (tiny context / compression.budget_tokens) can never
# be structurally smaller than the shapes it must hold.  That mismatch was
# a real failure: with the edge profile (budget_tokens: 32768 → ~4.9K-char
# budget) the fixed shapes overflowed every digest, the tail hard-cut then
# amputated the tool index / current position, and self-merge re-ingested
# the truncated digest so recent turns were lost permanently and silently.
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

# Char-budget bounds and section shares.  Floor 6000: the minimal digest
# (head-kept users + section floors + chrome) must always fit WITHOUT the
# last-resort hard cut — see the convergence test.  Shares apportion the
# budget so no section can starve the others: stale verbatim user quotes
# must never crowd out the tool index / current position (the anti-amnesia
# core this engine exists for).
_BUDGET_MIN = 6000
_BUDGET_MAX = 24000
_SHARES_WITH_LEGACY = {"users": 0.38, "spine": 0.18, "tools": 0.26, "legacy": 0.18}
_SHARES_NO_LEGACY = {"users": 0.44, "spine": 0.22, "tools": 0.34}
# Absolute per-section item floors: below these a section stops shrinking
# and the global pass moves on (their combined worst-case size fits inside
# _BUDGET_MIN by construction).
_FLOOR_USERS = 2   # in addition to head_keep
_FLOOR_SPINE = 3
_FLOOR_TOOLS = 4
_ELIDE_MARKER_COST = 90      # rendered marker line size, for fit arithmetic

_ELIDE_RE = re.compile(r"^- … (\d+) earlier item\(s\) elided")

# Rendering artifacts, not items: emitted for empty sections / missing
# position, and must never be parsed back as content on self-merge.
_NONE_PLACEHOLDER = "- (none in this window)"
_NO_POSITION_PLACEHOLDER = "(no substantive assistant report in this window)"
_HARD_CUT_MARKER = "… [digest hard-cut to budget]"


def _elision_marker(count: int) -> str:
    """The one marker format _ELIDE_RE recognises — every path that drops
    items must emit exactly this, or counts stop carrying across merges."""
    return (f"- … {count} earlier item(s) elided "
            f"(raw remains in the session DB) …")


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
            body = redact_sensitive_text(
                self._build_digest(turns_to_summarize, focus_topic)
            )
            self._previous_summary = body
        except Exception:
            # A mechanical engine must not resurrect the failure path: fall
            # back to a counting stub rather than returning None.  Do NOT
            # overwrite _previous_summary with the stub — it carries the
            # accumulated lineage (contracts, tool index), and one transient
            # builder error must not destroy it; the next compaction merges
            # from the kept lineage as usual.
            logger.exception("mechanical digest failed — emitting counting stub")
            body = (
                f"{_GEN_MARKER}\n"
                f"(digest builder error — see agent log)\n"
                f"{len(turns_to_summarize)} turn(s) compacted; raw turns remain "
                f"in the session DB."
            )
        return self._with_summary_prefix(body)

    # -- Budget-proportional shape -------------------------------------------

    def _shape(self) -> Dict[str, int]:
        """Effective shape limits for the current char budget.

        The budget derives from the inherited summary token budget (recomputed
        by ``update_model``), so it tracks the real window — including a
        ``compression.budget_tokens`` working-set cap.  Every shape limit
        scales with it: a digest's fixed shapes must never be allowed to
        exceed the space the digest is given.
        """
        budget = max(_BUDGET_MIN, min(_BUDGET_MAX, int(self.max_summary_tokens * 3)))
        return {
            "budget": budget,
            # 6000 → 250-char verbatim quotes, 24000 → the full 400.
            "user_verbatim": max(160, min(_USER_VERBATIM_MAX, budget // 24)),
            # 6000 → keep the 4 oldest contracts, 24000 → 6.
            "user_head_keep": max(2, min(_USER_HEAD_KEEP, budget // 1500)),
        }

    # -- Extraction ----------------------------------------------------------

    def _build_digest(
        self,
        turns: List[Dict[str, Any]],
        focus_topic: Optional[str],
    ) -> str:
        shape = self._shape()
        user_verbatim_max = shape["user_verbatim"]
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
                if len(flat) > user_verbatim_max:
                    kept = flat[:user_verbatim_max].rstrip()
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

        users = self._elide(
            users, _USER_MAX, focus_topic, head_keep=shape["user_head_keep"],
        )
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
            elif current == _SEC_LEGACY and stripped and stripped != _HARD_CUT_MARKER:
                # The hard-cut marker is a rendering artifact; ingesting it as
                # legacy content would replay it as text on every re-merge.
                sections[current].append(stripped)
        return sections

    @staticmethod
    def _merge_tool_lines(prev: List[str], new: List[str]) -> List[str]:
        """Dedup by the '- <tool> `<arg>` →' identity prefix, newest wins.

        Elision-marker counts from the previous digest are carried forward,
        not dropped — the tool index must account for every run it has ever
        elided, same as the user/spine sections.
        """
        def ident(line: str) -> str:
            return line.split("→", 1)[0].strip()
        new_idents = {ident(l) for l in new}
        carried = 0
        kept = []
        for l in prev:
            m = _ELIDE_RE.match(l)
            if m:
                carried += int(m.group(1))
            elif ident(l) not in new_idents:
                kept.append(l)
        merged = kept + new
        if carried:
            merged.insert(0, _elision_marker(carried))
        return merged

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
                items = [_elision_marker(carried)] + items
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
            merged.insert(len(head), _elision_marker(n_dropped))
        return merged

    # -- Char-budget fitting --------------------------------------------------

    def _fit_chars(
        self,
        items: List[str],
        char_cap: int,
        floor: int,
        focus_topic: Optional[str],
        head_keep: int = 0,
    ) -> List[str]:
        """Shrink a section to *char_cap* by dropping WHOLE oldest items.

        This is the budget mechanism that replaced the old tail hard-cut:
        the cut destroyed whichever sections rendered last (the tool index
        and current position — exactly the anti-amnesia data), and on
        self-merge the truncated digest was re-ingested as ground truth, so
        everything newer than the surviving head was lost permanently with
        no marker.  Dropping whole items oldest-first keeps every section
        alive, keeps the NEWEST content, and accounts for every dropped
        item in one explicit elision marker — a digest must never silently
        truncate.

        Order of sacrifice within the section: non-focus items first
        (oldest-first, after *head_keep*), then focus-pinned items
        (pinning is a preference, the cap is the constraint).
        """
        carried = 0
        real: List[str] = []
        for line in items:
            m = _ELIDE_RE.match(line)
            if m:
                carried += int(m.group(1))
            else:
                real.append(line)
        topic = (focus_topic or "").casefold()
        dropped = carried

        def total() -> int:
            return (sum(len(l) + 1 for l in real)
                    + (_ELIDE_MARKER_COST if dropped else 0))

        floor = max(floor, 1)
        while len(real) > max(floor, head_keep) and total() > char_cap:
            victim = None
            for i in range(head_keep, len(real)):
                if topic and topic in real[i].casefold():
                    continue
                victim = i
                break
            if victim is None:
                # Everything past the head is focus-pinned — sacrifice the
                # oldest pinned item rather than stalling above the cap.
                victim = min(head_keep, len(real) - 1)
            real.pop(victim)
            dropped += 1
        if dropped:
            real.insert(min(head_keep, len(real)), _elision_marker(dropped))
        return real

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
        shape = self._shape()
        budget = shape["budget"]
        head_keep = shape["user_head_keep"]

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

        # Phase 1 — apportion the budget across sections by fixed shares so
        # no section can starve the others (a wall of stale verbatim user
        # quotes must never crowd out the tool index / current position).
        # Chrome (header, section titles, position, footer) is measured, not
        # guessed, by rendering with the lists emptied.
        _u, _s, _t, _l = users, spine, tool_lines, legacy_block
        users, spine, tool_lines, legacy_block = [], [], [], ""
        chrome = len(render_once()) + (len(_SEC_LEGACY) + 2 if _l else 0)
        users, spine, tool_lines, legacy_block = _u, _s, _t, _l
        avail = max(budget - chrome, 1000)
        shares = _SHARES_WITH_LEGACY if legacy_block else _SHARES_NO_LEGACY

        if legacy_block:
            legacy_block = _single_line(
                legacy_block, max(200, int(avail * shares["legacy"])),
            )
        users = self._fit_chars(
            users, int(avail * shares["users"]),
            head_keep + _FLOOR_USERS, focus_topic, head_keep=head_keep,
        )
        spine = self._fit_chars(
            spine, int(avail * shares["spine"]), _FLOOR_SPINE, focus_topic,
        )
        tool_lines = self._fit_chars(
            tool_lines, int(avail * shares["tools"]), _FLOOR_TOOLS, focus_topic,
        )

        # Phase 2 — mop-up: markers and rounding can leave a small overshoot;
        # unused share (an empty section) is simply headroom.  Shrink the
        # largest still-shrinkable section toward its absolute floor.  Each
        # pass drops at least one whole item, so this terminates.
        def _real_len(ls: List[str]) -> int:
            return sum(1 for l in ls if not _ELIDE_RE.match(l))

        body = render_once()
        while len(body) > budget:
            candidates = [
                (sum(len(l) for l in spine), "spine"),
                (sum(len(l) for l in tool_lines), "tools"),
                (sum(len(l) for l in users), "users"),
            ]
            candidates.sort(reverse=True)
            for _, name in candidates:
                if name == "spine" and _real_len(spine) > 1:
                    spine = self._fit_chars(
                        spine, sum(len(l) for l in spine) * 3 // 4, 1, focus_topic,
                    )
                    break
                if name == "tools" and _real_len(tool_lines) > 1:
                    tool_lines = self._fit_chars(
                        tool_lines, sum(len(l) for l in tool_lines) * 3 // 4,
                        1, focus_topic,
                    )
                    break
                if name == "users" and _real_len(users) > 2:
                    users = self._fit_chars(
                        users, sum(len(l) for l in users) * 3 // 4, 2,
                        focus_topic, head_keep=min(head_keep, 2),
                    )
                    break
            else:
                if len(legacy_block) > 200:
                    legacy_block = _single_line(legacy_block, len(legacy_block) // 2)
                else:
                    break  # nothing left to shrink — fall through to last resort
            body = render_once()

        # Last resort — structurally unreachable (the all-floors render fits
        # inside _BUDGET_MIN by construction; pinned by test), kept as a
        # never-overflow guarantee.  Cut at a line boundary so no half-item
        # can be re-ingested as content on self-merge.
        if len(body) > budget:
            logger.warning(
                "mechanical digest exceeded budget at section floors "
                "(%d > %d) — hard-cutting at line boundary",
                len(body), budget,
            )
            cut = body[:budget]
            cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
            body = cut.rstrip() + "\n" + _HARD_CUT_MARKER
        return body


def register(ctx) -> None:
    ctx.register_context_engine(MechanicalDigestEngine())
