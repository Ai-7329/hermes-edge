"""Regression tests for the mechanical (LLM-free) context engine.

The failure mode this engine exists to kill: auto-compaction summarizes with
an auxiliary LLM at the moment the context is fullest; on a slow/single-slot
local backend every summary attempt times out, timeouts abort compaction
(nothing dropped), and the session livelocks compress→timeout→compress.
These tests pin the two structural properties that make that impossible here:
compaction never touches an LLM, and the summary can never be None.
"""

import json

import pytest

import agent.context_compressor as cc_mod
from plugins.context_engine import load_context_engine


@pytest.fixture()
def engine(monkeypatch):
    def _boom(*args, **kwargs):  # tripwire: any LLM call fails the test
        raise AssertionError("LLM was called during mechanical compaction")
    monkeypatch.setattr(cc_mod, "call_llm", _boom)
    eng = load_context_engine("mechanical")
    assert eng is not None, "mechanical engine failed to load via plugin loader"
    eng.update_model(model="test-model", context_length=131072)
    return eng


def _tool_call(cid, name, **args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _session(n_probe_repeats=4):
    """Synthetic session shaped like the real livelock: one instruction,
    then repeated identical blocked probes.

    Shape matters: the first 3 non-system messages sit in the protected
    head and the last ~5 in the protected tail (both stay live, correctly
    absent from any digest), so the contract turn and the probe cluster
    are placed in the compressible middle.
    """
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "おはよう"},
        {"role": "assistant", "content": "おはようございます。"},
        {"role": "assistant", "content": "準備完了。" + "前置き。" * 30},
        # ---- head boundary (system + 3 non-system) ----
        {"role": "user", "content": "モデルを調査して、仮説→検証でアイデア提案まで。"},
        {"role": "assistant", "content": "調査計画を立てた。" + "手順の説明。" * 40},
    ]
    for i in range(n_probe_repeats):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [_tool_call(f"c{i}", "terminal",
                                               command="python3 -c 'import pandas'")]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": json.dumps({"output": "", "exit_code": -1,
                                            "error": "BLOCKED: approval timed out"})})
    # ---- enough trailing turns that the protected tail stays behind the probes ----
    for j in range(3):
        msgs.append({"role": "assistant", "content": f"別経路の検討 {j}。" + "詳細。" * 60})
    msgs.append({"role": "user", "content": "続けて"})
    msgs.append({"role": "assistant", "content": "続行する。" + "内容。" * 60})
    return msgs


def test_compaction_without_llm(engine):
    """compress() completes with call_llm tripwired — the livelock's entry
    point (summary timeout) is structurally unreachable."""
    msgs = _session()
    engine.tail_token_budget = 60
    engine.protect_last_n = 3
    out = engine.compress(msgs, current_tokens=100_000)
    digests = [m for m in out
               if m.get("content") and cc_mod.SUMMARY_PREFIX in str(m["content"])]
    assert len(digests) == 1
    assert len(out) < len(msgs)


def test_digest_preserves_contracts_and_probe_repetition(engine):
    msgs = _session()
    engine.tail_token_budget = 60
    engine.protect_last_n = 3
    out = engine.compress(msgs, current_tokens=100_000)
    digest = next(str(m["content"]) for m in out
                  if m.get("content") and cc_mod.SUMMARY_PREFIX in str(m["content"]))
    # user turn verbatim (contract — originates outside the model)
    assert "モデルを調査して、仮説→検証でアイデア提案まで。" in digest
    # the blocked probe is visible with its outcome and repeat count
    assert "python3 -c 'import pandas'" in digest
    assert "BLOCKED" in digest
    assert "×4 runs" in digest


def test_generate_summary_never_none(engine):
    """None would resurrect the abort/fallback/cooldown failure paths."""
    assert engine._generate_summary([]) is not None
    # even on a poisoned window the engine must emit a counting stub
    poisoned = [{"role": "assistant", "tool_calls": [object()], "content": None}]
    out = engine._generate_summary(poisoned)
    assert out is not None and cc_mod.SUMMARY_PREFIX in out


def test_recompression_merges_own_digest_without_nesting(engine):
    engine.tail_token_budget = 60
    engine.protect_last_n = 3
    out1 = engine.compress(_session(), current_tokens=100_000)
    # grow the tail and compress again
    grown = list(out1)
    for i in range(10):
        grown.append({"role": "assistant", "content": f"追加の報告 {i}。" + "x" * 200})
    grown.append({"role": "user", "content": "さらに続けて"})
    grown.append({"role": "assistant", "content": "承知。" + "y" * 300})
    out2 = engine.compress(grown, current_tokens=100_000)
    digests = [m for m in out2
               if m.get("content") and cc_mod.SUMMARY_PREFIX in str(m["content"])]
    assert len(digests) == 1
    d2 = str(digests[0]["content"])
    # exactly one generator marker → previous digest merged, not nested
    assert d2.count("# mechanical context digest") == 1
    # round-1 contract survives the merge
    assert "モデルを調査して、仮説→検証でアイデア提案まで。" in d2


def test_tool_pairs_stay_well_formed(engine):
    engine.tail_token_budget = 60
    engine.protect_last_n = 3
    out = engine.compress(_session(), current_tokens=100_000)
    call_ids = {tc.get("id") for m in out if m.get("role") == "assistant"
                for tc in m.get("tool_calls") or []}
    orphans = [m for m in out if m.get("role") == "tool"
               and m.get("tool_call_id") not in call_ids]
    assert orphans == []


def _render_in_thread(engine, users, spine, tools, focus_topic):
    """Run _render on a daemon thread so a shrink-loop regression fails the
    test instead of hanging the whole suite."""
    import threading
    result = {}

    def _run():
        result["body"] = engine._render(
            users, spine, tools, position="tail position",
            legacy_block="", n_turns=1, focus_topic=focus_topic,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "_render did not terminate (shrink-loop livelock)"
    return result["body"]


def test_render_terminates_when_focus_pins_every_line(engine):
    """All-pinned sections once made _elide a no-op: the budget loop had no
    progress guarantee and spun forever — a livelock inside the engine that
    exists to remove a livelock."""
    users = [f"- 「probe request {i} : " + "u" * 150 + "」" for i in range(60)]
    spine = [f"- probe report {i} : " + "s" * 200 for i in range(40)]
    tools = [f"- terminal `probe cmd {i}` → exit 0" for i in range(60)]
    engine.max_summary_tokens = 1000  # char budget floor: 6000
    body = _render_in_thread(engine, users, spine, tools, focus_topic="probe")
    assert len(body) <= 6000


def test_render_terminates_at_section_floors_without_focus(engine):
    """A small summary budget with oversized sections must converge by
    whole-item elision — never by the last-resort hard cut."""
    users = [f"- 「turn {i} : " + "u" * 370 + "」" for i in range(60)]
    spine = [f"- report {i} : " + "s" * 200 for i in range(40)]
    tools = [f"- terminal `cmd {i}` → exit 0, output 3 chars" for i in range(60)]
    engine.max_summary_tokens = 1000
    body = _render_in_thread(engine, users, spine, tools, focus_topic=None)
    assert len(body) <= 6000
    assert "hard-cut" not in body


# ---------------------------------------------------------------------------
# Regressions for the edge-profile digest collapse: with a small char budget
# (compression.budget_tokens shrinks max_summary_tokens) the old renderer
# tail-hard-cut every digest from the 2nd compaction on — amputating the tool
# index and current position — and self-merge then re-ingested the truncated
# digest, so all recent user turns were lost permanently with no marker.
# ---------------------------------------------------------------------------

_ALL_SECTION_HEADERS = (
    "## User turns", "## Assistant report spine",
    "## Tool execution index", "## Current position",
)


def _long_session_window(base, n_turns=40):
    """One compression window of a busy session: user contracts + reports +
    tool traffic, sized like real edge traffic (tool_output.max_bytes ≈ 6KB)."""
    msgs = []
    for i in range(base, base + n_turns):
        if i % 4 == 0:
            msgs.append({"role": "user",
                         "content": f"タスク{i}: group_{i % 7}.py を修正して。" + "要件詳細。" * 30})
        msgs.append({"role": "assistant",
                     "content": f"ターン{i}の分析結果。" + "解析内容。" * 40,
                     "tool_calls": [_tool_call(f"e{i}", "terminal",
                                               command=f"pytest tests/group_{i % 7}.py -x")]})
        msgs.append({"role": "tool", "tool_call_id": f"e{i}",
                     "content": json.dumps({"output": f"output {i} " * 120,
                                            "exit_code": i % 3 and 1 or 0})})
        msgs.append({"role": "assistant",
                     "content": f"ターン{i}の結論: group_{i % 7} を更新済み。" + "状態説明。" * 20})
    return msgs


def test_small_budget_digest_keeps_every_section(engine):
    """Edge profile shape: budget_tokens: 32768 → max_summary_tokens 1638.
    Every section must survive within budget with no hard cut."""
    engine.max_summary_tokens = 1638
    digest = engine._generate_summary(_long_session_window(0, 60))
    budget = engine._shape()["budget"]
    for sec in _ALL_SECTION_HEADERS:
        assert sec in digest, f"{sec} amputated from small-budget digest"
    assert "hard-cut" not in digest
    body = digest.split("# mechanical context digest", 1)[1]
    assert len(body) <= budget


def test_small_budget_self_merge_keeps_newest_and_marks_elision(engine):
    """The collapse attractor: under repeated self-merge the old renderer
    converged to a wall of the OLDEST user turns and silently destroyed
    everything newer.  After the fix, the newest turns must survive every
    cycle, all sections must stay present, and any dropped items must be
    accounted for by an explicit elision marker."""
    import re
    engine.max_summary_tokens = 1638
    digest = None
    for cycle in range(8):
        digest = engine._generate_summary(_long_session_window(cycle * 40))
    latest_task = 7 * 40 + 36  # newest user turn of the final window
    assert f"タスク{latest_task}:" in digest, "newest user turn lost on self-merge"
    for sec in _ALL_SECTION_HEADERS:
        assert sec in digest
    assert "hard-cut" not in digest
    # 80 user turns were ingested across cycles; whatever no longer fits must
    # be declared, not silently dropped.
    user_sec = digest.split("## User turns", 1)[1].split("## ", 1)[0]
    kept = len(re.findall(r"タスク(\d+):", user_sec))
    elided = sum(int(n) for n in re.findall(r"(\d+) earlier item\(s\) elided", user_sec))
    assert kept + elided == 80, f"user turns unaccounted: kept={kept} elided={elided}"
    # the oldest contracts stay pinned at the head (head_keep)
    assert "タスク0:" in user_sec


def test_builder_error_stub_preserves_lineage(engine, monkeypatch):
    """A transient digest-builder error must emit the counting stub WITHOUT
    overwriting _previous_summary — one bad window must not destroy the
    accumulated lineage (contracts, tool index) of every prior compaction."""
    good = engine._generate_summary(
        [{"role": "user", "content": "契約: リリース前に必ずテストを回す"}]
    )
    assert "契約: リリース前に必ずテストを回す" in good
    lineage_before = engine._previous_summary
    monkeypatch.setattr(engine, "_build_digest",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    stub = engine._generate_summary([{"role": "user", "content": "lost window"}])
    assert "digest builder error" in stub
    assert engine._previous_summary == lineage_before, \
        "stub overwrote the accumulated digest lineage"
    monkeypatch.undo()
    nxt = engine._generate_summary([{"role": "user", "content": "next window"}])
    assert "契約: リリース前に必ずテストを回す" in nxt, \
        "lineage did not survive across the stub compaction"


def test_merge_tool_lines_carries_elision_counts():
    """The tool index must account for every run it has ever elided — the
    old merge silently dropped the previous digest's marker counts."""
    from plugins.context_engine.mechanical import (
        MechanicalDigestEngine, _elision_marker,
    )
    prev = [_elision_marker(7),
            "- terminal `old cmd` → exit 0",
            "- terminal `shared cmd` → exit 1"]
    new = ["- terminal `shared cmd` → exit 0", "- terminal `new cmd` → exit 0"]
    merged = MechanicalDigestEngine._merge_tool_lines(prev, new)
    assert merged[0] == _elision_marker(7)
    assert "- terminal `old cmd` → exit 0" in merged
    # newest wins on the shared identity
    assert "- terminal `shared cmd` → exit 0" in merged
    assert "- terminal `shared cmd` → exit 1" not in merged


def test_hard_cut_marker_not_reingested_as_legacy():
    """The hard-cut marker is a rendering artifact; if a (last-resort) cut
    ever lands after the legacy section, self-merge must not replay the
    marker as legacy content forever."""
    from plugins.context_engine.mechanical import (
        MechanicalDigestEngine, _HARD_CUT_MARKER, _SEC_LEGACY,
    )
    body = ("# mechanical context digest\nheader\n\n"
            f"{_SEC_LEGACY}\nreal legacy text\n{_HARD_CUT_MARKER}\n")
    secs = MechanicalDigestEngine._parse_own_digest(body)
    assert secs[_SEC_LEGACY] == ["real legacy text"]


def test_all_floors_render_fits_minimum_budget(engine):
    """Convergence invariant behind the 'hard cut is unreachable' claim: a
    render with every section populated at worst-case item sizes must fit the
    minimum budget via whole-item elision alone."""
    engine.max_summary_tokens = 1  # forces the 6000-char budget floor
    users = [f"- 「{'う' * 398}」" for _ in range(60)]
    spine = [f"- {'す' * 248}" for _ in range(40)]
    tools = [f"- terminal `{'c' * 158}` → {'o' * 118}  (×9 runs, latest shown)"
             for _ in range(60)]
    body = engine._render(users, spine, tools, position="p" * 300,
                          legacy_block="l" * 1500, n_turns=99, focus_topic=None)
    assert len(body) <= 6000
    assert "hard-cut" not in body
    for sec in _ALL_SECTION_HEADERS:
        assert sec in body


def test_placeholder_lines_do_not_accumulate_across_digests(engine):
    """'(none in this window)' is a rendering artifact; on self-merge it must
    not be re-ingested as a real item next to genuine content."""
    w1 = []
    for i in range(3):
        w1.append({"role": "assistant", "content": "",
                   "tool_calls": [_tool_call(f"p{i}", "terminal", command=f"ls {i}")]})
        w1.append({"role": "tool", "tool_call_id": f"p{i}",
                   "content": json.dumps({"output": "ok", "exit_code": 0})})
    d1 = engine._generate_summary(w1)
    assert "(none in this window)" in d1
    w2 = [{"role": "user", "content": "real user turn"},
          {"role": "assistant", "content": "real assistant report " + "r" * 150}]
    d2 = engine._generate_summary(w2)
    user_sec = d2.split("## User turns", 1)[1].split("## ", 1)[0]
    assert "real user turn" in user_sec
    assert "- (none in this window)" not in user_sec
