"""Tests for agent/local_runtime.py — endpoint gate + prefill-aware floors."""

import threading
import time

import pytest

import agent.local_runtime as lr


LOCAL_URL = "http://127.0.0.1:8080/v1"
CLOUD_URL = "https://api.openai.com/v1"


@pytest.fixture(autouse=True)
def _fresh_gate_state(monkeypatch):
    """Isolate lock table and config between tests."""
    monkeypatch.setattr(lr, "_LOCKS", {})
    monkeypatch.setattr(lr, "_read_cfg", lambda: {})
    lr.set_task_label("")
    yield


def _enable(monkeypatch, **overrides):
    cfg = {"single_flight": True, "busy_skip_wait": 0.1, "gate_wait_timeout": 900}
    cfg.update(overrides)
    monkeypatch.setattr(lr, "_read_cfg", lambda: cfg)
    return cfg


# ── gate on/off matrix ─────────────────────────────────────────────────


def test_gate_noop_when_disabled():
    gate = lr.EndpointGate(LOCAL_URL, purpose="main")
    with gate:
        assert gate._lock is None
        assert not gate._acquired


def test_gate_noop_for_cloud_endpoint(monkeypatch):
    _enable(monkeypatch)
    gate = lr.EndpointGate(CLOUD_URL, purpose="main")
    with gate:
        assert gate._lock is None


def test_gate_acquires_for_local_endpoint(monkeypatch):
    _enable(monkeypatch)
    gate = lr.EndpointGate(LOCAL_URL, purpose="main")
    with gate:
        assert gate._acquired
    assert not gate._acquired


def test_extra_gate_hosts_governs_non_local(monkeypatch):
    _enable(monkeypatch, gate_hosts=["inference.example.com"])
    gate = lr.EndpointGate("https://inference.example.com/v1", purpose="main")
    with gate:
        assert gate._acquired


# ── serialization semantics ────────────────────────────────────────────


def test_gate_serializes_across_threads(monkeypatch):
    _enable(monkeypatch)
    order = []
    first_holds = threading.Event()
    release_first = threading.Event()

    def _first():
        with lr.EndpointGate(LOCAL_URL, purpose="main"):
            order.append("first-in")
            first_holds.set()
            release_first.wait(timeout=5)
            order.append("first-out")

    def _second():
        first_holds.wait(timeout=5)
        with lr.EndpointGate(LOCAL_URL, purpose="main"):
            order.append("second-in")

    t1 = threading.Thread(target=_first)
    t2 = threading.Thread(target=_second)
    t1.start()
    t2.start()
    first_holds.wait(timeout=5)
    time.sleep(0.2)  # give second a chance to (incorrectly) enter
    assert order == ["first-in"]
    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order == ["first-in", "first-out", "second-in"]


def test_gate_reentrant_same_thread(monkeypatch):
    _enable(monkeypatch)
    with lr.EndpointGate(LOCAL_URL, purpose="main"):
        # Nested call paths (streaming → non-streaming fallback) re-enter.
        with lr.EndpointGate(LOCAL_URL, purpose="main") as inner:
            assert inner._acquired


def test_skippable_task_raises_busy(monkeypatch):
    _enable(monkeypatch)
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with lr.EndpointGate(LOCAL_URL, purpose="main"):
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_hold)
    t.start()
    held.wait(timeout=5)
    try:
        with pytest.raises(lr.GateBusyError):
            with lr.EndpointGate(LOCAL_URL, purpose="title_generation"):
                pytest.fail("skippable task must not enter a busy gate")
    finally:
        release.set()
        t.join(timeout=5)


def test_non_skippable_fails_open_after_wait(monkeypatch):
    _enable(monkeypatch, gate_wait_timeout=0.1)
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with lr.EndpointGate(LOCAL_URL, purpose="main"):
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_hold)
    t.start()
    held.wait(timeout=5)
    try:
        gate = lr.EndpointGate(LOCAL_URL, purpose="main")
        with gate:
            assert not gate._acquired  # proceeded without exclusivity
    finally:
        release.set()
        t.join(timeout=5)


def test_skip_list_configurable(monkeypatch):
    _enable(monkeypatch, gate_skip_when_busy=["session_search"])
    # title_generation is no longer skippable under the custom list — it
    # queues (and immediately acquires here since the gate is free).
    with lr.EndpointGate(LOCAL_URL, purpose="title_generation") as gate:
        assert gate._acquired


# ── gated client wrapper ───────────────────────────────────────────────


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("stream"):
            return iter([{"chunk": 1}, {"chunk": 2}])
        return {"ok": True}


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions()


def test_gate_openai_client_passthrough_result(monkeypatch):
    _enable(monkeypatch)
    client = lr.gate_openai_client(_FakeClient(), LOCAL_URL)
    assert client.chat.completions.create(model="m") == {"ok": True}


def test_gate_openai_client_idempotent(monkeypatch):
    _enable(monkeypatch)
    client = _FakeClient()
    lr.gate_openai_client(client, LOCAL_URL)
    wrapped_once = client.chat.completions.create
    lr.gate_openai_client(client, LOCAL_URL)
    assert client.chat.completions.create is wrapped_once


def test_gated_client_skips_busy_aux_without_http(monkeypatch):
    _enable(monkeypatch)
    client = _FakeClient()
    inner = client.chat.completions
    lr.gate_openai_client(client, LOCAL_URL)
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with lr.EndpointGate(LOCAL_URL, purpose="main"):
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_hold)
    t.start()
    held.wait(timeout=5)
    try:
        lr.set_task_label("title_generation")
        with pytest.raises(lr.GateBusyError):
            client.chat.completions.create(model="m")
        assert inner.calls == 0  # busy backend never saw the request
    finally:
        lr.set_task_label("")
        release.set()
        t.join(timeout=5)


def test_gated_stream_holds_until_exhausted(monkeypatch):
    _enable(monkeypatch)
    client = lr.gate_openai_client(_FakeClient(), LOCAL_URL)
    stream = client.chat.completions.create(model="m", stream=True)
    lock = lr._lock_for(lr._host_key(LOCAL_URL))
    # Gate is held while the stream is unconsumed: a skippable probe from
    # another thread must see busy.
    probe_result = {}

    def _probe():
        try:
            with lr.EndpointGate(LOCAL_URL, purpose="title_generation"):
                probe_result["entered"] = True
        except lr.GateBusyError:
            probe_result["busy"] = True

    t = threading.Thread(target=_probe)
    t.start()
    t.join(timeout=5)
    assert probe_result == {"busy": True}
    chunks = list(stream)
    assert len(chunks) == 2
    # Released after exhaustion.
    assert lock.acquire(timeout=1)
    lock.release()


def test_gated_stream_close_releases(monkeypatch):
    _enable(monkeypatch)
    client = lr.gate_openai_client(_FakeClient(), LOCAL_URL)
    stream = client.chat.completions.create(model="m", stream=True)
    stream.close()
    lock = lr._lock_for(lr._host_key(LOCAL_URL))
    assert lock.acquire(timeout=1)
    lock.release()


# ── prefill-aware floors ───────────────────────────────────────────────


def test_floors_off_without_prefill_tps(monkeypatch):
    _enable(monkeypatch)
    assert lr.prefill_floor_seconds(10_000) is None
    assert lr.local_aux_timeout_floor(LOCAL_URL, [{"role": "user", "content": "hi"}]) is None
    assert lr.bounded_local_stale_seconds(LOCAL_URL, 10_000) is None


def test_prefill_floor_math(monkeypatch):
    _enable(monkeypatch, prefill_tps=50, timeout_margin=30, decode_budget=900)
    assert lr.prefill_floor_seconds(15_000) == pytest.approx(30 + 15_000 / 50)
    assert lr.bounded_local_stale_seconds(LOCAL_URL, 15_000) == pytest.approx(
        30 + 15_000 / 50 + 900
    )
    # Aux floor adds a bounded response allowance, not the full decode budget.
    msgs = [{"role": "user", "content": "x" * 3500}]  # ~1000 estimated tokens
    est = lr.estimate_messages_tokens(msgs)
    assert lr.local_aux_timeout_floor(LOCAL_URL, msgs) == pytest.approx(
        30 + est / 50 + 120, rel=0.01
    )


def test_floors_ignore_cloud_endpoints(monkeypatch):
    _enable(monkeypatch, prefill_tps=50)
    assert lr.local_aux_timeout_floor(CLOUD_URL, [{"role": "user", "content": "hi"}]) is None
    assert lr.bounded_local_stale_seconds(CLOUD_URL, 10_000) is None


def test_estimate_messages_tokens_scales_with_content():
    small = lr.estimate_messages_tokens([{"role": "user", "content": "hi"}])
    large = lr.estimate_messages_tokens([{"role": "user", "content": "hi" * 5000}])
    assert small < large
    assert large >= 2500  # 10k chars / 3.5 ≈ 2857


def test_task_label_roundtrip():
    lr.set_task_label("title_generation")
    assert lr.current_task_label() == "title_generation"
    lr.set_task_label("")
    assert lr.current_task_label() == ""
