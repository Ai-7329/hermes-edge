"""Tests for compression.budget_tokens and post-compaction cache warmup."""

import threading
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


LOCAL_URL = "http://127.0.0.1:8080/v1"


def _make_compressor(budget, *, context_length=131072, max_tokens=8192, pct=0.5):
    cfg = {"compression": {"budget_tokens": budget}} if budget else {}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return ContextCompressor(
            model="qwen-local",
            threshold_percent=pct,
            config_context_length=context_length,
            quiet_mode=True,
            base_url=LOCAL_URL,
            max_tokens=max_tokens,
        )


class TestBudgetTokens:
    def test_budget_sets_threshold_directly(self):
        c = _make_compressor(32768)
        # min(budget, window - max_tokens) * pct — no MINIMUM floor, no
        # small-context raise.
        assert c.threshold_tokens == int(32768 * 0.5)
        assert c.threshold_percent == 0.5

    def test_no_budget_keeps_upstream_small_ctx_raise(self):
        c = _make_compressor(0)
        # Upstream: sub-512K windows trigger at >=75% of the input budget.
        assert c.threshold_percent == 0.75
        assert c.threshold_tokens == int((131072 - 8192) * 0.75)

    def test_budget_clamped_to_effective_window(self):
        c = _make_compressor(1_000_000, context_length=131072, max_tokens=8192)
        # A budget above the window falls back to the effective window.
        assert c.threshold_tokens == int((131072 - 8192) * 0.5)

    def test_summary_ceiling_follows_budget(self):
        c = _make_compressor(32768)
        assert c.max_summary_tokens == int(32768 * 0.05)

    def test_update_model_rereads_budget(self):
        c = _make_compressor(32768)
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"compression": {"budget_tokens": 16384}},
        ):
            c.update_model(model="qwen-local", context_length=131072)
        assert c.threshold_tokens == int(16384 * 0.5)

    def test_update_model_budget_removed_restores_upstream(self):
        c = _make_compressor(32768)
        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            c.update_model(model="qwen-local", context_length=131072)
        assert c.threshold_percent == 0.75

    def test_mechanical_engine_inherits_budget(self):
        from plugins.context_engine.mechanical import MechanicalDigestEngine

        cfg = {"compression": {"budget_tokens": 32768}}
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            engine = MechanicalDigestEngine()
            # agent_init drives plugin engines through update_model() — the
            # budget must survive that same path.
            engine.update_model(model="qwen-local", context_length=131072)
        assert engine.threshold_tokens == int(32768 * 0.5)
        assert engine.threshold_percent == 0.5


class _WarmupAgentStub:
    """Minimal surface for AIAgent._maybe_schedule_compression_warmup."""

    base_url = LOCAL_URL
    api_mode = "chat_completions"
    model = "qwen-local"
    ephemeral_system_prompt = None
    prefill_messages = None

    def __init__(self):
        self.created = []
        self.closed = []
        self.done = threading.Event()

    def _should_sanitize_tool_calls(self):
        return False

    def _sanitize_tool_calls_for_strict_api(self, msg, model=None):
        return msg

    def _sanitize_api_messages(self, msgs):
        return msgs

    def _drop_thinking_only_and_merge_users(self, msgs, drop_codex_reasoning_items=True):
        return msgs

    def _create_request_openai_client(self, reason="", api_kwargs=None):
        stub = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                stub.created.append(kwargs)
                stub.done.set()
                return {"ok": True}

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        return _Client()

    def _close_request_openai_client(self, client, reason=""):
        self.closed.append(reason)


def _run_warmup(stub, cfg):
    from run_agent import AIAgent

    def _fake_build(agent, api_messages):
        return {"model": agent.model, "messages": api_messages, "max_tokens": 512}

    with (
        patch("agent.local_runtime._read_cfg", return_value=cfg),
        patch("agent.chat_completion_helpers.build_api_kwargs", _fake_build),
    ):
        AIAgent._maybe_schedule_compression_warmup(
            stub,
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            "SYSTEM PROMPT",
        )
        # The worker thread reads config too — wait inside the patch scope.
        fired = stub.done.wait(timeout=5)
    return fired


class TestCompressionWarmup:
    def test_disabled_by_default(self):
        stub = _WarmupAgentStub()
        fired = _run_warmup(stub, {"single_flight": True})
        assert not fired
        assert stub.created == []

    def test_warmup_fires_single_token_request(self):
        stub = _WarmupAgentStub()
        cfg = {"single_flight": True, "compression_warmup": True}
        fired = _run_warmup(stub, cfg)
        assert fired
        assert len(stub.created) == 1
        kwargs = stub.created[0]
        assert kwargs["max_tokens"] == 1
        assert "stream" not in kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "SYSTEM PROMPT"}
        assert stub.closed == ["compression_warmup_done"]

    def test_requires_single_flight(self):
        stub = _WarmupAgentStub()
        fired = _run_warmup(stub, {"compression_warmup": True})
        assert not fired

    def test_skips_with_ephemeral_system_prompt(self):
        stub = _WarmupAgentStub()
        stub.ephemeral_system_prompt = "extra"
        cfg = {"single_flight": True, "compression_warmup": True}
        fired = _run_warmup(stub, cfg)
        assert not fired

    def test_skips_non_chat_api_modes(self):
        stub = _WarmupAgentStub()
        stub.api_mode = "anthropic_messages"
        cfg = {"single_flight": True, "compression_warmup": True}
        fired = _run_warmup(stub, cfg)
        assert not fired
