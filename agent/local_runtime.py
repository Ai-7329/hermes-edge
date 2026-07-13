"""Local-runtime governance for slow shared backends (edge fork).

Cloud providers absorb concurrency; a llama.cpp/Ollama/vLLM server on the
same low-spec box does not.  Two structural failure modes follow when the
agent treats a local endpoint like a cloud one:

1. **Cache eviction by interleaving.**  Auxiliary calls (title generation,
   background review, memory flush, ...) race the main conversation to the
   same server.  On backends whose prefix cache restores only from
   checkpoints (SWA / hybrid-attention models), losing the slot to an
   auxiliary request forces a full re-prefill of the main conversation —
   minutes of wall clock at local prefill speeds.

2. **Structurally unwinnable timeouts.**  Static timeout defaults assume
   cloud prefill latency.  At tens of tokens/second, any request carrying
   more than ``timeout × prefill_tps`` tokens loses before it starts — and
   because the client gives up while the server keeps crunching, the retry
   piles onto a busy server and loses again.

This module provides the two counter-mechanisms, both opt-in and both
inert for non-local endpoints:

* An **endpoint gate**: at most one in-flight request per governed (local)
  endpoint.  Fire-and-forget auxiliary tasks skip when the gate is busy
  instead of queueing; interactive work waits.  Reentrant per thread so
  wrapped call paths that fall back into each other (streaming →
  non-streaming) cannot self-deadlock.  Fail-open: a gate wait that
  exceeds its budget proceeds without exclusivity rather than wedging.

* **Prefill-aware timeout floors**: deadlines derived from request size and
  a measured prompt-processing speed (``prefill_tps``), replacing both the
  too-small static auxiliary defaults and the unbounded "local endpoints
  never go stale" escape hatch — a wedged server must still be detected in
  bounded time for unattended long-running operation.

Config (all optional, under a top-level ``local_runtime:`` section)::

    local_runtime:
      single_flight: true        # serialize requests per local endpoint
      gate_wait_timeout: 900     # seconds a blocking caller waits for the gate
      busy_skip_wait: 15         # seconds a skippable task waits before skipping
      gate_skip_when_busy:       # aux task names that skip instead of queue
        - title_generation
        - background_review
        - curator
        - profile_describer
        - tts_audio_tags
      gate_hosts: []             # extra hostnames to govern (beyond auto-local)
      prefill_tps: 50            # measured prompt tok/s of the backend; 0 = off
      decode_budget: 900         # max expected generation seconds (stale bound)
      timeout_margin: 30         # fixed slack added to computed floors

``single_flight`` and ``prefill_tps`` are independent: either can be enabled
without the other.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Task label for the current logical operation (set by auxiliary call paths,
# read by the gated client wrapper to decide queue-vs-skip and for logging).
# A context variable rather than a parameter because the wrapped OpenAI
# client's ``create()`` signature cannot carry hermes-specific arguments.
_TASK_LABEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hermes_local_runtime_task", default=""
)

_DEFAULT_SKIP_WHEN_BUSY = (
    "title_generation",
    "background_review",
    "curator",
    "profile_describer",
    "tts_audio_tags",
    # Post-compaction cache warmup is pure opportunism: if the main
    # conversation (or anything else) is already talking to the server,
    # the warmup's job is being done by that request — skip, never queue.
    "compression_warmup",
)

# Approximate bytes-per-token for mixed prose / code / JSON payloads.  Used
# only for timeout floors (safety margins absorb the error), never for
# billing or context-window math.
_CHARS_PER_TOKEN = 3.5


def set_task_label(label: str) -> None:
    """Record the auxiliary task driving subsequent LLM calls on this thread.

    Deliberately not scoped/reset: every auxiliary entry point sets its own
    label on entry, so a stale value can only mislabel logging between calls,
    never gate semantics for the main conversation (which passes an explicit
    purpose).
    """
    try:
        _TASK_LABEL.set(str(label or ""))
    except Exception:
        pass


def current_task_label() -> str:
    try:
        return _TASK_LABEL.get()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _read_cfg() -> Dict[str, Any]:
    """Return the ``local_runtime`` config section (cached upstream)."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    section = cfg.get("local_runtime") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def _cfg_float(key: str, default: float) -> float:
    raw = _read_cfg().get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value


def single_flight_enabled() -> bool:
    return bool(_read_cfg().get("single_flight", False))


def prefill_tps() -> Optional[float]:
    """Measured prompt-processing speed of the local backend, tok/s.

    Returns ``None`` when unset or non-positive — every consumer treats
    that as "feature off, keep upstream behavior".
    """
    raw = _read_cfg().get("prefill_tps", 0)
    try:
        tps = float(raw)
    except (TypeError, ValueError):
        return None
    return tps if tps > 0 else None


def _timeout_margin() -> float:
    return max(_cfg_float("timeout_margin", 30.0), 0.0)


def _decode_budget() -> float:
    return max(_cfg_float("decode_budget", 900.0), 0.0)


def _gate_wait_timeout() -> float:
    return max(_cfg_float("gate_wait_timeout", 900.0), 1.0)


def _busy_skip_wait() -> float:
    return max(_cfg_float("busy_skip_wait", 15.0), 0.0)


def _skip_when_busy_tasks() -> frozenset:
    raw = _read_cfg().get("gate_skip_when_busy", None)
    if isinstance(raw, (list, tuple)):
        return frozenset(str(item).strip() for item in raw if str(item).strip())
    return frozenset(_DEFAULT_SKIP_WHEN_BUSY)


def _extra_gate_hosts() -> frozenset:
    raw = _read_cfg().get("gate_hosts", None)
    if isinstance(raw, (list, tuple)):
        return frozenset(str(item).strip().lower() for item in raw if str(item).strip())
    return frozenset()


# ---------------------------------------------------------------------------
# Endpoint gate
# ---------------------------------------------------------------------------


class GateBusyError(RuntimeError):
    """A skippable auxiliary task found the local endpoint busy.

    Raised instead of issuing the HTTP request at all, so the busy backend
    never sees (and never has its prefix cache disturbed by) a request the
    caller was willing to drop.  Callers treat it like any transient
    auxiliary failure.
    """

    def __init__(self, host: str, task: str, waited: float):
        super().__init__(
            f"local endpoint {host} busy after {waited:.1f}s — "
            f"skipping deferrable task {task or 'aux'!r}"
        )
        self.host = host
        self.task = task


def _host_key(base_url: str) -> str:
    """Normalize a base URL to a host:port gate key."""
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    url = raw if "://" in raw else f"http://{raw}"
    try:
        parsed = urlparse(url)
    except Exception:
        return raw.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return raw.lower()
    port = parsed.port
    return f"{host}:{port}" if port else host


def _is_governed_endpoint(base_url: str) -> bool:
    key = _host_key(base_url)
    if not key:
        return False
    host = key.split(":", 1)[0]
    if host in _extra_gate_hosts():
        return True
    try:
        from agent.model_metadata import is_local_endpoint

        return bool(is_local_endpoint(base_url))
    except Exception:
        return False


_LOCKS_GUARD = threading.Lock()
_LOCKS: Dict[str, threading.RLock] = {}


def _lock_for(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


class EndpointGate:
    """Context manager holding the single-flight lock for one request.

    No-op (but still a valid context manager) when governance is off or the
    endpoint is not local.  Reentrant per thread via ``threading.RLock`` —
    nested call paths (streaming falling back to non-streaming, codex
    delegation) re-enter without deadlock.
    """

    def __init__(self, base_url: str, *, purpose: str = "", wait_timeout: Optional[float] = None):
        self._purpose = purpose or current_task_label() or "call"
        self._wait_timeout = wait_timeout
        self._lock: Optional[threading.RLock] = None
        self._acquired = False
        self._key = ""
        if single_flight_enabled() and _is_governed_endpoint(base_url):
            self._key = _host_key(base_url)
            self._lock = _lock_for(self._key)

    def __enter__(self) -> "EndpointGate":
        if self._lock is None:
            return self
        skippable = self._purpose in _skip_when_busy_tasks()
        budget = self._wait_timeout
        if budget is None:
            budget = _busy_skip_wait() if skippable else _gate_wait_timeout()
        started = time.monotonic()
        self._acquired = self._lock.acquire(timeout=max(budget, 0.0))
        waited = time.monotonic() - started
        if not self._acquired:
            if skippable:
                raise GateBusyError(self._key, self._purpose, waited)
            # Fail open: exclusivity is an optimization, wedging the
            # conversation is not an acceptable price for it.
            logger.warning(
                "endpoint gate: %s waited %.0fs for %s without acquiring — "
                "proceeding without exclusivity",
                self._purpose, waited, self._key,
            )
            return self
        if waited > 1.0:
            logger.info(
                "endpoint gate: %s waited %.1fs for %s", self._purpose, waited, self._key
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        if self._acquired and self._lock is not None:
            self._acquired = False
            try:
                self._lock.release()
            except RuntimeError:
                logger.debug("endpoint gate: release without ownership for %s", self._key)


def endpoint_gate(base_url: str, *, purpose: str = "", wait_timeout: Optional[float] = None) -> EndpointGate:
    """Build the single-flight gate for ``base_url`` (no-op when ungoverned)."""
    return EndpointGate(base_url, purpose=purpose, wait_timeout=wait_timeout)


class _GatedStream:
    """Stream proxy that holds the endpoint gate until the stream ends.

    A streaming ``create()`` returns before generation finishes; releasing
    the gate at that point would let another request preempt the slot
    mid-generation.  The proxy releases exactly once — on exhaustion, on
    error, on ``close()``, on context-manager exit, or at GC as a backstop.
    """

    def __init__(self, stream: Any, gate: EndpointGate):
        self._stream = stream
        self._gate = gate

    def __iter__(self):
        try:
            for chunk in self._stream:
                yield chunk
        finally:
            self._release()

    def __enter__(self):
        entered = getattr(self._stream, "__enter__", None)
        if entered is not None:
            entered()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            exiter = getattr(self._stream, "__exit__", None)
            if exiter is not None:
                return bool(exiter(exc_type, exc, tb))
            return False
        finally:
            self._release()

    def close(self):
        try:
            closer = getattr(self._stream, "close", None)
            if closer is not None:
                closer()
        finally:
            self._release()

    def _release(self):
        gate, self._gate = self._gate, None
        if gate is not None:
            gate.release()

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            self._release()
        except Exception:
            pass


def gate_openai_client(client: Any, base_url: str) -> Any:
    """Wrap an OpenAI-compatible client so ``chat.completions.create`` is gated.

    Idempotent; returns the client unchanged when wrapping is impossible or
    the endpoint is never governed.  The gate decision itself is re-evaluated
    per call (config is live), so a wrapped client on a disabled config is a
    plain passthrough.
    """
    if client is None:
        return client
    try:
        completions = client.chat.completions
        if getattr(completions, "_hermes_gate_wrapped", False):
            return client
        original_create = completions.create
    except Exception:
        return client

    def _gated_create(*args: Any, **kwargs: Any) -> Any:
        gate = EndpointGate(base_url, purpose=current_task_label() or "aux")
        gate.__enter__()
        try:
            response = original_create(*args, **kwargs)
        except BaseException:
            gate.release()
            raise
        if kwargs.get("stream"):
            return _GatedStream(response, gate)
        gate.release()
        return response

    try:
        completions.create = _gated_create
        completions._hermes_gate_wrapped = True
    except Exception:
        return client
    return client


# ---------------------------------------------------------------------------
# Prefill-aware timeout floors
# ---------------------------------------------------------------------------


def estimate_messages_tokens(messages: Optional[Iterable[Any]]) -> int:
    """Cheap size estimate of a chat payload for timeout floors only."""
    if not messages:
        return 0
    try:
        text = json.dumps(list(messages), ensure_ascii=False, default=str)
    except Exception:
        try:
            text = str(messages)
        except Exception:
            return 0
    return int(len(text) / _CHARS_PER_TOKEN)


def prefill_floor_seconds(est_tokens: int) -> Optional[float]:
    """Seconds the backend legitimately needs to *prefill* ``est_tokens``.

    ``None`` when ``prefill_tps`` is unconfigured (feature off).
    """
    tps = prefill_tps()
    if tps is None:
        return None
    return _timeout_margin() + max(int(est_tokens), 0) / tps


def local_aux_timeout_floor(base_url: str, messages: Optional[Iterable[Any]]) -> Optional[float]:
    """Timeout floor for an auxiliary call against a governed local endpoint.

    Static auxiliary defaults assume cloud prefill.  Locally the request is
    only winnable if the deadline covers its own prefill plus a response
    budget; anything shorter is a guaranteed loss that still costs the
    server a full prefill.  Queue wait is excluded by design — the endpoint
    gate is acquired before the HTTP clock starts.
    """
    if not _is_governed_endpoint(base_url):
        return None
    floor = prefill_floor_seconds(estimate_messages_tokens(messages))
    if floor is None:
        return None
    # Auxiliary responses are short (titles, digests, verdicts); a fixed
    # fraction of the decode budget is plenty.
    return floor + min(_decode_budget(), 120.0)


def compression_warmup_enabled(base_url: str) -> bool:
    """Whether post-compaction cache warmup may run for this endpoint.

    Requires all three: the feature flag, an endpoint the gate governs, and
    single-flight itself — the gate's skip-when-busy semantics are the only
    thing keeping an opportunistic warmup from colliding with the user's
    next message, so warmup without the gate is not offered.
    """
    if not bool(_read_cfg().get("compression_warmup", False)):
        return False
    return single_flight_enabled() and _is_governed_endpoint(base_url)


def bounded_local_stale_seconds(base_url: str, est_tokens: int) -> Optional[float]:
    """Bounded stale/TTFB deadline for a local main-conversation call.

    Replaces the upstream "local endpoints never go stale" infinity: correct
    for interactive use, wrong for unattended long-running operation where a
    wedged server must be detected in bounded time.  The bound covers a full
    prefill of the request plus the decode budget.
    """
    if not _is_governed_endpoint(base_url):
        return None
    floor = prefill_floor_seconds(est_tokens)
    if floor is None:
        return None
    return floor + _decode_budget()
