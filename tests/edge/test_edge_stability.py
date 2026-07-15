"""Pins the long-run stability model (docs/edge/audit/EdgeStability.lean) to
Python and asserts the shipped edge profile satisfies the design requirements
the Lean theorems derive (R1 exclusive tenancy, R2 bounded window, R3 archive).

Two halves, mirroring the audit discipline:

  * ``TestStabilityModel`` recomputes the Lean ``example`` numbers in Python so
    model and implementation are cross-anchored bit-for-bit (the same values
    the kernel checks with ``rfl`` / ``decide``).
  * ``TestProfileSatisfiesRequirements`` reads docs/edge/config.yaml and the
    launch scripts and fails if the shipped profile drifts off R1-R3 — the
    machine-checked reason the profile is shaped the way it is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "docs" / "edge" / "config.yaml"
LAUNCH_SH = REPO / "scripts" / "llama-server-edge.sh"
LAUNCH_BAT = REPO / "scripts" / "llama-server-edge.bat"


# ── model (mirror of EdgeStability.lean) ────────────────────────────────────

def prefill_cost(hit: bool, delta: int, window: int) -> int:
    """EdgeStability.prefillCost — delta on a checkpoint hit, full window on a miss."""
    return delta if hit else window


def prefill_ms(tokens: int, tps: int) -> int:
    """EdgeStability.prefillMs — floor division, matching Nat semantics."""
    return tokens * 1000 // tps


def accessible(resident: int, archived: int) -> int:
    return resident + archived


class TestStabilityModel:
    """Bit-for-bit with the Lean ``example``s in EdgeStability.lean §3-§4."""

    def test_miss_reprefills_whole_window(self):
        # example : prefillCost false 200 93551 = 93551
        assert prefill_cost(False, 200, 93551) == 93551

    def test_observed_miss_is_29_minutes(self):
        # example : prefillMs 93551 54 = 1732425 ; > 1_500_000 ms (25 min)
        assert prefill_ms(93551, 54) == 1732425
        assert 1_500_000 < prefill_ms(93551, 54)

    def test_bounded_window_caps_the_miss(self):
        # example : prefillMs 32768 54 = 606814 < prefillMs 93551 54
        assert prefill_ms(32768, 54) == 606814
        assert prefill_ms(32768, 54) < prefill_ms(93551, 54)

    def test_warm_turn_pays_only_delta(self):
        # example : prefillMs (prefillCost true 2000 93551) 54 = 37037
        assert prefill_ms(prefill_cost(True, 2000, 93551), 54) == 37037

    def test_per_turn_cost_le_budget(self):
        # theorem per_turn_cost_le_budget: window ≤ B ∧ delta ≤ B ⇒ cost ≤ B
        B, window, delta = 32768, 32000, 2000
        for hit in (True, False):
            assert prefill_cost(hit, delta, window) <= B

    def test_archive_preserves_accessible(self):
        # theorem archive_preserves_accessible: demote d resident→archived
        r, a, d = 93551, 0, 60000
        assert accessible(r - d, a + d) == accessible(r, a)

    def test_delete_loses_information(self):
        # theorem delete_loses_information: no archive ⇒ strict loss
        r, d = 93551, 60000
        assert (r - d) < accessible(r, 0)


# ── profile satisfies the derived requirements ──────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def launch_sh() -> str:
    return LAUNCH_SH.read_text()


def _flag_value(text: str, flag: str) -> str | None:
    """Extract the token following ``flag`` in a launch script's real exec
    lines — comment lines (``#`` / ``rem``) are ignored so prose mentioning a
    flag can't be mistaken for the actual argument."""
    code = "\n".join(
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith(("#", "rem ", "rem\t"))
    )
    m = re.search(rf"{re.escape(flag)}\s+([^\s\\^]+)", code)
    return m.group(1) if m else None


class TestProfileSatisfiesRequirements:

    def test_R1_single_slot_main_server(self, launch_sh):
        """R1 exclusive tenancy: the main-model server runs one slot, so no
        LRU eviction of the main conversation is possible (two_slots_thrash).
        """
        parallel = _flag_value(launch_sh, "--parallel")
        assert parallel == "1", (
            f"--parallel must be 1 for exclusive tenancy, got {parallel!r}. "
            "Multi-slot without a honored cache_key pin thrashes (see "
            "EdgeStability.two_slots_thrash)."
        )
        # both launch scripts must agree (parsed from real exec lines)
        assert _flag_value(LAUNCH_BAT.read_text(), "--parallel") == "1"

    def test_R1_no_aux_llm_on_main_endpoint(self, cfg):
        """R1: the main endpoint must see only the main conversation. The
        compaction path must not issue an aux LLM call to it — guaranteed by
        the mechanical engine (no summary request at all).
        """
        assert cfg["context"]["engine"] == "mechanical"
        # deferrable aux that would otherwise call the model is off
        assert cfg["auxiliary"]["title_generation"]["enabled"] is False

    def test_R2_bounded_window(self, cfg):
        """R2: a working-set budget must be set so the resident window — and
        the worst-case miss — is a constant (per_turn_cost_le_budget)."""
        budget = cfg["compression"].get("budget_tokens")
        assert isinstance(budget, int) and budget > 0, (
            "compression.budget_tokens must be a positive int (R2). "
            "Without it the window rides to the provider limit and every miss "
            "is O(context)."
        )

    def test_R3_archive_enabled(self, cfg):
        """R3: R2 is only sound with the cold tier present, so a bounded
        window relocates rather than deletes (archive_preserves_accessible)."""
        cli = cfg["platform_toolsets"]["cli"]
        assert "session_search" in cli, (
            "session_search (cold tier) must be enabled so budget_tokens "
            "demotes instead of deletes (EdgeStability.delete_loses_information)."
        )

    def test_context_length_matches_single_slot_window(self, cfg, launch_sh):
        """With --parallel 1 the per-slot window equals --ctx-size, which must
        equal model.context_length (else mid-session hard-rejects)."""
        ctx_size = _flag_value(launch_sh, "--ctx-size")
        assert ctx_size is not None
        assert int(ctx_size) == cfg["model"]["context_length"], (
            f"--ctx-size {ctx_size} must equal model.context_length "
            f"{cfg['model']['context_length']} under --parallel 1."
        )
