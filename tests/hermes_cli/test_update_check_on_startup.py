"""updates.check_on_startup — startup update-check off switch (edge fork)."""

from unittest.mock import patch

from hermes_cli import main as main_mod


def _cfg(value):
    return {"updates": {"check_on_startup": value}}


def _should_prefetch(cfg):
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch.object(main_mod, "_is_termux_startup_environment", return_value=False),
    ):
        return main_mod._termux_should_prefetch_update_check()


def test_default_checks_on_startup():
    assert _should_prefetch({}) is True


def test_false_disables_startup_check():
    assert _should_prefetch(_cfg(False)) is False


def test_string_off_disables_startup_check():
    assert _should_prefetch(_cfg("off")) is False


def test_explicit_true_keeps_check():
    assert _should_prefetch(_cfg(True)) is True
