"""Tests for Settings parsing from env file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flip_options_bot.config import Settings


def test_settings_defaults_when_no_env(tmp_path: Path, monkeypatch):
    for k in (
        "FOB_PHASE",
        "LIVETRADE_ENABLED",
        "FOB_MAX_POSITIONS",
        "FOB_PER_TRADE_RISK_PCT",
        "FOB_LONG_OPTION_HIGH_REWARD_MODE",
        "FOB_LONG_OPTION_OTM_LADDER_PCT",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    # No env file in tmp_path either
    settings = Settings.from_env()
    assert settings.phase == "paper"
    assert settings.live_trade_enabled is False
    assert settings.max_positions == 3
    assert settings.per_trade_risk_pct == 2.0
    assert settings.daily_loss_cap_pct == 6.0


def test_settings_overrides_via_env(monkeypatch):
    monkeypatch.setenv("FOB_PHASE", "live")
    monkeypatch.setenv("LIVETRADE_ENABLED", "true")
    monkeypatch.setenv("FOB_MAX_POSITIONS", "5")
    monkeypatch.setenv("FOB_PER_TRADE_RISK_PCT", "3.5")
    settings = Settings.from_env()
    assert settings.phase == "live"
    assert settings.live_trade_enabled is True
    assert settings.max_positions == 5
    assert settings.per_trade_risk_pct == 3.5


def test_is_live_requires_double_gate(monkeypatch):
    monkeypatch.setenv("FOB_PHASE", "live")
    monkeypatch.setenv("LIVETRADE_ENABLED", "false")
    settings = Settings.from_env()
    assert settings.is_live() is False

    monkeypatch.setenv("FOB_PHASE", "paper")
    monkeypatch.setenv("LIVETRADE_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.is_live() is False

    monkeypatch.setenv("FOB_PHASE", "live")
    monkeypatch.setenv("LIVETRADE_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.is_live() is True


def test_settings_immutable():
    """Settings is a frozen dataclass — cannot mutate at runtime."""
    settings = Settings.from_env()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        settings.phase = "live"  # type: ignore[misc]


def test_settings_loads_from_dotenv_file(tmp_path: Path, monkeypatch):
    # Clear inherited process env vars so the test isn't polluted by the shell.
    for k in ("APCA_API_KEY_ID_PAPER", "APCA_API_SECRET_KEY_PAPER", "FOB_PHASE", "LIVETRADE_ENABLED", "FOB_MAX_POSITIONS"):
        monkeypatch.delenv(k, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "FOB_PHASE=paper\n"
        "LIVETRADE_ENABLED=false\n"
        "FOB_MAX_POSITIONS=7\n"
        "APCA_API_KEY_ID_PAPER=PK_test\n"
        "APCA_API_SECRET_KEY_PAPER=secret_test\n"
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    assert settings.max_positions == 7
    assert settings.alpaca_paper_key == "PK_test"
    assert settings.alpaca_paper_secret == "secret_test"


def test_high_reward_directional_env(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOB_LONG_OPTION_HIGH_REWARD_MODE", "true")
    monkeypatch.setenv("FOB_LONG_OPTION_OTM_LADDER_PCT", "0.003,0.007,0.015")
    monkeypatch.setenv("FOB_LONG_OPTION_MIN_PREMIUM", "0.20")
    monkeypatch.setenv("FOB_LONG_OPTION_MAX_SPREAD_PCT", "0.30")
    monkeypatch.setenv("FOB_LONG_OPTION_CONVEXITY_WEIGHT", "0.25")

    settings = Settings.from_env()

    assert settings.long_option_high_reward_mode is True
    assert settings.long_option_otm_ladder_pct == (0.003, 0.007, 0.015)
    assert settings.long_option_min_premium == 0.20
    assert settings.long_option_max_spread_pct == 0.30
    assert settings.long_option_convexity_weight == 0.25


def test_has_paper_creds(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)  # ensure no .env file is loaded
    monkeypatch.delenv("APCA_API_KEY_ID_PAPER", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY_PAPER", raising=False)
    settings = Settings.from_env()
    assert settings.has_paper_creds() is False

    monkeypatch.setenv("APCA_API_KEY_ID_PAPER", "PK_test")
    monkeypatch.setenv("APCA_API_SECRET_KEY_PAPER", "secret_test")
    settings = Settings.from_env()
    assert settings.has_paper_creds() is True