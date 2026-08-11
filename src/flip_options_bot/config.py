"""flip-options-bot — configuration loaded from env.

This module is the single source of truth for all knobs. Every other module
imports `Settings` from here. The Settings object is constructed once at
daemon startup and passed by reference; mutable fields require an explicit
`Settings.reload()` (which the journal + journal_correlator call when the
.env file changes on disk).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

DEFAULT_RUN_DIR = Path("/root/flip/projects/flip-options-bot/runs")
DEFAULT_DASHBOARD_PORT = 8100


def _load_env_file() -> dict[str, str]:
    """Read the env file from the current working directory.

    Only loads `.env` from cwd. Process env (set by systemd EnvironmentFile=)
    wins. This makes the test surface predictable — tests use
    `monkeypatch.chdir(tmp_path)` to control which env file is read.
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return {k: v for k, v in dotenv_values(cwd_env).items() if v is not None}
    return {}


def _coerce_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def _coerce_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _coerce_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    """Immutable settings object. Construct via `Settings.from_env()`."""

    phase: Literal["paper", "live"] = "paper"
    live_trade_enabled: bool = False
    equity_start: float = 10_000.0

    alpaca_paper_key: str = ""
    alpaca_paper_secret: str = ""
    alpaca_paper_base: str = "https://paper-api.alpaca.markets"
    alpaca_data_base: str = "https://data.alpaca.markets"
    alpaca_live_key: str = ""
    alpaca_live_secret: str = ""
    alpaca_live_base: str = "https://api.alpaca.markets"

    max_positions: int = 3
    per_trade_risk_pct: float = 2.0
    daily_loss_cap_pct: float = 6.0
    weekly_loss_cap_pct: float = 12.0
    max_contract_dollar: int = 500

    min_dte: int = 1
    target_dte: int = 5
    max_dte: int = 14

    scan_interval_s: int = 60
    limit_fill_window_s: int = 60
    resubmit_cooldown_s: int = 120
    max_quote_age_s: int = 15

    avoid_fomc: bool = True
    avoid_earnings: bool = True
    earnings_otm_min_pct: float = 0.005

    trailing_profit_floor_pct: float = 0.05
    trailing_peak_gain_retention_pct: float = 0.50
    daily_profit_lock_arm_pct: float = 2.0
    daily_profit_lock_retention_pct: float = 0.50

    long_call_enabled: bool = True
    long_call_min_direction_move_pct: float = 0.003
    long_call_max_vwap_extension_pct: float = 0.020
    long_call_min_short_momentum_pct: float = 0.0010
    long_call_min_conviction: float = 0.45
    long_call_directional_lookback_minutes: int = 20

    diagonal_enabled: bool = False  # off until exit-logic revision ships

    entry_hours_et: str = "09:30-15:30"

    close_eod_minutes: int = 5

    dashboard: bool = True
    dashboard_port: int = DEFAULT_DASHBOARD_PORT

    alert_level: str = "normal"
    discord_webhook: str = ""

    run_dir: Path = field(default_factory=lambda: DEFAULT_RUN_DIR)

    position_monitor_interval_s: int = 15

    def is_live(self) -> bool:
        """Live mode is double-gated: phase=live AND live_trade_enabled=true."""
        return self.phase == "live" and self.live_trade_enabled

    def has_paper_creds(self) -> bool:
        return bool(self.alpaca_paper_key and self.alpaca_paper_secret)

    def has_live_creds(self) -> bool:
        return bool(self.alpaca_live_key and self.alpaca_live_secret)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build from disk env file, with process env overriding."""
        file_env = _load_env_file()
        merged = {**file_env, **os.environ}

        run_dir_raw = merged.get("FOB_RUN_DIR", str(DEFAULT_RUN_DIR))
        return cls(
            phase=merged.get("FOB_PHASE", "paper"),  # type: ignore[arg-type]
            live_trade_enabled=_coerce_bool(merged, "LIVETRADE_ENABLED", False),
            equity_start=_coerce_float(merged, "FOB_EQUITY_START", 10_000.0),
            alpaca_paper_key=merged.get("APCA_API_KEY_ID_PAPER", ""),
            alpaca_paper_secret=merged.get("APCA_API_SECRET_KEY_PAPER", ""),
            alpaca_paper_base=merged.get(
                "APCA_API_BASE_URL_PAPER", "https://paper-api.alpaca.markets"
            ),
            alpaca_data_base=merged.get("APCA_DATA_BASE_URL", "https://data.alpaca.markets"),
            alpaca_live_key=merged.get("APCA_API_KEY_ID_LIVE", ""),
            alpaca_live_secret=merged.get("APCA_API_SECRET_KEY_LIVE", ""),
            alpaca_live_base=merged.get("APCA_API_BASE_URL_LIVE", "https://api.alpaca.markets"),
            max_positions=_coerce_int(merged, "FOB_MAX_POSITIONS", 3),
            per_trade_risk_pct=_coerce_float(merged, "FOB_PER_TRADE_RISK_PCT", 2.0),
            daily_loss_cap_pct=_coerce_float(merged, "FOB_DAILY_LOSS_CAP_PCT", 6.0),
            weekly_loss_cap_pct=_coerce_float(merged, "FOB_WEEKLY_LOSS_CAP_PCT", 12.0),
            max_contract_dollar=_coerce_int(merged, "FOB_MAX_CONTRACT_DOLLAR", 500),
            min_dte=_coerce_int(merged, "FOB_MIN_DTE", 1),
            target_dte=_coerce_int(merged, "FOB_TARGET_DTE", 5),
            max_dte=_coerce_int(merged, "FOB_MAX_DTE", 14),
            scan_interval_s=_coerce_int(merged, "FOB_SCAN_INTERVAL_S", 60),
            limit_fill_window_s=_coerce_int(merged, "FOB_LIMIT_FILL_WINDOW_S", 60),
            resubmit_cooldown_s=_coerce_int(merged, "FOB_RESUBMIT_COOLDOWN_S", 120),
            max_quote_age_s=_coerce_int(merged, "FOB_MAX_QUOTE_AGE_S", 15),
            avoid_fomc=_coerce_bool(merged, "FOB_AVOID_FOMC", True),
            avoid_earnings=_coerce_bool(merged, "FOB_AVOID_EARNINGS", True),
            earnings_otm_min_pct=_coerce_float(merged, "FOB_EARNINGS_OTM_MIN_PCT", 0.005),
            trailing_profit_floor_pct=_coerce_float(merged, "FOB_TRAILING_PROFIT_FLOOR_PCT", 0.05),
            trailing_peak_gain_retention_pct=_coerce_float(
                merged, "FOB_TRAILING_PEAK_GAIN_RETENTION_PCT", 0.50
            ),
            daily_profit_lock_arm_pct=_coerce_float(merged, "FOB_DAILY_PROFIT_LOCK_ARM_PCT", 2.0),
            daily_profit_lock_retention_pct=_coerce_float(
                merged, "FOB_DAILY_PROFIT_LOCK_RETENTION_PCT", 0.50
            ),
            long_call_enabled=_coerce_bool(merged, "FOB_LONG_CALL_ENABLED", True),
            long_call_min_direction_move_pct=_coerce_float(
                merged, "FOB_LONG_CALL_MIN_DIRECTION_MOVE_PCT", 0.003
            ),
            long_call_max_vwap_extension_pct=_coerce_float(
                merged, "FOB_LONG_CALL_MAX_VWAP_EXTENSION_PCT", 0.020
            ),
            long_call_min_short_momentum_pct=_coerce_float(
                merged, "FOB_LONG_CALL_MIN_SHORT_MOMENTUM_PCT", 0.0010
            ),
            long_call_min_conviction=_coerce_float(merged, "FOB_LONG_CALL_MIN_CONVICTION", 0.45),
            long_call_directional_lookback_minutes=_coerce_int(
                merged, "FOB_LONG_CALL_DIRECTIONAL_LOOKBACK_MINUTES", 20
            ),
            diagonal_enabled=_coerce_bool(merged, "FOB_DIAGONAL_ENABLED", False),
            entry_hours_et=merged.get("FOB_ENTRY_HOURS_ET", "09:30-15:30"),
            close_eod_minutes=_coerce_int(merged, "FOB_CLOSE_EOD_MINUTES", 5),
            dashboard=_coerce_bool(merged, "FOB_DASHBOARD", True),
            dashboard_port=_coerce_int(merged, "FOB_DASHBOARD_PORT", DEFAULT_DASHBOARD_PORT),
            alert_level=merged.get("FOB_ALERT_LEVEL", "normal"),
            discord_webhook=merged.get("FOB_DISCORD_WEBHOOK", ""),
            run_dir=Path(run_dir_raw),
            position_monitor_interval_s=_coerce_int(
                merged, "FOB_POSITION_MONITOR_INTERVAL_S", 15
            ),
        )


def get_settings() -> Settings:
    """Module-level singleton. Re-importing does not reload."""
    return Settings.from_env()
