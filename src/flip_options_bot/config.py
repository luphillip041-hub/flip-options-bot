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


def _coerce_float_tuple(
    env: dict[str, str], key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


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
    # Throughput control: submit the best N ranked candidates per scan. Keep
    # this separate from max_positions so a bigger book does not become an
    # uncontrolled one-tick order storm.
    max_submissions_per_scan: int = 3

    # Directional options: prefer 0DTE gamma, but allow up to two weeks when
    # same-day chains are too expensive/illiquid or no clean setup exists.
    min_dte: int = 0
    target_dte: int = 0
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

    # Gain-protection monitor knobs
    sl_threshold_pct: float = 0.50       # SL fires at 50% of entry
    tp_multiplier: float = 1.50          # partial TP at +50%
    tp_full_multiplier: float = 2.00     # full TP only if no partial AND +100%
    trailing_arm_pct: float = 0.10       # trailing floor arms after +10%
    trailing_retention: float = 0.50     # 50% of peak (legacy knob)
    profit_floor_pct: float = 1.10       # never give gains back to entry
    min_tp_profit_dollar: float = 25.0   # don't take tiny profit at wash

    long_call_enabled: bool = True
    # 0.10% (10 bps) — current SPY 20-min moves are running -0.05% to
    # -0.13% throughout the day. Anything tighter than 0.10% yields
    # zero setups on quiet days. The conviction filter (0.45) is the
    # real gate; this just sets the floor for entry consideration.
    long_call_min_direction_move_pct: float = 0.0010
    long_call_max_vwap_extension_pct: float = 0.020
    long_call_min_short_momentum_pct: float = 0.0010
    long_call_min_conviction: float = 0.45
    long_call_directional_lookback_minutes: int = 20
    # OTM calls are the high-convexity 0DTE vehicle. 0.30% OTM keeps the
    # contract close enough to fill/move while still giving gamma upside.
    long_call_target_otm_pct: float = 0.003

    # ===== Long Put bearish exposure =====
    # Buy puts only on confirmed downtrends. Same option risk/monitor path as
    # long_call: debit capped, limit orders only, soft TP/SL monitor after fill.
    long_put_enabled: bool = False
    long_put_min_direction_move_pct: float = 0.0010
    long_put_max_vwap_extension_pct: float = 0.020
    long_put_min_short_momentum_pct: float = 0.0010
    long_put_min_conviction: float = 0.45
    long_put_directional_lookback_minutes: int = 20
    # OTM puts are the bearish high-convexity vehicle. For puts, OTM means
    # below spot by this fraction.
    long_put_target_otm_pct: float = 0.003

    # ===== High-risk / high-reward long-premium mode =====
    # When enabled, directional scans prefer farther OTM contracts from this
    # ladder while still requiring real bid/ask liquidity and contract caps.
    long_option_high_reward_mode: bool = False
    long_option_otm_ladder_pct: tuple[float, ...] = (0.003, 0.006, 0.010, 0.015)
    long_option_min_premium: float = 0.15
    long_option_max_spread_pct: float = 0.35
    long_option_convexity_weight: float = 0.15
    # If one underlying realizes this much directional loss in the current ET
    # session, stop opening new long calls/puts on that underlying for the day.
    directional_underlying_loss_lockout_dollar: float = 50.0

    # ===== Free yfinance 1DTE+ confirmation sidecar =====
    # Yahoo/yfinance is not broker truth and is not historical OPRA. Use it only
    # to enrich/rank 1DTE+ paper candidates with current/delayed chain fields.
    yfinance_confirm_1dte_enabled: bool = False
    yfinance_confirm_min_dte: int = 1
    yfinance_strict_gate: bool = False
    yfinance_min_volume: int = 50
    yfinance_max_spread_pct: float = 0.35
    yfinance_bidask_bonus: float = 0.03
    yfinance_volume_bonus: float = 0.01
    # Last-price/volume-only Yahoo rows are non-executable. Require the row's
    # last trade date to match today's ET session before using volume as even a
    # tiny ranking nudge; stale rows still get logged in notes.
    yfinance_require_current_trade_date_for_volume_bonus: bool = True

    # ===== Long Equity fallback =====
    # Bullish long exposure when call premiums are too expensive or chains are
    # too wide. Limit orders only; same paper/live gates as options.
    long_equity_enabled: bool = False
    long_equity_min_direction_move_pct: float = 0.0010
    long_equity_max_vwap_extension_pct: float = 0.020
    long_equity_min_short_momentum_pct: float = 0.0010
    long_equity_min_conviction: float = 0.55
    long_equity_directional_lookback_minutes: int = 20
    long_equity_max_position_dollar: float = 500.0
    long_equity_stop_loss_pct: float = 0.004
    long_equity_take_profit_pct: float = 0.008

    diagonal_enabled: bool = False  # off until exit-logic revision ships

    # ===== Bull Put Credit Spread (BPCS) =====
    # Off by default — needs separate risk caps + per-trade max-loss
    # enforcement (defined-risk spread, max_loss = width*100 - credit).
    bpcs_enabled: bool = False
    # Higher target_dte than long_call: theta decay advantage
    # (30-45 DTE is Tastytrade canonical for credit spreads).
    bpcs_target_dte: int = 35
    bpcs_min_dte: int = 25
    bpcs_max_dte: int = 50
    # Strike selection: short strike delta target
    bpcs_short_delta: float = 0.30
    # Spread width in dollars. SPY spreads are typically $5-$15 wide.
    # Allow up to $50 to handle SPY/QQQ/IBM/etc.
    bpcs_min_width: float = 2.0
    bpcs_max_width: float = 50.0
    # Minimum credit as fraction of width (don't sell <20% premium).
    # 20% is a sensible floor — anything lower means you're selling
    # too little premium relative to your max loss exposure.
    bpcs_min_credit_pct_of_width: float = 0.20
    # Per-trade MAX LOSS cap (as fraction of equity) — this is the
    # REAL risk bound for a credit spread (not the credit received).
    # 5% of equity is the canonical tastytrade cap; combined with the
    # absolute cap below, the bot uses the SMALLER of the two.
    bpcs_max_loss_pct: float = 5.0
    # Absolute max loss per spread in dollars (safety floor for small
    # equity accounts where 5% of equity is small)
    bpcs_max_loss_dollar: float = 600.0
    # Profit target: take profit at 50% of credit received
    bpcs_profit_target_pct: float = 0.50
    # Closing days before expiry: close at 21 DTE to avoid gamma risk
    bpcs_close_at_dte: int = 21

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
    def from_env(cls) -> Settings:
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
            max_submissions_per_scan=_coerce_int(merged, "FOB_MAX_SUBMISSIONS_PER_SCAN", 3),
            min_dte=_coerce_int(merged, "FOB_MIN_DTE", 0),
            target_dte=_coerce_int(merged, "FOB_TARGET_DTE", 0),
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
            sl_threshold_pct=_coerce_float(merged, "FOB_SL_THRESHOLD_PCT", 0.50),
            tp_multiplier=_coerce_float(merged, "FOB_TP_MULTIPLIER", 1.50),
            tp_full_multiplier=_coerce_float(merged, "FOB_TP_FULL_MULTIPLIER", 2.00),
            trailing_arm_pct=_coerce_float(merged, "FOB_TRAILING_ARM_PCT", 0.10),
            trailing_retention=_coerce_float(merged, "FOB_TRAILING_RETENTION", 0.50),
            profit_floor_pct=_coerce_float(merged, "FOB_PROFIT_FLOOR_PCT", 1.10),
            min_tp_profit_dollar=_coerce_float(merged, "FOB_MIN_TP_PROFIT_DOLLAR", 25.0),
            long_call_enabled=_coerce_bool(merged, "FOB_LONG_CALL_ENABLED", True),
            long_call_min_direction_move_pct=_coerce_float(
                merged, "FOB_LONG_CALL_MIN_DIRECTION_MOVE_PCT", 0.0010
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
            long_call_target_otm_pct=_coerce_float(
                merged, "FOB_LONG_CALL_TARGET_OTM_PCT", 0.003
            ),
            long_put_enabled=_coerce_bool(merged, "FOB_LONG_PUT_ENABLED", False),
            long_put_min_direction_move_pct=_coerce_float(
                merged, "FOB_LONG_PUT_MIN_DIRECTION_MOVE_PCT", 0.0010
            ),
            long_put_max_vwap_extension_pct=_coerce_float(
                merged, "FOB_LONG_PUT_MAX_VWAP_EXTENSION_PCT", 0.020
            ),
            long_put_min_short_momentum_pct=_coerce_float(
                merged, "FOB_LONG_PUT_MIN_SHORT_MOMENTUM_PCT", 0.0010
            ),
            long_put_min_conviction=_coerce_float(merged, "FOB_LONG_PUT_MIN_CONVICTION", 0.45),
            long_put_directional_lookback_minutes=_coerce_int(
                merged, "FOB_LONG_PUT_DIRECTIONAL_LOOKBACK_MINUTES", 20
            ),
            long_put_target_otm_pct=_coerce_float(
                merged, "FOB_LONG_PUT_TARGET_OTM_PCT", 0.003
            ),
            long_option_high_reward_mode=_coerce_bool(
                merged, "FOB_LONG_OPTION_HIGH_REWARD_MODE", False
            ),
            long_option_otm_ladder_pct=_coerce_float_tuple(
                merged, "FOB_LONG_OPTION_OTM_LADDER_PCT", (0.003, 0.006, 0.010, 0.015)
            ),
            long_option_min_premium=_coerce_float(
                merged, "FOB_LONG_OPTION_MIN_PREMIUM", 0.15
            ),
            long_option_max_spread_pct=_coerce_float(
                merged, "FOB_LONG_OPTION_MAX_SPREAD_PCT", 0.35
            ),
            long_option_convexity_weight=_coerce_float(
                merged, "FOB_LONG_OPTION_CONVEXITY_WEIGHT", 0.15
            ),
            directional_underlying_loss_lockout_dollar=_coerce_float(
                merged, "FOB_DIRECTIONAL_UNDERLYING_LOSS_LOCKOUT_DOLLAR", 50.0
            ),
            yfinance_confirm_1dte_enabled=_coerce_bool(
                merged, "FOB_YFINANCE_CONFIRM_1DTE_ENABLED", False
            ),
            yfinance_confirm_min_dte=_coerce_int(merged, "FOB_YFINANCE_CONFIRM_MIN_DTE", 1),
            yfinance_strict_gate=_coerce_bool(merged, "FOB_YFINANCE_STRICT_GATE", False),
            yfinance_min_volume=_coerce_int(merged, "FOB_YFINANCE_MIN_VOLUME", 50),
            yfinance_max_spread_pct=_coerce_float(merged, "FOB_YFINANCE_MAX_SPREAD_PCT", 0.35),
            yfinance_bidask_bonus=_coerce_float(merged, "FOB_YFINANCE_BIDASK_BONUS", 0.03),
            yfinance_volume_bonus=_coerce_float(merged, "FOB_YFINANCE_VOLUME_BONUS", 0.01),
            yfinance_require_current_trade_date_for_volume_bonus=_coerce_bool(
                merged, "FOB_YFINANCE_REQUIRE_CURRENT_TRADE_DATE_FOR_VOLUME_BONUS", True
            ),
            long_equity_enabled=_coerce_bool(merged, "FOB_LONG_EQUITY_ENABLED", False),
            long_equity_min_direction_move_pct=_coerce_float(
                merged, "FOB_LONG_EQUITY_MIN_DIRECTION_MOVE_PCT", 0.0010
            ),
            long_equity_max_vwap_extension_pct=_coerce_float(
                merged, "FOB_LONG_EQUITY_MAX_VWAP_EXTENSION_PCT", 0.020
            ),
            long_equity_min_short_momentum_pct=_coerce_float(
                merged, "FOB_LONG_EQUITY_MIN_SHORT_MOMENTUM_PCT", 0.0010
            ),
            long_equity_min_conviction=_coerce_float(merged, "FOB_LONG_EQUITY_MIN_CONVICTION", 0.55),
            long_equity_directional_lookback_minutes=_coerce_int(
                merged, "FOB_LONG_EQUITY_DIRECTIONAL_LOOKBACK_MINUTES", 20
            ),
            long_equity_max_position_dollar=_coerce_float(
                merged, "FOB_LONG_EQUITY_MAX_POSITION_DOLLAR", 500.0
            ),
            long_equity_stop_loss_pct=_coerce_float(merged, "FOB_LONG_EQUITY_STOP_LOSS_PCT", 0.004),
            long_equity_take_profit_pct=_coerce_float(merged, "FOB_LONG_EQUITY_TAKE_PROFIT_PCT", 0.008),
            diagonal_enabled=_coerce_bool(merged, "FOB_DIAGONAL_ENABLED", False),
            bpcs_enabled=_coerce_bool(merged, "FOB_BPCS_ENABLED", False),
            bpcs_target_dte=_coerce_int(merged, "FOB_BPCS_TARGET_DTE", 35),
            bpcs_min_dte=_coerce_int(merged, "FOB_BPCS_MIN_DTE", 25),
            bpcs_max_dte=_coerce_int(merged, "FOB_BPCS_MAX_DTE", 50),
            bpcs_short_delta=_coerce_float(merged, "FOB_BPCS_SHORT_DELTA", 0.30),
            bpcs_min_width=_coerce_float(merged, "FOB_BPCS_MIN_WIDTH", 2.0),
            bpcs_max_width=_coerce_float(merged, "FOB_BPCS_MAX_WIDTH", 50.0),
            bpcs_min_credit_pct_of_width=_coerce_float(
                merged, "FOB_BPCS_MIN_CREDIT_PCT_OF_WIDTH", 0.20
            ),
            bpcs_max_loss_pct=_coerce_float(merged, "FOB_BPCS_MAX_LOSS_PCT", 5.0),
            bpcs_max_loss_dollar=_coerce_float(merged, "FOB_BPCS_MAX_LOSS_DOLLAR", 600.0),
            bpcs_profit_target_pct=_coerce_float(merged, "FOB_BPCS_PROFIT_TARGET_PCT", 0.50),
            bpcs_close_at_dte=_coerce_int(merged, "FOB_BPCS_CLOSE_AT_DTE", 21),
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
