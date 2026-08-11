"""Long-call strategy: 0-14 DTE ATM-ish directional long call.

This is the canonical "first strategy" for flip-options-bot. It mirrors the
shape of flip-alpaca-bot's long-call but with the structural fixes:

1. Every entry generates a `position_id` UUID. The same (symbol, date) tuple
   CAN have multiple positions — they're distinguished by position_id.

2. The strategy returns a STRUCTURED signal that the risk engine + executor
   can verify. No silent assumptions about order types or fill prices.

3. Conviction is computed before submission, not after — so the funnel
   recorder can show distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class LongCallSignal:
    """What the strategy wants to trade. Filled in by the scanner."""

    symbol: str
    expiry: str  # YYYY-MM-DD
    strike: float
    side: Literal["buy"] = "buy"
    option_type: Literal["call"] = "call"
    qty: int = 1
    limit_price: float = 0.0
    conviction: float = 0.0  # 0..1
    dte: int = 0
    strategy_id: str = "long_call"
    notes: str = ""
    ts: str = ""


@dataclass
class LongCallFilters:
    """Configurable filter set, sourced from Settings.

    Default values match the FOB_LONG_CALL_* env vars (see config.py)."""

    min_dte: int
    target_dte: int
    max_dte: int
    min_direction_move_pct: float
    max_vwap_extension_pct: float
    min_short_momentum_pct: float
    min_conviction: float
    directional_lookback_minutes: int


def make_filters_from_settings(settings) -> LongCallFilters:
    """Build LongCallFilters from the runtime Settings."""
    return LongCallFilters(
        min_dte=settings.min_dte,
        target_dte=settings.target_dte,
        max_dte=settings.max_dte,
        min_direction_move_pct=settings.long_call_min_direction_move_pct,
        max_vwap_extension_pct=settings.long_call_max_vwap_extension_pct,
        min_short_momentum_pct=settings.long_call_min_short_momentum_pct,
        min_conviction=settings.long_call_min_conviction,
        directional_lookback_minutes=settings.long_call_directional_lookback_minutes,
    )


def passes_dte_window(dte: int, filters: LongCallFilters) -> bool:
    """A candidate's DTE must be within the configured window."""
    return filters.min_dte <= dte <= filters.max_dte


def pick_target_expiry(available_expiries: list[str], filters: LongCallFilters) -> str | None:
    """Pick the expiry closest to `target_dte`, constrained by min/max.

    `available_expiries` is a sorted list of YYYY-MM-DD strings. Returns None
    if no expiry falls within the DTE window.
    """
    if not available_expiries:
        return None
    today = datetime.utcnow().date()
    candidates = []
    for exp_str in available_expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if passes_dte_window(dte, filters):
            candidates.append((abs(dte - filters.target_dte), exp_str))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def compute_conviction(
    direction_move: float,
    vwap_extension: float,
    short_momentum: float,
    spread_pct: float,
    filters: LongCallFilters,
) -> float:
    """Compute conviction in [0, 1].

    Components (each scaled 0..1):
    - direction_move: bigger = better. > min_direction_move_pct → 1, else 0.
    - vwap_extension: smaller = better. > max_vwap_extension_pct → 0, else 1.
    - short_momentum: bigger = better. > min_short_momentum_pct → 1, else 0.
    - spread_pct: smaller = better. < 0.05 → 1, else linearly to 0 at 0.20.

    Final conviction = weighted mean. Weighting defaults to equal.
    """
    dir_score = 1.0 if direction_move >= filters.min_direction_move_pct else direction_move / filters.min_direction_move_pct
    vwap_score = 1.0 if vwap_extension <= filters.max_vwap_extension_pct else 0.0
    mom_score = 1.0 if short_momentum >= filters.min_short_momentum_pct else short_momentum / filters.min_short_momentum_pct
    spread_score = max(0.0, 1.0 - max(0.0, (spread_pct - 0.05) / 0.15))

    weights = [0.30, 0.20, 0.30, 0.20]
    return (
        weights[0] * dir_score
        + weights[1] * vwap_score
        + weights[2] * mom_score
        + weights[3] * spread_score
    )


def passes_conviction(conviction: float, filters: LongCallFilters) -> bool:
    return conviction >= filters.min_conviction