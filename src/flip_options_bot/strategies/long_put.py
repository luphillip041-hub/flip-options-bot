"""Long-put strategy: bearish directional put buys with limit orders.

This mirrors long_call on the bearish side: buy puts only when the tape is
actually down, short momentum confirms the move, VWAP extension is not
stretched, and option spreads are tight enough to protect gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


@dataclass
class LongPutSignal:
    """What the strategy wants to trade. Filled in by the scanner."""

    symbol: str
    expiry: str
    strike: float
    side: Literal["buy"] = "buy"
    option_type: Literal["put"] = "put"
    qty: int = 1
    limit_price: float = 0.0
    conviction: float = 0.0
    dte: int = 0
    strategy_id: str = "long_put"
    notes: str = ""
    ts: str = ""


@dataclass
class LongPutFilters:
    min_dte: int
    target_dte: int
    max_dte: int
    min_direction_move_pct: float
    max_vwap_extension_pct: float
    min_short_momentum_pct: float
    min_conviction: float
    directional_lookback_minutes: int
    target_otm_pct: float = 0.003


def make_filters_from_settings(settings) -> LongPutFilters:
    return LongPutFilters(
        min_dte=settings.min_dte,
        target_dte=settings.target_dte,
        max_dte=settings.max_dte,
        min_direction_move_pct=settings.long_put_min_direction_move_pct,
        max_vwap_extension_pct=settings.long_put_max_vwap_extension_pct,
        min_short_momentum_pct=settings.long_put_min_short_momentum_pct,
        min_conviction=settings.long_put_min_conviction,
        directional_lookback_minutes=settings.long_put_directional_lookback_minutes,
        target_otm_pct=settings.long_put_target_otm_pct,
    )


def passes_dte_window(dte: int, filters: LongPutFilters) -> bool:
    return filters.min_dte <= dte <= filters.max_dte


def pick_target_expiry(available_expiries: list[str], filters: LongPutFilters) -> str | None:
    if not available_expiries:
        return None
    today = datetime.now(timezone.utc).date()
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
    filters: LongPutFilters,
) -> float:
    """Compute bearish long-put conviction in [0, 1]."""
    # Puts require confirmed bearish tape. Don't buy puts on a bounce or a
    # mixed tape; that is how premium gets chopped up.
    if direction_move >= 0 or short_momentum >= 0:
        return 0.0

    down_move = abs(direction_move)
    down_momentum = abs(short_momentum)
    dir_score = 1.0 if down_move >= filters.min_direction_move_pct else down_move / filters.min_direction_move_pct
    vwap_score = 1.0 if vwap_extension <= filters.max_vwap_extension_pct else 0.0
    mom_score = 1.0 if down_momentum >= filters.min_short_momentum_pct else down_momentum / filters.min_short_momentum_pct
    spread_score = max(0.0, 1.0 - max(0.0, (spread_pct - 0.05) / 0.15))

    weights = [0.30, 0.20, 0.30, 0.20]
    return (
        weights[0] * dir_score
        + weights[1] * vwap_score
        + weights[2] * mom_score
        + weights[3] * spread_score
    )


def passes_conviction(conviction: float, filters: LongPutFilters) -> bool:
    return conviction >= filters.min_conviction
