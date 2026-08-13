"""Long-equity fallback strategy for bullish tape when calls are too expensive.

This is deliberately simple and paper-first: it reuses the long-call tape
features (direction move, VWAP extension, short momentum), but buys shares
instead of options when option premiums blow through the per-contract dollar
cap. That lets the bot express bullish exposure without forcing bad option
fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LongEquitySignal:
    symbol: str
    side: Literal["buy"] = "buy"
    qty: int = 0
    limit_price: float = 0.0
    conviction: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    strategy_id: str = "long_equity"
    notes: str = ""
    ts: str = ""


@dataclass
class LongEquityFilters:
    min_direction_move_pct: float
    max_vwap_extension_pct: float
    min_short_momentum_pct: float
    min_conviction: float
    directional_lookback_minutes: int
    max_position_dollar: float
    stop_loss_pct: float
    take_profit_pct: float


def make_filters_from_settings(settings) -> LongEquityFilters:
    return LongEquityFilters(
        min_direction_move_pct=settings.long_equity_min_direction_move_pct,
        max_vwap_extension_pct=settings.long_equity_max_vwap_extension_pct,
        min_short_momentum_pct=settings.long_equity_min_short_momentum_pct,
        min_conviction=settings.long_equity_min_conviction,
        directional_lookback_minutes=settings.long_equity_directional_lookback_minutes,
        max_position_dollar=settings.long_equity_max_position_dollar,
        stop_loss_pct=settings.long_equity_stop_loss_pct,
        take_profit_pct=settings.long_equity_take_profit_pct,
    )


def compute_conviction(
    direction_move: float,
    vwap_extension: float,
    short_momentum: float,
    filters: LongEquityFilters,
) -> float:
    """Compute bullish-share conviction in [0, 1]."""
    if direction_move <= 0 or short_momentum <= 0:
        return 0.0
    if vwap_extension > filters.max_vwap_extension_pct:
        return 0.0

    dir_score = min(1.0, direction_move / max(filters.min_direction_move_pct, 0.0001))
    mom_score = min(1.0, short_momentum / max(filters.min_short_momentum_pct, 0.0001))
    vwap_score = 1.0 - min(1.0, vwap_extension / max(filters.max_vwap_extension_pct, 0.0001))
    return 0.45 * dir_score + 0.40 * mom_score + 0.15 * vwap_score


def passes_conviction(conviction: float, filters: LongEquityFilters) -> bool:
    return conviction >= filters.min_conviction
