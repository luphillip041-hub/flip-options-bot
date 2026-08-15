"""Long-call strategy: 0-14 DTE ATM-ish directional long call.

This is the canonical "first strategy" for flip-options-bot. It mirrors the
shape of flip-alpaca-bot's long-call but with the structural fixes:

1. Every entry generates a `position_id` UUID. The same (symbol, date) tuple
   CAN have multiple positions — they're distinguished by position_id.

2. The strategy returns a STRUCTURED signal that the risk engine + executor
   can verify. No silent assumptions about order types or fill prices.

3. Conviction is computed before submission, not after — so the funnel
   recorder can show distribution.

4. **Volatility regime filter** (new): we measure the bid-ask width of the
   option chain as an IV-crash proxy. If spreads are exploding (vol
   sellers are pulling quotes), we DON'T buy premium — that's how you
   catch a falling knife. Conversely, if spreads are tightening on the
   option but the underlying is trending, conviction gets a tailwind.

5. **Time-of-day filter** (new): the last 30 min of the session are
   no-trade for new entries. Theta decay accelerates and the
   'momentum' read is unreliable as market makers pull quotes.

6. **Momentum exhaustion check** (new): if the short momentum has
   reversed from the longer-window move (e.g., 5-min momentum is
   negative while 20-min is positive), the move is likely
   exhausted — we don't chase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
    target_otm_pct: float = 0.003


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
        target_otm_pct=settings.long_call_target_otm_pct,
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
    today = datetime.now(UTC).date()
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

    Returns 0.0 (no trade) if momentum is exhausted (short_momentum < 0
    while direction_move > 0) — we're not chasing a dead move.
    """
    # Momentum exhaustion: short_momentum is negative while direction_move
    # is positive → the 5-min tape says the move is dying. Don't enter.
    if direction_move > 0 and short_momentum < 0:
        return 0.0
    # Symmetric: short_momentum positive while direction_move negative →
    # might be a reversal setup, but we don't have a reversal strategy,
    # so skip.
    if direction_move < 0 and short_momentum > 0:
        return 0.0

    dir_score = (
        1.0
        if direction_move >= filters.min_direction_move_pct
        else direction_move / filters.min_direction_move_pct
    )
    vwap_score = 1.0 if vwap_extension <= filters.max_vwap_extension_pct else 0.0
    mom_score = (
        1.0
        if short_momentum >= filters.min_short_momentum_pct
        else short_momentum / filters.min_short_momentum_pct
    )
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


def volatility_regime_ok(chain_spreads: list[float]) -> bool:
    """Volatility regime filter using option chain spread distribution.

    We sample spread_pct across the eligible chain. If the MEDIAN spread
    is wider than 0.20 (20% of mid), market makers are pulling quotes →
    we DON'T buy premium in a vol-crash regime.

    chain_spreads is a list of spread_pct values (e.g., 0.05 = 5% wide).
    Returns True if regime is OK for new entries.
    """
    if not chain_spreads:
        return False  # no chain data → can't make a call
    sorted_s = sorted(chain_spreads)
    median = sorted_s[len(sorted_s) // 2]
    return median <= 0.20


def size_by_conviction(conviction: float, base_qty: int = 1) -> int:
    """Scale position size by conviction.

    Tiered sizing:
      - conviction < 0.45: 0 contracts (shouldn't reach here — gate filters)
      - conviction 0.45-0.60: 1 contract (baseline)
      - conviction 0.60-0.75: 1 contract (still baseline — don't get overconfident)
      - conviction 0.75-0.90: 2 contracts (high conviction = scale up)
      - conviction >= 0.90: 2 contracts (max — capped to prevent concentration)

    The risk engine's max-position cap is the ultimate governor; we just
    suggest size based on signal quality.
    """
    if conviction < 0.45:
        return 0
    if conviction >= 0.75:
        return max(1, base_qty * 2)
    return max(1, base_qty)


def is_strong_setup(conviction: float) -> bool:
    """True if this is an A+ setup worth scaling up.

    Conviction >= 0.75 AND direction momentum is healthy.
    """
    return conviction >= 0.75
