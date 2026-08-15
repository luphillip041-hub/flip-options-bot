"""Bull put credit spread (BPCS) strategy.

Defined-risk alternative to naked calls. We SELL a put spread (short
strike K1, long strike K2 where K1 > K2) and collect premium. Our
max loss is (K1 - K2) * 100 - credit received. Our max gain is the
credit received.

When to enter:
- Underlying is in an uptrend (or neutral-to-up)
- IV is rich (premium sellers want rich IV)
- 30-45 DTE for theta decay advantage (Tastytrade canonical)
- Strike selection: short strike ~10-15% OTM, long strike 5-10 pts lower

Edge vs naked long_call:
- Defined risk: we know max loss upfront
- Positive theta: we make money every day (decay)
- Higher win rate (60-70% historically)
- Lower absolute return per trade (bounded by premium)

This module computes:
- strike selection (K1 short, K2 long)
- credit received (estimate)
- max loss / max gain
- conviction score

For the first cut, we DISABLE BPCS by default in Settings (it's a
premium-selling strategy and requires different risk caps).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import Settings


@dataclass
class BPCSFilters:
    """Filter set for BPCS, sourced from Settings."""
    target_dte: int
    min_dte: int
    max_dte: int
    min_width: float
    max_width: float
    min_credit_pct_of_width: float
    min_conviction: float = 0.45


def make_filters_from_settings(settings: Settings) -> BPCSFilters:
    """Build BPCSFilters from runtime Settings."""
    return BPCSFilters(
        target_dte=settings.bpcs_target_dte,
        min_dte=settings.bpcs_min_dte,
        max_dte=settings.bpcs_max_dte,
        min_width=settings.bpcs_min_width,
        max_width=settings.bpcs_max_width,
        min_credit_pct_of_width=settings.bpcs_min_credit_pct_of_width,
    )


def passes_dte_window(dte: int, filters: BPCSFilters) -> bool:
    """A candidate's DTE must be within the configured window."""
    return filters.min_dte <= dte <= filters.max_dte


def pick_target_expiry(available_expiries: list[str], filters: BPCSFilters) -> str | None:
    """Pick the expiry closest to `target_dte`, constrained by min/max.

    For BPCS we usually want the 30-45 DTE bucket (theta decay sweet spot).
    """
    if not available_expiries:
        return None
    candidates = []
    for exp_str in available_expiries:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        # Compute DTE vs today (UTC). The strategy caller passes a
        # caller-provided `now` via the symbol-level wrapper; we just
        # approximate here based on UTC date. The scanner overrides this.
        today = datetime.utcnow().date()
        dte = (exp_date - today).days
        if passes_dte_window(dte, filters):
            candidates.append((abs(dte - filters.target_dte), exp_str))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def passes_bpcs_conviction(conviction: float, filters: BPCSFilters) -> bool:
    return conviction >= filters.min_conviction


@dataclass
class BullPutSpreadSignal:
    short_strike: float
    long_strike: float
    expiry: str  # YYYY-MM-DD
    credit_estimate: float  # per share (so *100 = per contract)
    max_loss_per_contract: float
    max_gain_per_contract: float
    pop: float  # probability of profit (0..1)
    conviction: float  # 0..1
    # Limit prices for both legs (per share). The short put is sold at
    # this price; the long put is bought at this price.
    short_strike_price_estimate: float = 0.0
    long_strike_price_estimate: float = 0.0
    # OCC symbols for both legs (filled by scanner)
    short_put_symbol: str = ""
    long_put_symbol: str = ""
    strategy_id: str = "bull_put_credit_spread"
    notes: str = ""
    ts: str = ""


def select_strikes(
    spot: float,
    short_delta_target: float = 0.30,
    long_delta_target: float = 0.15,
    available_strikes: list[float] | None = None,
) -> tuple[float, float] | None:
    """Pick short and long put strikes by delta.

    short_delta_target: 0.30 → 30 delta → ~70% POP (probability OTM)
    long_delta_target: 0.15 → defines the spread width

    Returns (short_strike, long_strike) or None if strikes can't be
    selected (need options chain).

    Strike selection is a HEURISTIC: we use rough %-OTM mappings
    because we don't have access to a greeks endpoint on Alpaca paper.
      - Short put target: 2% OTM (~delta 0.30 for SPY/QQQ)
      - Long put target:  4% OTM (~delta 0.15)

    If `available_strikes` is provided, we snap to the closest available
    strike ≤ target. This handles Alpaca paper's incomplete chain
    coverage (e.g., SPY chain stops at $639 when spot is $772).
    """
    if spot <= 0:
        return None
    short_target = spot * 0.98  # ~2% OTM
    long_target = spot * 0.96   # ~4% OTM

    if available_strikes:
        # Snap to closest available strike ≤ target
        short_candidates = [s for s in available_strikes if s <= short_target]
        long_candidates = [s for s in available_strikes if s <= long_target]
        if not short_candidates or not long_candidates:
            return None
        short_strike = max(short_candidates)  # closest strike ≤ target
        long_strike = max(long_candidates)    # closest strike ≤ target
        # long_strike must be < short_strike
        if long_strike >= short_strike:
            return None
    else:
        short_strike = round(short_target, 0)
        long_strike = round(long_target, 0)
        if long_strike >= short_strike:
            return None
    width = short_strike - long_strike
    if width < 1.0:
        return None
    return (short_strike, long_strike)


def estimate_credit(
    short_put_bid: float,
    short_put_ask: float,
    long_put_bid: float,
    long_put_ask: float,
) -> float:
    """Estimate the net credit per share.

    Credit = short_put_mid - long_put_mid
    """
    short_mid = (short_put_bid + short_put_ask) / 2
    long_mid = (long_put_bid + long_put_ask) / 2
    credit = short_mid - long_mid
    return max(0.0, credit)


def compute_bpcs_conviction(
    spot: float,
    short_strike: float,
    long_strike: float,
    credit: float,
    iv_rank_proxy: float,
    direction_move_pct: float,  # positive = bullish
    filters: BPCSFilters | None = None,
) -> float:
    """Compute conviction for a bull put credit spread.

    Higher when:
    - Direction is bullish (or neutral)
    - IV is rich (premium sellers want this)
    - Credit > 1/3 of spread width (good risk/reward)

    The `filters` argument is currently unused (kept for parity with
    long_call.compute_conviction) but allows future tuning.
    """
    # Direction score: full credit for bullish, half for neutral
    if direction_move_pct > 0.003:  # +0.3% or more = bullish
        dir_score = 1.0
    elif direction_move_pct > 0:
        dir_score = 0.7
    elif direction_move_pct > -0.003:
        dir_score = 0.5  # neutral, BPCS still works
    else:
        return 0.0  # bearish → don't enter BPCS

    # IV score: BPCS wants rich IV
    if iv_rank_proxy > 0.30:
        iv_score = 1.0
    elif iv_rank_proxy > 0.15:
        iv_score = 0.7
    else:
        iv_score = 0.4  # cheap premium = weak signal

    # Risk/reward score: credit should be > 1/3 of spread width
    spread_width = short_strike - long_strike
    rr_score = min(1.0, (credit / spread_width) * 3) if spread_width > 0 else 0

    weights = [0.40, 0.30, 0.30]
    return (
        weights[0] * dir_score
        + weights[1] * iv_score
        + weights[2] * rr_score
    )
