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
    strategy_id: str = "bull_put_credit_spread"
    notes: str = ""
    ts: str = ""


def select_strikes(
    spot: float,
    short_delta_target: float = 0.30,
    long_delta_target: float = 0.15,
) -> tuple[float, float] | None:
    """Pick short and long put strikes by delta.

    short_delta_target: 0.30 → 30 delta → ~70% POP (probability OTM)
    long_delta_target: 0.15 → defines the spread width

    Returns (short_strike, long_strike) or None if strikes can't be
    selected (need options chain).

    For the FIRST CUT we use a rough mapping: delta 0.30 ≈ 5% OTM,
    delta 0.15 ≈ 10% OTM. Real delta lookup requires the greeks
    endpoint.
    """
    if spot <= 0:
        return None
    short_strike = round(spot * 0.95, 0)  # ~5% OTM (~delta 0.30)
    long_strike = round(spot * 0.90, 0)   # ~10% OTM (~delta 0.15)
    if long_strike >= short_strike:
        return None
    # Spread width must be reasonable ($1 to $20 typical)
    width = short_strike - long_strike
    if width < 1.0 or width > 20.0:
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
) -> float:
    """Compute conviction for a bull put credit spread.

    Higher when:
    - Direction is bullish (or neutral)
    - IV is rich (premium sellers want this)
    - Credit > 1/3 of spread width (good risk/reward)
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