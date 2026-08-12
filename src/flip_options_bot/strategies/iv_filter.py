"""IV-rank and IV-percentile filter.

IV-rank = (current IV - 52w low IV) / (52w high IV - 52w low IV)
IV-percentile = % of days in past 52w where IV was below current.

Cheap options (IV-rank < 25%) → not enough premium to make directional
plays worth it. Skip.
Expensive options (IV-rank > 75%) → IV crush risk post-event, but
ALSO directional premium sellers want this. We're premium BUYERS
(long_call), so IV-rank > 75% is OK (we benefit from moves).

For long_call (premium buyer):
  - Skip if IV-rank < 25% (cheap premium = low edge)
  - Trade if IV-rank in [25%, 100%] (normal or rich regime)

Without real historical IV data on Alpaca paper, we approximate IV-rank
using ATM straddle width as a proxy:
  straddle_width_pct = (call_ask + put_ask - call_bid - put_bid) / spot

If straddle is < 0.5% of spot → very cheap premium → skip
If straddle is in [0.5%, 5%] → normal regime
If straddle is > 5% → very rich (good for buying premium on directional moves)
"""

from __future__ import annotations


def straddle_iv_proxy(
    call_bid: float, call_ask: float, put_bid: float, put_ask: float, spot: float
) -> float:
    """Approximate ATM straddle width as a percentage of spot.

    Returns straddle_width_pct = (call_mid + put_mid) / spot.

    For ATM options, the straddle price relative to spot is a rough
    proxy for implied volatility (higher IV = more premium = wider
    straddle as % of spot). ATM strike ≈ spot.

    Example: spot=$100, call_mid=$1.50, put_mid=$1.45
      straddle = $2.95 = 2.95% of spot
    """
    if spot <= 0:
        return 0.0
    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2
    straddle_mid = call_mid + put_mid
    return straddle_mid / spot


def iv_regime_ok(straddle_iv: float) -> bool:
    """Whether the IV regime is OK for long_call (premium buyer).

    Skip if straddle IV < 0.5% (very cheap premium = low edge).
    OK if straddle IV in [0.5%, 15%] (normal or rich).
    Skip if straddle IV > 15% (something weird happening).
    """
    if straddle_iv <= 0:
        return False  # no data → can't make a call
    return 0.005 <= straddle_iv <= 0.15


def iv_regime_boost(straddle_iv: float) -> float:
    """Conviction boost based on IV regime.

    For premium BUYERS (long_call), rich IV is a tailwind because:
      - More dollar premium per contract
      - Larger intraday moves possible
      - Spreads are tighter (more market makers competing)

    Returns a multiplier in [0.85, 1.20]:
      - straddle_iv < 1%: multiplier 0.85 (cheap premium = low edge)
      - straddle_iv in [1%, 3%]: 1.0 (normal)
      - straddle_iv > 3%: up to 1.20 (rich premium)
    """
    if straddle_iv <= 0:
        return 1.0
    if straddle_iv < 0.01:
        return 0.85
    if straddle_iv > 0.03:
        # Linear boost from 1.0 to 1.20 as IV goes from 3% to 10%
        return min(1.20, 1.0 + (straddle_iv - 0.03) / 0.07 * 0.20)
    return 1.0