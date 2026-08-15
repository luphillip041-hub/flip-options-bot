"""Tests for the strategy's gain-protection logic:

- volatility_regime_ok rejects wide-spread chains (vol crash)
- compute_conviction returns 0 when momentum is exhausted
"""

from flip_options_bot.strategies.long_call import (
    LongCallFilters,
    compute_conviction,
    volatility_regime_ok,
)


def _f() -> LongCallFilters:
    return LongCallFilters(
        min_dte=0,
        target_dte=5,
        max_dte=14,
        min_direction_move_pct=0.003,
        max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.0010,
        min_conviction=0.45,
        directional_lookback_minutes=20,
    )


def test_vol_regime_ok_when_spreads_tight():
    assert volatility_regime_ok([0.05, 0.07, 0.04, 0.06]) is True


def test_vol_regime_rejects_wide_spreads():
    # median is 0.25, > 0.20 cap
    assert volatility_regime_ok([0.10, 0.20, 0.25, 0.30, 0.40]) is False


def test_vol_regime_empty_chain_rejects():
    assert volatility_regime_ok([]) is False


def test_compute_conviction_blocks_exhausted_momentum():
    """direction_move positive but short_momentum negative → exhaustion → 0."""
    f = _f()
    c = compute_conviction(
        direction_move=0.01,    # +1% over 20min (good)
        vwap_extension=0.005,
        short_momentum=-0.001,  # -0.1% in last 5min (dying)
        spread_pct=0.05,
        filters=f,
    )
    assert c == 0.0


def test_compute_conviction_blocks_negative_direction_positive_momentum():
    """direction_move negative but short_momentum positive → might be a
    reversal, but we don't have a reversal strategy → 0."""
    f = _f()
    c = compute_conviction(
        direction_move=-0.01,
        vwap_extension=0.005,
        short_momentum=0.005,
        spread_pct=0.05,
        filters=f,
    )
    assert c == 0.0


def test_compute_conviction_passes_when_aligned():
    """Both moves positive → conviction computed normally."""
    f = _f()
    c = compute_conviction(
        direction_move=0.01,
        vwap_extension=0.005,
        short_momentum=0.005,
        spread_pct=0.05,
        filters=f,
    )
    assert c > 0.0


def test_compute_conviction_perfect_setup():
    """max direction, tight vwap, big momentum, tight spread → conviction = 1.0."""
    f = _f()
    c = compute_conviction(
        direction_move=0.10,    # way above 0.003 threshold
        vwap_extension=0.001,   # way below 0.020 threshold
        short_momentum=0.05,    # way above 0.001 threshold
        spread_pct=0.03,        # below 0.05 → 1.0 spread_score
        filters=f,
    )
    # weighted mean of (1.0, 1.0, 1.0, 1.0) = 1.0
    assert c == 1.0
