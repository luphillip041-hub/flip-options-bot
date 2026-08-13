"""Tests for the long_call strategy — pure functions, no broker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flip_options_bot.strategies.long_call import (
    LongCallFilters,
    compute_conviction,
    passes_conviction,
    passes_dte_window,
    pick_target_expiry,
)


def test_passes_dte_window_prefers_zero_but_allows_two_weeks():
    f = LongCallFilters(
        min_dte=0, target_dte=0, max_dte=14,
        min_direction_move_pct=0.003, max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.001, min_conviction=0.45,
        directional_lookback_minutes=20,
    )
    assert passes_dte_window(0, f) is True
    assert passes_dte_window(14, f) is True
    assert passes_dte_window(15, f) is False  # too long


def test_pick_target_expiry_prefers_0dte_with_two_week_fallback():
    f = LongCallFilters(
        min_dte=0, target_dte=0, max_dte=14,
        min_direction_move_pct=0.003, max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.001, min_conviction=0.45,
        directional_lookback_minutes=20,
    )
    today = datetime.now(timezone.utc).date()
    expiries = [
        (today + timedelta(days=14)).strftime("%Y-%m-%d"),
        (today + timedelta(days=7)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    ]
    assert pick_target_expiry(expiries, f) == today.strftime("%Y-%m-%d")
    fallback_expiries = [
        (today + timedelta(days=14)).strftime("%Y-%m-%d"),
        (today + timedelta(days=7)).strftime("%Y-%m-%d"),
    ]
    assert pick_target_expiry(fallback_expiries, f) == (today + timedelta(days=7)).strftime("%Y-%m-%d")
    assert pick_target_expiry([], f) is None


def test_compute_conviction_high_when_all_signals_fire():
    f = LongCallFilters(
        min_dte=1, target_dte=5, max_dte=14,
        min_direction_move_pct=0.003, max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.001, min_conviction=0.45,
        directional_lookback_minutes=20,
    )
    c = compute_conviction(
        direction_move=0.005,  # > 0.003
        vwap_extension=0.015,   # < 0.020
        short_momentum=0.002,   # > 0.001
        spread_pct=0.04,        # < 0.05 (perfect)
        filters=f,
    )
    assert c > 0.85


def test_compute_conviction_low_when_no_signals():
    f = LongCallFilters(
        min_dte=1, target_dte=5, max_dte=14,
        min_direction_move_pct=0.003, max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.001, min_conviction=0.45,
        directional_lookback_minutes=20,
    )
    # dir_score=0.33, vwap_score=0 (over extension), mom_score=0.5, spread_score=0.67
    # weighted = 0.30*0.33 + 0.20*0.0 + 0.30*0.5 + 0.20*0.67 = 0.10 + 0 + 0.15 + 0.133 ≈ 0.38
    # Under the 0.45 conviction floor → no trade.
    c = compute_conviction(
        direction_move=0.001,
        vwap_extension=0.025,
        short_momentum=0.0005,
        spread_pct=0.10,
        filters=f,
    )
    assert c < f.min_conviction  # doesn't pass the conviction floor


def test_passes_conviction_threshold():
    f = LongCallFilters(
        min_dte=1, target_dte=5, max_dte=14,
        min_direction_move_pct=0.003, max_vwap_extension_pct=0.020,
        min_short_momentum_pct=0.001, min_conviction=0.45,
        directional_lookback_minutes=20,
    )
    assert passes_conviction(0.50, f) is True
    assert passes_conviction(0.40, f) is False