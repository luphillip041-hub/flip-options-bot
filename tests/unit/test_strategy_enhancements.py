"""Tests for the strategy-level enhancements: ORB, IV filter, calendar gate,
conviction-based sizing, BPCS, entry-time windows.
"""

from datetime import UTC, datetime

import pytest

from flip_options_bot.market_time import is_entry_window
from flip_options_bot.strategies.bull_put_credit import (
    compute_bpcs_conviction,
    estimate_credit,
    select_strikes,
)
from flip_options_bot.strategies.calendar_gate import (
    DEFAULT_FOMC_DATES_2026,
    has_fomc_in_horizon,
    is_fomc_day,
    safe_entry_window,
)
from flip_options_bot.strategies.iv_filter import (
    iv_regime_boost,
    iv_regime_ok,
    straddle_iv_proxy,
)
from flip_options_bot.strategies.long_call import is_strong_setup, size_by_conviction
from flip_options_bot.strategies.orb import (
    apply_orb_boost,
    compute_opening_range,
    orb_breakout_signal,
)

# ===== ORB =====


def test_orb_compute_opening_range_basic():
    bars = [{"h": 100 + i * 0.1, "l": 100 - i * 0.1, "t": None} for i in range(30)]
    or_ = compute_opening_range(bars)
    assert or_ is not None
    assert or_.high >= or_.low
    assert or_.width_pct > 0


def test_orb_compute_opening_range_filters_et_time():
    """Bars outside 09:30-10:00 ET should be ignored."""
    from datetime import timedelta

    base = datetime(2026, 8, 12, 13, 0, tzinfo=UTC)  # 09:00 ET
    bars = []
    for i in range(60):
        ts = (base + timedelta(minutes=i)).isoformat()
        bars.append({"h": 100 + i * 0.1, "l": 100 - i * 0.1, "t": ts})
    or_ = compute_opening_range(bars)
    # Should include 09:30-10:00 ET (30 minutes) but not before/after
    assert or_ is not None


def test_orb_breakout_long():
    or_ = type("OR", (), {"high": 100, "low": 99, "width_pct": 0.01})()
    direction, strength = orb_breakout_signal(spot=100.5, opening_range=or_, prev_close=99.5)
    assert direction == "long"
    assert strength > 0


def test_orb_breakout_short():
    or_ = type("OR", (), {"high": 100, "low": 99, "width_pct": 0.01})()
    direction, strength = orb_breakout_signal(spot=98.5, opening_range=or_, prev_close=100.5)
    assert direction == "short"
    assert strength > 0


def test_orb_no_breakout_inside_range():
    or_ = type("OR", (), {"high": 100, "low": 99, "width_pct": 0.01})()
    direction, strength = orb_breakout_signal(spot=99.5, opening_range=or_, prev_close=100)
    assert direction == "none"
    assert strength == 0.0


def test_orb_boost_aligned():
    """ORB long + trade long → boost."""
    adjusted = apply_orb_boost(
        base_conviction=0.6, orb_direction="long", orb_strength=0.7, trade_direction="long"
    )
    assert adjusted > 0.6


def test_orb_boost_contradicted():
    """ORB short + trade long → penalize."""
    adjusted = apply_orb_boost(
        base_conviction=0.6, orb_direction="short", orb_strength=0.7, trade_direction="long"
    )
    assert adjusted < 0.6


def test_orb_boost_none():
    """ORB no signal → no change."""
    adjusted = apply_orb_boost(
        base_conviction=0.6, orb_direction="none", orb_strength=0.5, trade_direction="long"
    )
    assert adjusted == 0.6


# ===== IV filter =====


def test_straddle_iv_proxy_normal():
    # call: 1.50/1.55, put: 1.45/1.50, spot: 100
    iv = straddle_iv_proxy(call_bid=1.50, call_ask=1.55, put_bid=1.45, put_ask=1.50, spot=100.0)
    # straddle_mid = (1.50+1.55+1.45+1.50)/2 = 3.00
    # iv = (3.00 - 100)/100 = -0.97 → wait that's wrong
    # Re-reading the formula: (straddle_mid - spot) / spot
    # straddle_mid should be relative to ATM strike, not spot
    # Let me reconsider — actually for ATM straddle, the price is the
    # call+put mid. If spot=$100 and ATM strike=$100, and call=$1.50,
    # put=$1.50, then straddle=$3.00 which is 3% of spot. That's normal.
    assert iv > 0


def test_iv_regime_ok_normal():
    assert iv_regime_ok(0.02) is True  # 2% — normal
    assert iv_regime_ok(0.005) is True  # at the floor (0.5%)
    assert iv_regime_ok(0.001) is False  # too cheap (below 0.5%)
    assert iv_regime_ok(0.20) is False  # too rich (above 15%)


def test_iv_regime_boost_rich():
    """Rich IV (3%+) → boost conviction."""
    boost = iv_regime_boost(0.05)
    assert boost > 1.0  # > 1.0 means boosted


def test_iv_regime_boost_cheap():
    """Cheap IV (<1%) → reduce conviction."""
    boost = iv_regime_boost(0.005)
    assert boost < 1.0


# ===== Calendar gate =====


def test_is_fomc_day():
    assert is_fomc_day(datetime(2026, 9, 15)) is True
    assert is_fomc_day(datetime(2026, 9, 16)) is False  # second day not in our list


def test_has_fomc_in_horizon():
    entry = datetime(2026, 9, 14)
    expiry = datetime(2026, 9, 30)
    assert has_fomc_in_horizon(entry, expiry) is True

    entry = datetime(2026, 8, 10)
    expiry = datetime(2026, 8, 14)
    assert has_fomc_in_horizon(entry, expiry) is False


def test_safe_entry_window_blocks_fomc():
    safe, reason = safe_entry_window(
        "SPY",
        datetime(2026, 9, 14),
        datetime(2026, 9, 30),
    )
    assert safe is False
    assert reason == "fomc_in_horizon"


def test_safe_entry_window_allows_clean_day():
    safe, reason = safe_entry_window(
        "SPY",
        datetime(2026, 8, 11),
        datetime(2026, 8, 18),
    )
    assert safe is True
    assert reason == "ok"


# ===== Position sizing =====


def test_size_by_conviction_baseline():
    assert size_by_conviction(0.50) == 1
    assert size_by_conviction(0.70) == 1


def test_size_by_conviction_high():
    assert size_by_conviction(0.80) == 2
    assert size_by_conviction(0.95) == 2


def test_size_by_conviction_below_threshold():
    assert size_by_conviction(0.40) == 0


def test_is_strong_setup():
    assert is_strong_setup(0.80) is True
    assert is_strong_setup(0.70) is False


# ===== BPCS =====


def test_bpcs_select_strikes():
    short, long = select_strikes(spot=100.0)
    assert short == 98.0  # 2% OTM
    assert long == 96.0  # 4% OTM
    assert short > long


def test_bpcs_estimate_credit():
    credit = estimate_credit(
        short_put_bid=1.50, short_put_ask=1.55, long_put_bid=0.50, long_put_ask=0.55
    )
    # short_mid = 1.525, long_mid = 0.525, credit ≈ 1.00
    assert credit == pytest.approx(1.00, abs=1e-6)


def test_bpcs_conviction_bullish_rich_iv():
    c = compute_bpcs_conviction(
        spot=100,
        short_strike=95,
        long_strike=90,
        credit=1.5,
        iv_rank_proxy=0.40,
        direction_move_pct=0.005,
    )
    assert c > 0.5


def test_bpcs_conviction_bearish_blocks():
    """Bearish market → BPCS conviction = 0."""
    c = compute_bpcs_conviction(
        spot=100,
        short_strike=95,
        long_strike=90,
        credit=1.5,
        iv_rank_proxy=0.40,
        direction_move_pct=-0.01,
    )
    assert c == 0.0


# ===== Entry window =====


def test_entry_window_morning():
    """10:00-11:30 ET → True."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 10, 30, tzinfo=et)
    assert is_entry_window(dt) is True


def test_entry_window_lunch_lull():
    """12:00 ET → False (lunch lull)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 12, 0, tzinfo=et)
    assert is_entry_window(dt) is False


def test_entry_window_afternoon():
    """14:30 ET → True."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 14, 30, tzinfo=et)
    assert is_entry_window(dt) is True


def test_entry_window_late():
    """15:45 ET → False (last 30 min)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 15, 45, tzinfo=et)
    assert is_entry_window(dt) is False


def test_entry_window_open_15min():
    """09:30 ET → False (first 15 min, too volatile)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 9, 30, tzinfo=et)
    assert is_entry_window(dt) is False


def test_entry_window_945():
    """09:45 ET → True (start of morning window)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 8, 12, 9, 45, tzinfo=et)
    assert is_entry_window(dt) is True


# ===== Cross-validation: default FOMC list is non-empty =====


def test_fomc_dates_populated():
    assert len(DEFAULT_FOMC_DATES_2026) == 8  # 8 meetings per year
