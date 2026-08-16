"""Tests for the market-time helpers (ET conversion, DST, weekend awareness)."""

from datetime import UTC, datetime

from flip_options_bot.market_time import (
    is_entry_window,
    is_market_open,
    is_weekday,
    minutes_to_close,
    minutes_to_open,
    now_utc,
    to_et,
    today_et_iso_date,
)


def test_to_et_handles_edt():
    """August = EDT (UTC-4)."""
    utc_dt = datetime(2026, 8, 12, 14, 0, 0, tzinfo=UTC)
    et = to_et(utc_dt)
    assert et.hour == 10
    assert et.minute == 0


def test_to_et_handles_est():
    """November = EST (UTC-5)."""
    utc_dt = datetime(2026, 11, 5, 14, 0, 0, tzinfo=UTC)
    et = to_et(utc_dt)
    assert et.hour == 9


def test_is_market_open_during_session():
    """10:00 ET on a Tuesday should be market-open."""
    utc_dt = datetime(2026, 8, 11, 14, 0, 0, tzinfo=UTC)  # Tue 10:00 ET
    assert is_market_open(utc_dt) is True


def test_is_market_open_after_close():
    """17:00 ET should NOT be market-open."""
    utc_dt = datetime(2026, 8, 11, 21, 0, 0, tzinfo=UTC)  # Tue 17:00 ET
    assert is_market_open(utc_dt) is False


def test_is_market_open_weekend():
    """Saturday should NOT be market-open even at 10am ET."""
    utc_dt = datetime(2026, 8, 15, 14, 0, 0, tzinfo=UTC)  # Sat 10:00 ET
    assert is_market_open(utc_dt) is False


def test_is_market_open_rejects_labor_day():
    utc_dt = datetime(2026, 9, 7, 14, 0, 0, tzinfo=UTC)  # Labor Day 10:00 ET
    assert is_weekday(utc_dt) is False
    assert is_market_open(utc_dt) is False
    assert is_entry_window(utc_dt) is False


def test_is_entry_window_excludes_first_15_min():
    """09:35 ET — within market hours but outside entry window."""
    utc_dt = datetime(2026, 8, 11, 13, 35, 0, tzinfo=UTC)  # Tue 09:35 ET
    assert is_market_open(utc_dt) is True
    assert is_entry_window(utc_dt) is False


def test_is_entry_window_excludes_last_15_min():
    """15:50 ET — within market hours but outside entry window."""
    utc_dt = datetime(2026, 8, 11, 19, 50, 0, tzinfo=UTC)  # Tue 15:50 ET
    assert is_market_open(utc_dt) is True
    assert is_entry_window(utc_dt) is False


def test_is_entry_window_allows_mid_session():
    """11:00 ET — full mid-session, should be in entry window."""
    utc_dt = datetime(2026, 8, 11, 15, 0, 0, tzinfo=UTC)  # Tue 11:00 ET
    assert is_entry_window(utc_dt) is True


def test_minutes_to_open_when_open_now():
    """11:00 ET should return -1 (open now)."""
    utc_dt = datetime(2026, 8, 11, 15, 0, 0, tzinfo=UTC)
    assert minutes_to_open(utc_dt) == -1


def test_minutes_to_open_pre_market_weekday():
    """08:00 ET should return ~90 minutes."""
    utc_dt = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)  # 08:00 ET
    m = minutes_to_open(utc_dt)
    assert 80 <= m <= 100


def test_minutes_to_open_friday_evening_to_monday():
    """Friday 18:00 ET → next market open is Monday 09:30 ET."""
    utc_dt = datetime(2026, 8, 14, 22, 0, 0, tzinfo=UTC)  # Fri 18:00 ET
    m = minutes_to_open(utc_dt)
    # Friday 18:00 ET to Monday 09:30 ET = 63.5 hours = 3810 minutes
    assert 3750 <= m <= 3850


def test_is_weekday_basic():
    """Mon-Fri = True, Sat-Sun = False."""
    assert is_weekday(datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)) is True  # Tue
    assert is_weekday(datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)) is False  # Sat
    assert is_weekday(datetime(2026, 8, 16, 12, 0, 0, tzinfo=UTC)) is False  # Sun


def test_now_utc_is_aware():
    n = now_utc()
    assert n.tzinfo is not None
    assert n.tzinfo == UTC


def test_today_et_iso_date_returns_str():
    s = today_et_iso_date()
    assert isinstance(s, str)
    assert len(s) == 10
    assert s[4] == "-" and s[7] == "-"


def test_minutes_to_close_past_close_returns_minus_one():
    """Past close → -1."""
    utc_dt = datetime(2026, 8, 11, 21, 0, 0, tzinfo=UTC)  # Tue 17:00 ET
    assert minutes_to_close(utc_dt) == -1


def test_minutes_to_close_at_open():
    """09:30 ET → 6.5 hours to close = 390 min."""
    utc_dt = datetime(2026, 8, 11, 13, 30, 0, tzinfo=UTC)
    assert minutes_to_close(utc_dt) == 390
