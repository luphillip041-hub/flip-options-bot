"""US market hours — proper ET conversion with DST awareness.

Used by:
- daemon.scan_window gate (don't scan outside market hours)
- position_monitor._minutes_to_close (EOD flatten)
- observation.end_of_day_record hook (post-market day-record)

No pytz dependency; we use the stdlib zoneinfo (added in Python 3.9).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")  # handles EST/EDT automatically

# US market hours (regular session)
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)

# EOD flatten starts this many minutes before close
DEFAULT_EOD_FLATTEN_MINUTES = 15

# Allowed order time window
DEFAULT_ENTRY_OPEN_ET = time(9, 45)  # don't fire first 15 min of open
DEFAULT_ENTRY_CLOSE_ET = time(15, 45)  # last 15 min too volatile


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Gregorian Easter (Meeus/Jones/Butcher), used for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * correction) // 451
    month = (h + correction - 7 * m + 114) // 31
    day = (h + correction - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def market_holidays(year: int) -> frozenset[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    holidays.add(_observed(date(year + 1, 1, 1)))
    return frozenset(day for day in holidays if day.year == year)


def is_market_session_date(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def now_utc() -> datetime:
    """UTC-aware now."""
    return datetime.now(UTC)


def to_et(dt_utc: datetime) -> datetime:
    """UTC → ET (handles DST automatically)."""
    return dt_utc.astimezone(ET)


def is_weekday(dt_utc: datetime | None = None) -> bool:
    """True on a scheduled regular US equity session date."""
    d = to_et(dt_utc or now_utc())
    return is_market_session_date(d.date())


def is_market_open(dt_utc: datetime | None = None) -> bool:
    """True during scheduled regular-session market hours."""
    d = to_et(dt_utc or now_utc())
    if not is_market_session_date(d.date()):
        return False
    return MARKET_OPEN_ET <= d.time() < MARKET_CLOSE_ET


def is_entry_window(dt_utc: datetime | None = None) -> bool:
    """True if a new trade entry is allowed.

    Default windows (avoiding open/close volatility AND lunch lull):
      - 09:45-11:30 ET (morning ORB + continuation)
      - 14:00-15:30 ET (afternoon continuation)

    The 09:30-09:45 gap is just the first 15 min — too volatile.
    The lunch lull (11:30-14:00) is low-volume + choppy — avoid it.
    The last 30 min (15:30-16:00) have gamma + spread issues — avoid.
    """
    d = to_et(dt_utc or now_utc())
    if not is_market_session_date(d.date()):
        return False
    t = d.time()
    # Morning window: 09:45 - 11:30 (includes ORB breakouts after 10:00 open)
    if time(9, 45) <= t < time(11, 30):
        return True
    # Afternoon window: 14:00 - 15:30
    if time(14, 0) <= t < time(15, 30):
        return True
    return False


def minutes_to_close(dt_utc: datetime | None = None) -> int:
    """Minutes until 16:00 ET today. Returns -1 if market is closed for the day
    (after 16:00 ET, or before today's open is fine).

    Negative means we're past today's close.
    """
    d = to_et(dt_utc or now_utc())
    close_dt = d.replace(
        hour=MARKET_CLOSE_ET.hour, minute=MARKET_CLOSE_ET.minute, second=0, microsecond=0
    )
    diff_minutes = (close_dt - d).total_seconds() / 60.0
    if diff_minutes < 0:
        return -1  # past today's close
    return int(diff_minutes)


def minutes_to_open(dt_utc: datetime | None = None) -> int:
    """Minutes until next 09:30 ET (today or tomorrow). Returns -1 if open now."""
    d = to_et(dt_utc or now_utc())
    open_today = d.replace(
        hour=MARKET_OPEN_ET.hour, minute=MARKET_OPEN_ET.minute, second=0, microsecond=0
    )
    now_t = d.time()
    if MARKET_OPEN_ET <= now_t < MARKET_CLOSE_ET and is_market_session_date(d.date()):
        return -1
    if now_t < MARKET_OPEN_ET and is_market_session_date(d.date()):
        return int((open_today - d).total_seconds() / 60.0)
    # After close or weekend — next weekday morning
    days_ahead = 1
    while True:
        next_day = d + timedelta(days=days_ahead)
        if is_market_session_date(next_day.date()):
            next_open = next_day.replace(
                hour=MARKET_OPEN_ET.hour, minute=MARKET_OPEN_ET.minute, second=0, microsecond=0
            )
            return int((next_open - d).total_seconds() / 60.0)
        days_ahead += 1
        if days_ahead > 14:
            return -1  # shouldn't happen


def today_et_iso_date() -> str:
    """YYYY-MM-DD string for today in ET (NOT UTC). For observation-day recording."""
    return to_et(now_utc()).strftime("%Y-%m-%d")
