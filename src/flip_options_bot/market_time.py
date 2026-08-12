"""US market hours — proper ET conversion with DST awareness.

Used by:
- daemon.scan_window gate (don't scan outside market hours)
- position_monitor._minutes_to_close (EOD flatten)
- observation.end_of_day_record hook (post-market day-record)

No pytz dependency; we use the stdlib zoneinfo (added in Python 3.9).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")  # handles EST/EDT automatically

# US market hours (regular session)
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)

# EOD flatten starts this many minutes before close
DEFAULT_EOD_FLATTEN_MINUTES = 15

# Allowed order time window
DEFAULT_ENTRY_OPEN_ET = time(9, 45)   # don't fire first 15 min of open
DEFAULT_ENTRY_CLOSE_ET = time(15, 45)  # last 15 min too volatile


def now_utc() -> datetime:
    """UTC-aware now."""
    return datetime.now(timezone.utc)


def to_et(dt_utc: datetime) -> datetime:
    """UTC → ET (handles DST automatically)."""
    return dt_utc.astimezone(ET)


def is_weekday(dt_utc: datetime | None = None) -> bool:
    """Mon-Fri in ET. Doesn't account for market holidays (NYSE/NASDAQ)."""
    d = to_et(dt_utc or now_utc())
    return d.weekday() < 5


def is_market_open(dt_utc: datetime | None = None) -> bool:
    """True if regular-session market hours and a weekday."""
    d = to_et(dt_utc or now_utc())
    if d.weekday() >= 5:
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
    if d.weekday() >= 5:
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
    close_dt = d.replace(hour=MARKET_CLOSE_ET.hour, minute=MARKET_CLOSE_ET.minute,
                         second=0, microsecond=0)
    diff_minutes = (close_dt - d).total_seconds() / 60.0
    if diff_minutes < 0:
        return -1  # past today's close
    return int(diff_minutes)


def minutes_to_open(dt_utc: datetime | None = None) -> int:
    """Minutes until next 09:30 ET (today or tomorrow). Returns -1 if open now."""
    d = to_et(dt_utc or now_utc())
    open_today = d.replace(hour=MARKET_OPEN_ET.hour, minute=MARKET_OPEN_ET.minute,
                            second=0, microsecond=0)
    now_t = d.time()
    if MARKET_OPEN_ET <= now_t < MARKET_CLOSE_ET and d.weekday() < 5:
        return -1
    if now_t < MARKET_OPEN_ET and d.weekday() < 5:
        return int((open_today - d).total_seconds() / 60.0)
    # After close or weekend — next weekday morning
    days_ahead = 1
    while True:
        next_day = d + timedelta(days=days_ahead)
        if next_day.weekday() < 5:
            next_open = next_day.replace(hour=MARKET_OPEN_ET.hour,
                                          minute=MARKET_OPEN_ET.minute,
                                          second=0, microsecond=0)
            return int((next_open - d).total_seconds() / 60.0)
        days_ahead += 1
        if days_ahead > 14:
            return -1  # shouldn't happen


def today_et_iso_date() -> str:
    """YYYY-MM-DD string for today in ET (NOT UTC). For observation-day recording."""
    return to_et(now_utc()).strftime("%Y-%m-%d")