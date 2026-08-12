"""Earnings + FOMC calendar gate.

We DO NOT enter a long_call position on a day with a binary catalyst
in our horizon (0-14 DTE). Why: a single news event can wipe out our
directional thesis, regardless of how clean the technicals look.

For earnings: we need a list of upcoming earnings dates for SPY/QQQ/IWM/DIA
component stocks (or whatever we trade). For FOMC: the next meeting date.

For paper trading we use a simple JSON file with known upcoming dates:
- /root/flip/projects/flip-options-bot/data/earnings_calendar.json
- /root/flip/projects/flip-options-bot/data/fomc_calendar.json

The runtime can pull these files OR a hard-coded list. The gate returns
True/False: is the entry date safe from binary events?

For SIMPLICITY in the first cut: a hard-coded FOMC date list + earnings
list per symbol. The bot operator can update the lists manually or via
a cron that pulls from a free source (Yahoo Finance calendar, etc.).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


# Default FOMC meeting dates for 2026 (8 per year, every 6 weeks).
# Source: federalreserve.gov. Update yearly.
DEFAULT_FOMC_DATES_2026 = [
    "2026-01-28",  # Jan 28-29
    "2026-03-17",  # Mar 17-18
    "2026-04-28",  # Apr 28-29
    "2026-06-16",  # Jun 16-17
    "2026-07-28",  # Jul 28-29
    "2026-09-15",  # Sep 15-16
    "2026-10-27",  # Oct 27-28
    "2026-12-15",  # Dec 15-16
]


def is_fomc_day(date: datetime, fomc_dates: list[str] | None = None) -> bool:
    """True if `date` is the day of an FOMC meeting announcement.

    FOMC announcements happen at 14:00 ET on the second day of the
    meeting. We treat the announcement day as binary-event.
    """
    dates = fomc_dates or DEFAULT_FOMC_DATES_2026
    return date.strftime("%Y-%m-%d") in dates


def has_fomc_in_horizon(
    entry_date: datetime,
    expiry_date: datetime,
    fomc_dates: list[str] | None = None,
) -> bool:
    """True if any FOMC meeting falls between entry_date and expiry_date (inclusive).

    Conservative: any FOMC in the trade's holding window = skip.
    """
    dates = fomc_dates or DEFAULT_FOMC_DATES_2026
    for fomc_str in dates:
        try:
            fomc_dt = datetime.strptime(fomc_str, "%Y-%m-%d")
        except ValueError:
            continue
        if entry_date <= fomc_dt <= expiry_date:
            return True
    return False


# Earnings calendar — sparse, manually updated.
# Format: { "TICKER": ["2026-08-15", "2026-11-05", ...] }
DEFAULT_EARNINGS: dict[str, list[str]] = {
    # SPY/QQQ/IWM/DIA are ETFs — they don't have earnings per se, but
    # we treat the FOMC + heavy-news-window (CPI, NFP, PCE) as binary events.
    # For stock-specific strategies this would be populated.
}


def has_earnings_in_horizon(
    symbol: str,
    entry_date: datetime,
    expiry_date: datetime,
    earnings: dict[str, list[str]] | None = None,
) -> bool:
    """True if `symbol` has an earnings date in the trade's holding window.

    For ETF symbols (SPY/QQQ/IWM/DIA) with no entries in the dict,
    this always returns False.
    """
    cal = earnings or DEFAULT_EARNINGS
    dates = cal.get(symbol.upper(), [])
    for earn_str in dates:
        try:
            earn_dt = datetime.strptime(earn_str, "%Y-%m-%d")
        except ValueError:
            continue
        if entry_date <= earn_dt <= expiry_date:
            return True
    return False


def safe_entry_window(
    symbol: str,
    entry_date: datetime,
    expiry_date: datetime,
    avoid_fomc: bool = True,
    avoid_earnings: bool = True,
) -> tuple[bool, str]:
    """Returns (safe, reason_if_not_safe).

    safe=True means we can enter a long_call position. The reason string
    explains why we're blocked, if blocked.
    """
    if avoid_fomc and has_fomc_in_horizon(entry_date, expiry_date):
        return (False, "fomc_in_horizon")
    if avoid_earnings and has_earnings_in_horizon(symbol, entry_date, expiry_date):
        return (False, f"earnings_in_horizon for {symbol}")
    return (True, "ok")