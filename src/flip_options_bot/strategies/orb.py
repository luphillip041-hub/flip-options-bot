"""Opening Range Breakout (ORB) signal layer.

ORB is one of the most reliable intraday edges. The first 30 minutes
of the session (09:30-10:00 ET) establish a "range" — high and low.
A breakout above the high or below the low, with volume confirmation,
has historically a 60%+ continuation rate over the next 1-3 hours.

This module exposes:
- compute_opening_range(bars_30min): get high/low of the first 30 min
- orb_breakout_signal(spot, opening_range, prev_close): direction + strength

We integrate this as a CONVICTION BOOST, not a separate strategy. The
base long_call strategy still decides what to trade; ORB simply
multiplies conviction when the breakout aligns with the trend.

Usage:
  or_high, or_low, or_width = compute_opening_range(morning_bars)
  orb_dir, orb_strength = orb_breakout_signal(spot, (or_high, or_low), prev_close)
  adjusted_conviction = apply_orb_boost(conviction, orb_dir, direction_move, orb_strength)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass
class OpeningRange:
    high: float
    low: float
    width_pct: float  # (high - low) / low
    start_utc: datetime | None = None
    end_utc: datetime | None = None


def _et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET)


def compute_opening_range(morning_bars: list[dict]) -> OpeningRange | None:
    """Compute the opening range from morning bars.

    `morning_bars` should be minute bars covering 09:30-10:00 ET.
    Returns the high/low of those bars, or None if too few bars.

    Bars format: each bar has 'h', 'l', 't' keys.
    """
    if not morning_bars:
        return None
    # If bars have ET timestamps, filter to 09:30-10:00. Otherwise just take all.
    session_bars = []
    for b in morning_bars:
        ts = b.get("t")
        if ts is None:
            continue
        try:
            bar_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            et_time = _et(bar_dt).time()
            if time(9, 30) <= et_time < time(10, 0):
                session_bars.append(b)
        except (ValueError, TypeError):
            continue
    if not session_bars:
        # Fall back to first 30 bars (assume they're the morning session)
        session_bars = morning_bars[:30]
    if len(session_bars) < 5:
        return None  # not enough morning data

    highs = [float(b["h"]) for b in session_bars if b.get("h") is not None]
    lows = [float(b["l"]) for b in session_bars if b.get("l") is not None]
    if not highs or not lows:
        return None
    or_high = max(highs)
    or_low = min(lows)
    width_pct = (or_high - or_low) / or_low if or_low > 0 else 0.0

    # Compute window bounds for diagnostics (may be None if bars missing ts)
    start_utc = None
    end_utc = None
    if session_bars:
        start_raw = session_bars[0].get("t")
        end_raw = session_bars[-1].get("t")
        if isinstance(start_raw, str):
            start_utc = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        elif isinstance(start_raw, datetime):
            start_utc = start_raw
        if isinstance(end_raw, str):
            end_utc = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
        elif isinstance(end_raw, datetime):
            end_utc = end_raw

    return OpeningRange(
        high=or_high,
        low=or_low,
        width_pct=width_pct,
        start_utc=start_utc,
        end_utc=end_utc,
    )


def orb_breakout_signal(
    spot: float,
    opening_range: OpeningRange,
    prev_close: float,
) -> tuple[str, float]:
    """Detect an opening-range breakout.

    Returns (direction, strength):
      direction: "long", "short", or "none"
      strength: 0.0 to 1.0 (0 = no signal, 1 = strong breakout)

    Logic:
      - If spot > OR_high → "long" (breakout above)
      - If spot < OR_low → "short" (breakdown below)
      - Strength = how far beyond the level (in % of OR width)
      - If OR width is too narrow (< 0.10%) the breakout is suspect → strength *= 0.5
    """
    if spot <= 0 or opening_range.high <= 0 or opening_range.low <= 0:
        return ("none", 0.0)

    or_width = opening_range.high - opening_range.low
    if or_width <= 0:
        return ("none", 0.0)

    # LONG breakout
    if spot > opening_range.high:
        strength = (spot - opening_range.high) / or_width
        # Penalize narrow OR (low conviction breakout)
        if opening_range.width_pct < 0.0010:  # 0.10%
            strength *= 0.5
        # Bonus: aligned with prior close (gap up is more bullish)
        if prev_close > 0 and spot > prev_close:
            strength = min(1.0, strength * 1.20)
        return ("long", min(1.0, strength))

    # SHORT breakdown
    if spot < opening_range.low:
        strength = (opening_range.low - spot) / or_width
        if opening_range.width_pct < 0.0010:
            strength *= 0.5
        if prev_close > 0 and spot < prev_close:
            strength = min(1.0, strength * 1.20)
        return ("short", min(1.0, strength))

    return ("none", 0.0)


def apply_orb_boost(
    base_conviction: float,
    orb_direction: str,
    orb_strength: float,
    trade_direction: str,  # "long" or "short"
) -> float:
    """Apply an ORB conviction multiplier.

    Rules:
      - If ORB direction matches trade direction → boost
      - If ORB direction contradicts trade direction → penalize
      - If ORB is "none" → no change
      - Multiplier scales linearly with strength

    Returns adjusted conviction in [0, 1].
    """
    if orb_direction == "none" or orb_strength <= 0:
        return base_conviction

    # Multiplier range: [0.50, 1.50]
    if orb_direction == trade_direction:
        # Aligned — boost
        multiplier = 1.0 + 0.50 * orb_strength  # 1.0 to 1.5
    else:
        # Contradicted — penalize
        multiplier = 1.0 - 0.50 * orb_strength  # 1.0 down to 0.5

    adjusted = base_conviction * multiplier
    return min(1.0, max(0.0, adjusted))