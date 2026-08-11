"""Tests for the position monitor.

These tests use mocked broker + journal + closer so no live API calls.
They verify:
- monitor.tick returns 0 closes when no positions are open
- SL trigger fires when mark <= sl_threshold_pct of entry
- TP trigger fires when mark >= tp_multiplier of entry
- Trailing-floor trigger fires after arm + drop
- EOD trigger fires within close_eod_minutes of 16:00 ET
- monitor does NOT trigger when positions are healthy
- monitor's close calls go through the Closer (not the broker directly)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from flip_options_bot.execution import Closer
from flip_options_bot.journal import Journal, TradeEvent
from flip_options_bot.market_time import minutes_to_close
from flip_options_bot.monitor.position_monitor import PositionMonitor


def _open_position(journal: Journal, symbol: str, qty: int, avg_entry: float) -> str:
    pos_id = Journal.new_position_id()
    open_ev = TradeEvent(
        event_id=f"open-{pos_id}",
        ts=Journal.now_iso(),
        kind="open",
        symbol=symbol,
        side="buy",
        qty=qty,
        price=avg_entry,
        position_id=pos_id,
        strategy_id="long_call",
    )
    journal.append(open_ev)
    return pos_id


def _wire_monitor(tmp_path: Path, snapshot: dict | None, **settings_kwargs) -> tuple:
    journal = Journal(tmp_path / "runs")
    broker = MagicMock()
    broker.get_option_snapshot.return_value = snapshot
    closer = Closer.__new__(Closer)
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = MagicMock()
    closer.flatten_position = MagicMock(return_value=MagicMock(accepted=True))
    settings = MagicMock()
    settings.close_eod_minutes = settings_kwargs.get("close_eod_minutes", 5)
    settings.sl_threshold_pct = settings_kwargs.get("sl_threshold_pct", 0.50)
    settings.tp_multiplier = settings_kwargs.get("tp_multiplier", 1.50)
    settings.trailing_arm_pct = settings_kwargs.get("trailing_arm_pct", 0.10)
    settings.trailing_retention = settings_kwargs.get("trailing_retention", 0.50)
    monitor = PositionMonitor(settings, broker, journal, closer.risk, closer)
    monitor.journal = journal
    return monitor, journal, closer


def test_no_positions_no_closes(tmp_path: Path) -> None:
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 1.0, "ask": 1.10}
    )
    tick = monitor.tick(closer.risk)
    assert tick.positions_seen == 0
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_sl_trigger_fires_when_mark_below_50pct(tmp_path: Path) -> None:
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 0.50, "ask": 0.55}
    )
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 1
    assert tick.reasons.get("sl") == 1
    closer.flatten_position.assert_called_once()
    kwargs = closer.flatten_position.call_args.kwargs
    assert kwargs["symbol"] == "SPY260815C00770000"
    assert kwargs["reason"] == "sl"


def test_tp_trigger_fires_when_mark_above_tp_multiplier(tmp_path: Path) -> None:
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 3.90, "ask": 4.00}
    )
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 1
    assert tick.reasons.get("tp") == 1


def test_trailing_floor_trigger_after_arm_and_drop(tmp_path: Path) -> None:
    """Position peaked at +30% then dropped below retention (50%) of peak."""
    monitor, journal, closer = _wire_monitor(tmp_path, snapshot=None)
    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    # Manually set peak to 3.25 (+30%), then current mark = 1.60 (below 3.25*0.5=1.625)
    import sqlite3
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute("UPDATE positions SET peak_mark = ? WHERE position_id = ?", (3.25, pos_id))
        conn.commit()
    # Now set a snapshot that gives mark = 1.60
    monitor.broker.get_option_snapshot.return_value = {"bid": 1.55, "ask": 1.65}
    tick = monitor.tick(closer.risk)
    # mark=1.60, avg=2.50, peak=3.25
    # gain_pct = (3.25-2.50)/2.50 = 0.30 (>= arm 0.10)
    # retention_target = 3.25 * 0.50 = 1.625; mark=1.60 <= 1.625 → trailing_floor
    assert tick.closes_triggered == 1
    assert tick.reasons.get("trailing_floor") == 1


def test_no_trigger_when_position_healthy(tmp_path: Path) -> None:
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 2.50, "ask": 2.60}
    )
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_skips_position_with_zero_remaining_qty(tmp_path: Path) -> None:
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 0.50, "ask": 0.55}
    )
    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    journal.append(TradeEvent(
        event_id="close-skip",
        ts=Journal.now_iso(),
        kind="close",
        symbol="SPY260815C00770000",
        side="sell",
        qty=1,
        price=2.75,
        position_id=pos_id,
        realized_pnl=0.25,
    ))
    tick = monitor.tick(closer.risk)
    assert tick.positions_seen == 0
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_handles_missing_snapshot(tmp_path: Path) -> None:
    """If get_option_snapshot returns None, fall back to avg_entry and skip."""
    monitor, journal, closer = _wire_monitor(tmp_path, snapshot=None)
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_minutes_to_close_basic() -> None:
    """Sanity-check the market_time helper (the canonical minutes_to_close)."""
    # 14:00 UTC = 10:00 ET (EDT) → 6 hours = 360 minutes to 16:00 ET
    now_10am = datetime(2026, 8, 12, 14, 0, 0, tzinfo=timezone.utc)
    assert minutes_to_close(now_10am) == 360
    # 19:30 UTC = 15:30 ET → 30 minutes to 16:00 ET
    now_330pm = datetime(2026, 8, 12, 19, 30, 0, tzinfo=timezone.utc)
    assert minutes_to_close(now_330pm) == 30
    # 21:00 UTC = 17:00 ET → already closed, returns -1
    now_5pm = datetime(2026, 8, 12, 21, 0, 0, tzinfo=timezone.utc)
    assert minutes_to_close(now_5pm) == -1


def test_market_time_handles_est_after_november() -> None:
    """November should fall back to EST (UTC-5). 14:00 UTC = 09:00 EST."""
    # First Sunday in November 2026 is Nov 1. After that, EST kicks in.
    # Nov 5, 2026 14:00 UTC = 09:00 EST → 7 hours to 16:00 = 420 minutes.
    nov5_14 = datetime(2026, 11, 5, 14, 0, 0, tzinfo=timezone.utc)
    assert minutes_to_close(nov5_14) == 420


def test_eod_trigger_fires_within_eod_window(tmp_path: Path) -> None:
    """15:55 ET (4:55pm) — within the 5-min EOD window → close."""
    monitor, journal, closer = _wire_monitor(tmp_path, snapshot={"bid": 2.50, "ask": 2.60})
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    # Patch minutes_to_close to return 3 (within window)
    import flip_options_bot.monitor.position_monitor as pm
    orig = pm.minutes_to_close
    pm.minutes_to_close = lambda *_a, **_kw: 3
    try:
        tick = monitor.tick(closer.risk)
    finally:
        pm.minutes_to_close = orig
    assert tick.closes_triggered == 1
    assert tick.reasons.get("eod") == 1


def test_eod_trigger_skips_on_weekend(tmp_path: Path) -> None:
    """Even if minutes_to_close returns 3, weekend should not trigger."""
    monitor, journal, closer = _wire_monitor(tmp_path, snapshot={"bid": 2.50, "ask": 2.60})
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    import flip_options_bot.monitor.position_monitor as pm
    orig_min = pm.minutes_to_close
    orig_is_weekday = pm.is_weekday
    pm.minutes_to_close = lambda *_a, **_kw: 3
    pm.is_weekday = lambda *_a, **_kw: False  # weekend
    try:
        tick = monitor.tick(closer.risk)
    finally:
        pm.minutes_to_close = orig_min
        pm.is_weekday = orig_is_weekday
    assert tick.closes_triggered == 0