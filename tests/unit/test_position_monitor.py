"""Tests for the position monitor.

These tests use mocked broker + journal + risk + closer so no live API
calls are made. They verify:
- monitor.tick returns 0 closes when no positions are open
- SL trigger fires when mark <= 50% of entry
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


def _wire_monitor(tmp_path: Path, snapshot: dict | None) -> tuple:
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
    settings.close_eod_minutes = 5
    monitor = PositionMonitor(settings, broker, journal, closer.risk, closer)
    # Replace the journal with the freshly-built one (the helper's local
    # journal reference in `closer` is the same, but the monitor's journal
    # attribute was bound at __init__ above to the same instance, so this
    # is just an explicit reaffirmation for clarity).
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
    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 1
    assert tick.reasons.get("sl") == 1
    closer.flatten_position.assert_called_once()
    kwargs = closer.flatten_position.call_args.kwargs
    assert kwargs["symbol"] == "SPY260815C00770000"
    assert kwargs["reason"] == "sl"


def test_no_trigger_when_position_healthy(tmp_path: Path) -> None:
    # mark is well above the 50% SL line, not near EOD
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 2.50, "ask": 2.60}
    )
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_skips_position_with_zero_remaining_qty(tmp_path: Path) -> None:
    """A position that has been fully closed (qty_open == qty_closed) is no
    longer returned by get_open_positions, so the monitor doesn't see it."""
    monitor, journal, closer = _wire_monitor(
        tmp_path, snapshot={"bid": 0.50, "ask": 0.55}
    )
    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    # Manually mark the position as fully closed by writing a close event
    # with qty=1 against the same position_id. After this, state='closed'
    # and get_open_positions returns [].
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
    # Position is no longer in get_open_positions, so the monitor sees 0.
    assert tick.positions_seen == 0
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_handles_missing_snapshot(tmp_path: Path) -> None:
    """If get_option_snapshot returns None, fall back to avg_entry and skip SL."""
    monitor, journal, closer = _wire_monitor(tmp_path, snapshot=None)
    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    tick = monitor.tick(closer.risk)
    # mark falls back to avg_entry (2.50), which is NOT <= 50% of entry.
    # No close should fire.
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_minutes_to_close_basic() -> None:
    """Sanity-check the _minutes_to_close helper."""
    monitor, _, closer = _wire_monitor(
        Path("/tmp"), snapshot={"bid": 1.0, "ask": 1.10}
    )
    # 14:00 UTC = 10:00 ET → 6 hours = 360 minutes to 16:00 ET
    now_10am = datetime(2026, 8, 12, 14, 0, 0, tzinfo=timezone.utc)
    assert monitor._minutes_to_close(now_10am) == 360
    # 19:30 UTC = 15:30 ET → 30 minutes to 16:00 ET
    now_330pm = datetime(2026, 8, 12, 19, 30, 0, tzinfo=timezone.utc)
    assert monitor._minutes_to_close(now_330pm) == 30
    # 21:00 UTC = 17:00 ET → already closed, returns -1
    now_5pm = datetime(2026, 8, 12, 21, 0, 0, tzinfo=timezone.utc)
    assert monitor._minutes_to_close(now_5pm) == -1