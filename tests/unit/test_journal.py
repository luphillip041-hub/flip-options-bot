"""Tests for the journal — append-only, idempotent on event_id."""

from __future__ import annotations

from pathlib import Path

import pytest

from flip_options_bot.journal import Journal, TradeEvent


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path)


def _open(journal: Journal, symbol: str = "SPY260815C00770000", qty: int = 1, price: float = 2.50):
    pid = Journal.new_position_id()
    return TradeEvent(
        event_id=f"open-{pid}",
        ts=Journal.now_iso(),
        kind="open",
        symbol=symbol,
        side="buy",
        qty=qty,
        price=price,
        position_id=pid,
        strategy_id="long_call",
    )


def _close(journal: Journal, position_id: str, symbol: str = "SPY260815C00770000", qty: int = 1, price: float = 2.55, pnl: float = 0.05):
    return TradeEvent(
        event_id=f"close-{position_id}",
        ts=Journal.now_iso(),
        kind="close",
        symbol=symbol,
        side="sell",
        qty=qty,
        price=price,
        position_id=position_id,
        realized_pnl=pnl,
        strategy_id="long_call",
    )


def test_open_then_close_writes_both_events(journal: Journal):
    pid = Journal.new_position_id()
    open_ev = TradeEvent(
        event_id=f"open-{pid}",
        ts=Journal.now_iso(),
        kind="open",
        symbol="SPY260815C00770000",
        side="buy",
        qty=1,
        price=2.50,
        position_id=pid,
    )
    assert journal.append(open_ev) is True

    close_ev = TradeEvent(
        event_id=f"close-{pid}",
        ts=Journal.now_iso(),
        kind="close",
        symbol="SPY260815C00770000",
        side="sell",
        qty=1,
        price=2.55,
        position_id=pid,
        realized_pnl=0.05,
    )
    assert journal.append(close_ev) is True


def test_duplicate_event_id_is_idempotent(journal: Journal):
    """The structural fix: duplicate event_id returns False and is silently ignored.

    Reproduces the 'duplicate close events' artifact that produced the -$43k
    phantom in flip-alpaca-bot.
    """
    pid = Journal.new_position_id()
    open_ev = _open(journal)
    open_ev.position_id = pid

    assert journal.append(open_ev) is True
    assert journal.append(open_ev) is False  # same event_id

    assert journal.has_event(open_ev.event_id) is True

    # Total realized P&L is still 0 (no close yet, no double-count)
    assert journal.total_realized_pnl() == 0.0


def test_total_realized_pnl_only_counts_closes(journal: Journal):
    pid = Journal.new_position_id()
    journal.append(_open(journal, qty=1, price=2.50))
    # Construct a close with explicit position_id
    close_ev = _close(journal, pid, pnl=0.05)
    close_ev.position_id = pid
    journal.append(close_ev)
    assert journal.total_realized_pnl() == pytest.approx(0.05)


def test_multiple_positions_same_symbol_are_distinguished(journal: Journal):
    """Same (symbol, date) tuple can have multiple positions, distinguished
    by position_id UUID. This is the structural fix for the 'invalid OCC
    symbol' artifact class which collapsed multiple real positions into one."""
    open1 = _open(journal)
    open2 = _open(journal)
    assert open1.position_id != open2.position_id
    assert journal.append(open1)
    assert journal.append(open2)

    positions = journal.get_all_positions()
    assert len(positions) == 2


def test_get_open_positions_includes_partial(journal: Journal):
    pid = Journal.new_position_id()
    open_ev = _open(journal, qty=2, price=2.50)
    open_ev.position_id = pid
    journal.append(open_ev)
    positions = journal.get_all_positions()
    assert len(positions) == 1
    assert positions[0]["state"] == "open"


def test_close_transitions_position_to_closed(journal: Journal):
    pid = Journal.new_position_id()
    open_ev = _open(journal, qty=1, price=2.50)
    open_ev.position_id = pid
    journal.append(open_ev)

    close_ev = _close(journal, pid, qty=1, price=2.55, pnl=0.05)
    close_ev.position_id = pid
    journal.append(close_ev)

    positions = journal.get_all_positions()
    assert len(positions) == 1
    assert positions[0]["state"] == "closed"
    assert positions[0]["closed_at"] != ""