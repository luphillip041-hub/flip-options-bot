"""Tests for the close executor (Closer).

These tests use mocked broker + journal + risk so no live API calls
are made. They verify:
- flatten_position submits a SELL limit via the broker
- the journal receives a non-mutating close_attempt with position_id + reason
- flatten_all iterates all open positions
- idempotency: a duplicate event_id is silently skipped
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from flip_options_bot.execution.closer import Closer
from flip_options_bot.journal import Journal, TradeEvent


def _make_broker_mock() -> MagicMock:
    b = MagicMock()
    b.submit_close_sell.return_value = MagicMock(
        id="order-close-123",
        status="accepted",
        submitted_at="2026-08-12T15:00:00Z",
    )
    b.get_option_snapshot.return_value = {"bid": 1.0, "ask": 1.10}
    return b


def _make_risk_mock() -> MagicMock:
    r = MagicMock()
    return r


def _open_position(journal: Journal, symbol: str, qty: int, avg_entry: float) -> str:
    """Open a position so the journal has something for the closer to flatten."""
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


def test_flatten_position_submits_sell(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    broker = _make_broker_mock()
    risk = _make_risk_mock()
    closer = Closer.__new__(Closer)  # skip Settings init for unit test
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = risk

    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    res = closer.flatten_position(
        symbol="SPY260815C00770000",
        qty=1,
        position_id=pos_id,
        limit_price=2.75,
        reason="tp",
    )
    assert res.accepted, res.reason
    assert res.client_order_id.startswith("close-")
    broker.submit_close_sell.assert_called_once()
    assert journal.has_event(res.client_order_id)
    event = journal.get_event(res.client_order_id)
    assert event is not None
    assert event["kind"] == "close_attempt"
    # The close_attempt must not mark the position closed before broker fill.
    pos = journal.get_all_positions()[0]
    assert pos["position_id"] == pos_id
    assert pos["qty_closed"] == 0
    assert pos["state"] == "open"


def test_flatten_position_zero_qty_rejected(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    broker = _make_broker_mock()
    closer = Closer.__new__(Closer)
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = _make_risk_mock()

    res = closer.flatten_position(
        symbol="SPY260815C00770000",
        qty=0,
        position_id="abc",
        limit_price=2.0,
    )
    assert not res.accepted
    assert "zero_qty" in res.reason
    broker.submit_close_sell.assert_not_called()


def test_flatten_position_broker_error_returns_failure(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    broker = _make_broker_mock()
    broker.submit_close_sell.side_effect = RuntimeError("network down")
    closer = Closer.__new__(Closer)
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = _make_risk_mock()

    res = closer.flatten_position(
        symbol="SPY260815C00770000",
        qty=1,
        position_id="abc",
        limit_price=2.0,
    )
    assert not res.accepted
    assert "broker_error" in res.reason


def test_flatten_all_iterates_open_positions(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    broker = _make_broker_mock()
    risk = _make_risk_mock()
    closer = Closer.__new__(Closer)
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = risk

    _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    _open_position(journal, "QQQ260815C00770000", qty=2, avg_entry=3.00)

    n = closer.flatten_all(risk, reason="panic")
    assert n == 2, n
    assert broker.submit_close_sell.call_count == 2


def test_journal_idempotent_on_close_event_id(tmp_path: Path) -> None:
    """If reconcile_fills later writes the canonical close event with the
    same event_id, the journal's upsert() must overwrite price + realized_pnl
    with the broker's real fill, and recompute the position_state."""
    journal = Journal(tmp_path / "runs")
    broker = _make_broker_mock()
    closer = Closer.__new__(Closer)
    closer.settings = MagicMock()
    closer.broker = broker
    closer.journal = journal
    closer.risk = _make_risk_mock()

    pos_id = _open_position(journal, "SPY260815C00770000", qty=1, avg_entry=2.50)
    res = closer.flatten_position(
        symbol="SPY260815C00770000",
        qty=1,
        position_id=pos_id,
        limit_price=2.75,
    )
    # The close_attempt is in the table. Now simulate reconcile_fills writing
    # the canonical close via upsert() — this overwrites price + realized_pnl
    # with the real fill values from the broker and only then closes the row.
    canonical = TradeEvent(
        event_id=res.client_order_id,
        ts=Journal.now_iso(),
        kind="close",
        symbol="SPY260815C00770000",
        side="sell",
        qty=1,
        price=2.76,  # real fill, not the limit
        position_id=pos_id,
        realized_pnl=0.26,  # 2.76 - 2.50
        strategy_id="long_call",
    )
    created = journal.upsert(canonical)
    assert created is False  # close_attempt row already existed and was canonicalized

    pos = next(p for p in journal.get_all_positions() if p["position_id"] == pos_id)
    assert pos["state"] == "closed"
    assert pos["avg_exit_price"] == 2.76
    assert pos["realized_pnl"] == 0.26

    # Reading the canonical row from the trades table must show 2.76, not 2.75.
    import sqlite3
    with sqlite3.connect(journal.db_path) as conn:
        rows = conn.execute(
            "SELECT price, realized_pnl FROM trades WHERE event_id = ?",
            (res.client_order_id,),
        ).fetchall()
    assert rows == [(2.76, 0.26)]
