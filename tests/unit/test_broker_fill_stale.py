"""Tests for the broker's list_filled_orders and the executor's
cancel_stale_orders — the two paths that the old flip-alpaca-bot got wrong.

We mock alpaca-py entirely; no live calls.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flip_options_bot.broker import BrokerClient
from flip_options_bot.execution import Executor
from flip_options_bot.journal import Journal, TradeEvent
from flip_options_bot.risk import RiskEngine


def _make_broker() -> BrokerClient:
    settings = MagicMock()
    settings.is_live.return_value = False
    settings.alpaca_paper_key = "PK_FAKE"
    settings.alpaca_paper_secret = "SK_FAKE"
    settings.alpaca_paper_base = "https://paper-api.alpaca.markets"
    settings.alpaca_live_key = ""
    settings.alpaca_live_secret = ""
    settings.alpaca_live_base = "https://api.alpaca.markets"
    return BrokerClient.__new__(BrokerClient)


def test_list_filled_orders_filters_correctly(tmp_path: Path):
    broker = _make_broker()

    # Mock the underlying trading client
    from alpaca.trading.enums import OrderStatus, AssetClass
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    old_submitted = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new_submitted = now + timedelta(seconds=10)  # must be > since_ts

    def make_order(sym, status, asset_class, submitted):
        o = MagicMock()
        o.symbol = sym
        o.status = status
        o.asset_class = asset_class
        o.submitted_at = submitted
        o.id = f"id-{sym}-{submitted.timestamp()}"
        return o

    fake_orders = [
        make_order("SPY260815C00750000", OrderStatus.FILLED, AssetClass.US_OPTION, new_submitted),
        make_order("SPY260815C00751000", OrderStatus.CANCELED, AssetClass.US_OPTION, new_submitted),  # not filled
        make_order("SPY", OrderStatus.FILLED, AssetClass.US_EQUITY, new_submitted),  # filled long-equity fallback
        make_order("SPY260815C00752000", OrderStatus.FILLED, AssetClass.US_OPTION, old_submitted),  # too old
    ]
    broker.trading = MagicMock()
    broker.trading.get_orders = MagicMock(return_value=fake_orders)

    # since_ts = 1 day ago → should accept new_submitted orders
    since = now - timedelta(days=1)
    filled = broker.list_filled_orders(since_ts=since)
    # Should return the new FILLED option + filled equity fallback order
    assert len(filled) == 2
    assert {o.symbol for o in filled} == {"SPY260815C00750000", "SPY"}


def test_list_filled_orders_handles_after_typeerror(tmp_path: Path):
    """If `after=datetime` raises TypeError (known SDK bug), fall back
    to client-side cutoff."""
    broker = _make_broker()
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus, AssetClass

    # First call (with after=) raises TypeError. Second call (without) succeeds.
    cutoff = datetime(2026, 8, 1, tzinfo=timezone.utc)
    new_submitted = datetime(2026, 8, 12, tzinfo=timezone.utc)

    def make_order(sym, status, submitted):
        o = MagicMock()
        o.symbol = sym
        o.status = status
        o.asset_class = AssetClass.US_OPTION
        o.submitted_at = submitted
        o.id = f"id-{sym}"
        return o

    fallback_orders = [
        make_order("FILLED_NEW", OrderStatus.FILLED, new_submitted),
        make_order("FILLED_OLD", OrderStatus.FILLED, datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ]

    def get_orders(req):
        # First call has `after=` param — raise TypeError
        if req.after is not None:
            raise TypeError("'>' not supported between 'str' and 'int'")
        return fallback_orders

    broker.trading = MagicMock()
    broker.trading.get_orders = MagicMock(side_effect=get_orders)

    filled = broker.list_filled_orders(since_ts=cutoff)
    # Should return 1 (FILLED_NEW). FILLED_OLD filtered by client-side cutoff.
    assert len(filled) == 1
    assert filled[0].symbol == "FILLED_NEW"


def test_cancel_stale_orders_cancels_old_only(tmp_path: Path):
    """An order ACCEPTED for >120s gets cancelled; a fresh one doesn't."""
    broker = _make_broker()
    journal = Journal(tmp_path)
    risk = RiskEngine.__new__(RiskEngine)
    risk.db_path = tmp_path / "state.db"

    executor = Executor.__new__(Executor)
    executor.broker = broker
    executor.journal = journal

    from alpaca.trading.enums import OrderStatus, AssetClass
    now = datetime.now(timezone.utc)
    old = datetime(2026, 8, 11, 22, 0, 0, tzinfo=timezone.utc)
    fresh = now

    def make_order(coid, status, submitted):
        o = MagicMock()
        o.id = f"id-{coid}"
        o.symbol = "SPY260815C00750000"
        o.status = status
        o.client_order_id = coid
        o.submitted_at = submitted
        o.asset_class = AssetClass.US_OPTION
        return o

    broker.trading = MagicMock()
    broker.list_open_orders = MagicMock(return_value=[
        make_order("STALE-1", OrderStatus.ACCEPTED, old),  # 3+ hours old
        make_order("FRESH-1", OrderStatus.ACCEPTED, fresh),  # <120s
    ])
    broker.cancel_order = MagicMock()
    journal.has_event = MagicMock(return_value=False)

    n = executor.cancel_stale_orders(older_than_seconds=120)
    assert n == 1
    # Verify only STALE-1 was cancelled
    cancelled_ids = [call.args[0] for call in broker.cancel_order.call_args_list]
    assert "id-STALE-1" in cancelled_ids
    assert "id-FRESH-1" not in cancelled_ids


def test_cancel_stale_orders_skips_journaled(tmp_path: Path):
    """Don't cancel orders that are already journaled as 'open'."""
    broker = _make_broker()
    journal = Journal(tmp_path)
    risk = RiskEngine.__new__(RiskEngine)

    executor = Executor.__new__(Executor)
    executor.broker = broker
    executor.journal = journal

    from alpaca.trading.enums import OrderStatus
    old = datetime(2026, 8, 11, 22, 0, 0, tzinfo=timezone.utc)

    order = MagicMock()
    order.id = "id-open-123"
    order.symbol = "SPY260815C00750000"
    order.status = OrderStatus.ACCEPTED
    order.client_order_id = "open-123"
    order.submitted_at = old

    broker.list_open_orders = MagicMock(return_value=[order])
    broker.cancel_order = MagicMock()
    journal.append(TradeEvent(
        event_id="open-123",
        ts="2026-08-11T21:59:00+00:00",
        kind="open",
        symbol="SPY260815C00750000",
        side="buy",
        qty=1,
        price=1.0,
        position_id=Journal.new_position_id(),
        strategy_id="long_call",
    ))

    n = executor.cancel_stale_orders(older_than_seconds=120)
    assert n == 0
    broker.cancel_order.assert_not_called()


def test_cancel_stale_orders_cancels_stale_close_attempt(tmp_path: Path):
    """A stale close_attempt is an unfilled exit, so it must be cancelable/repriceable."""
    broker = _make_broker()
    journal = Journal(tmp_path)

    executor = Executor.__new__(Executor)
    executor.broker = broker
    executor.journal = journal

    from alpaca.trading.enums import OrderStatus, OrderSide
    old = datetime(2026, 8, 11, 22, 0, 0, tzinfo=timezone.utc)
    coid = "close-123"

    order = MagicMock()
    order.id = "id-close-123"
    order.symbol = "SPY260815C00750000"
    order.status = OrderStatus.ACCEPTED
    order.side = OrderSide.SELL
    order.client_order_id = coid
    order.submitted_at = old

    journal.append(TradeEvent(
        event_id=coid,
        ts="2026-08-11T21:59:00+00:00",
        kind="close_attempt",
        symbol="SPY260815C00750000",
        side="sell",
        qty=1,
        price=0.9,
        position_id=Journal.new_position_id(),
        raw_broker_fill={"close_position_id": "pos-1"},
    ))
    broker.list_open_orders = MagicMock(return_value=[order])
    broker.cancel_order = MagicMock()

    n = executor.cancel_stale_orders(older_than_seconds=120)
    assert n == 1
    broker.cancel_order.assert_called_once_with("id-close-123")