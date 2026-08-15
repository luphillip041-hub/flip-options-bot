"""Tests for the broker bracket-order construction.

These tests verify the bracket-order request shape WITHOUT actually
calling Alpaca. We patch `submit_order` so the trading client never hits
the wire, and we capture the LimitOrderRequest that was passed in.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from alpaca.trading.enums import OrderClass
from alpaca.trading.requests import LimitOrderRequest

from flip_options_bot.broker.alpaca import BrokerClient


class FakeTrading:
    def __init__(self) -> None:
        self.submitted: list = []

    def submit_order(self, req):
        self.submitted.append(req)
        # Fake the response shape with the fields we read downstream.
        return MagicMock(id="fake-order-uuid", status="accepted")


def _build_broker_with_fake_trading(settings) -> tuple[BrokerClient, FakeTrading]:
    fake = FakeTrading()
    broker = BrokerClient.__new__(BrokerClient)
    broker.settings = settings
    broker.trading = fake  # type: ignore[assignment]
    broker.data_options = MagicMock()
    broker.data_stocks = MagicMock()
    broker._account_cache = None
    broker._account_cache_ts = None
    return broker, fake


def _fake_settings() -> MagicMock:
    s = MagicMock()
    s.is_live.return_value = False
    s.alpaca_paper_key = "PAPER_KEY"
    s.alpaca_paper_secret = "PAPER_SECRET"
    s.alpaca_live_key = ""
    s.alpaca_live_secret = ""
    s.alpaca_live_base = ""
    s.alpaca_data_base = "https://data.alpaca.markets"
    s.phase = "paper"
    return s


def test_submit_bracket_buy_request_shape() -> None:
    """submit_bracket_buy must produce a LimitOrderRequest with:
    - order_class=BRACKET
    - take_profit.limit_price = tp_price
    - stop_loss.stop_price = sl_trigger_price
    - stop_loss.limit_price = sl_limit_price
    - limit_price = entry
    - client_order_id propagated
    """
    settings = _fake_settings()
    broker, fake = _build_broker_with_fake_trading(settings)

    broker.submit_bracket_buy(
        contract_symbol="SPY260815C00770000",
        qty=1,
        limit_price=2.50,
        tp_price=3.00,
        sl_trigger_price=1.80,
        sl_limit_price=1.75,
        client_order_id="open-test-uuid",
    )

    assert len(fake.submitted) == 1
    req = fake.submitted[0]
    assert isinstance(req, LimitOrderRequest)
    assert req.symbol == "SPY260815C00770000"
    assert req.qty == 1
    assert req.limit_price == 2.50
    assert req.client_order_id == "open-test-uuid"
    assert req.order_class == OrderClass.BRACKET
    # take_profit is a TakeProfitRequest object
    assert req.take_profit is not None
    assert float(req.take_profit.limit_price) == 3.00
    # stop_loss is a StopLossRequest object
    assert req.stop_loss is not None
    assert float(req.stop_loss.stop_price) == 1.80
    assert float(req.stop_loss.limit_price) == 1.75  # type: ignore[arg-type]


def test_submit_bracket_buy_rounds_prices() -> None:
    """Prices must be rounded to 2 decimal places before sending."""
    settings = _fake_settings()
    broker, fake = _build_broker_with_fake_trading(settings)

    broker.submit_bracket_buy(
        contract_symbol="SPY260815C00770000",
        qty=1,
        limit_price=2.5555,  # rounds to 2.56
        tp_price=3.1234,  # rounds to 3.12
        sl_trigger_price=1.8012,  # rounds to 1.80
        sl_limit_price=1.7508,  # rounds to 1.75
        client_order_id="open-round-test",
    )
    req = fake.submitted[0]
    assert req.limit_price == 2.56
    assert float(req.take_profit.limit_price) == 3.12
    assert float(req.stop_loss.stop_price) == 1.80
    assert float(req.stop_loss.limit_price) == 1.75  # type: ignore[arg-type]
