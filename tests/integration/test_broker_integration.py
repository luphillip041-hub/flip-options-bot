"""Integration tests for the broker layer.

These tests require live Alpaca paper creds. They are skipped if the
creds aren't available. Run with:

    pytest tests/integration/test_broker_integration.py -v

The tests are guarded by the `integration` pytest marker so `pytest tests/`
without `-m integration` runs only the unit tests.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from flip_options_bot.broker import BrokerClient
from flip_options_bot.config import Settings


def _load_paper_creds() -> tuple[str, str]:
    env_path = Path("/root/.config/flip-options-bot/.env")
    if not env_path.exists():
        pytest.skip("no .env at /root/.config/flip-options-bot/.env")
    creds = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            creds[k] = v
    key = creds.get("APCA_API_KEY_ID_PAPER")
    secret = creds.get("APCA_API_SECRET_KEY_PAPER")
    if not key or not secret:
        pytest.skip("paper creds missing in .env")
    return key, secret


pytestmark = pytest.mark.integration


@pytest.fixture
def settings() -> Settings:
    creds = _load_paper_creds()
    if creds is None:
        pytest.skip("no paper creds at /root/.config/flip-options-bot/.env")
    return Settings(
        phase="paper",
        live_trade_enabled=False,
        alpaca_paper_key=creds[0],
        alpaca_paper_secret=creds[1],
        run_dir=Path("/tmp/flip-options-bot-test"),
    )


@pytest.fixture
def broker(settings: Settings) -> BrokerClient:
    return BrokerClient(settings)


def test_account_is_paper_active(broker: BrokerClient):
    acct = broker.get_account()
    assert acct["status"].endswith("ACTIVE")
    assert acct["trading_blocked"] is False
    assert acct["options_trading_level"] >= 1


def test_get_stock_quote_returns_bid_ask(broker: BrokerClient):
    q = broker.get_stock_quote("SPY")
    assert q is not None
    assert q["bid"] > 0
    assert q["ask"] > 0
    assert q["ask"] >= q["bid"]


def test_get_stock_bars_minute_returns_data(broker: BrokerClient):
    bars = broker.get_stock_bars_minute("SPY", lookback_minutes=60)
    # May be 0 if market closed, but if open should have data.
    # Just assert the shape is right when there is data.
    if bars:
        for b in bars[:3]:
            assert "t" in b
            assert "c" in b
            assert b["c"] > 0


def test_list_option_contracts_returns_active(broker: BrokerClient):
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    expiry_gte = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    expiry_lte = (now + timedelta(days=14)).strftime("%Y-%m-%d")
    contracts = broker.list_option_contracts("SPY", expiry_gte, expiry_lte)
    assert len(contracts) > 0
    assert all("symbol" in c for c in contracts)
    assert all(c["type"] == "call" for c in contracts)


def test_account_cache_ttl(broker: BrokerClient):
    a1 = broker.get_account()
    a2 = broker.get_account()  # should hit cache
    assert a1 is a2
    a3 = broker.get_account(force_refresh=True)
    # Same content, but new dict
    assert a1 == a3
    assert a1 is not a3
