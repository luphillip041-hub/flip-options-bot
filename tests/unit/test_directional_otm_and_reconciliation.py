from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from flip_options_bot.broker import BrokerClient
from flip_options_bot.config import Settings
from flip_options_bot.execution.executor import Executor
from flip_options_bot.journal import Journal
from flip_options_bot.risk import RiskEngine
from flip_options_bot.data.yfinance_options import YFinanceOptionQuote
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner
from flip_options_bot.strategies.long_call import LongCallSignal
from flip_options_bot.strategies.long_put import LongPutSignal


def _bars(start: float, step: float, n: int = 60):
    return [
        {"o": start + i * step, "h": start + i * step + 0.05, "l": start + i * step - 0.05, "c": start + i * step, "v": 1000}
        for i in range(n)
    ]


class FakeYFinanceProvider:
    def __init__(self, quote: YFinanceOptionQuote | None):
        self.quote = quote
        self.calls: list[tuple[str, str, str]] = []

    def get_quote(self, underlying: str, expiry: str, contract_symbol: str):
        self.calls.append((underlying, expiry, contract_symbol))
        return self.quote


class ExplodingYFinanceProvider:
    def get_quote(self, underlying: str, expiry: str, contract_symbol: str):
        raise AssertionError("yfinance should not be queried for 0DTE")


def test_long_call_prefers_configured_otm_strike(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=0,
        long_call_target_otm_pct=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_C_102", "expiry": expiry, "type": "call", "strike": 102.0, "open_interest": 100},
        {"symbol": "SPY_C_103", "expiry": expiry, "type": "call", "strike": 103.0, "open_interest": 100},
        {"symbol": "SPY_C_104", "expiry": expiry, "type": "call", "strike": 104.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {c["symbol"]: {"bid": 1.00, "ask": 1.06} for c in contracts}
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    # Spot 102, +1% OTM target is ~103.02; 103 should rank first.
    assert sig.strike == 103.0
    assert sig.dte == 0


def test_long_put_prefers_configured_otm_strike(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=True,
        long_equity_enabled=False,
        long_call_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=0,
        long_put_target_otm_pct=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(104.0, -0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_P_100", "expiry": expiry, "type": "put", "strike": 100.0, "open_interest": 100},
        {"symbol": "SPY_P_101", "expiry": expiry, "type": "put", "strike": 101.0, "open_interest": 100},
        {"symbol": "SPY_P_102", "expiry": expiry, "type": "put", "strike": 102.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {c["symbol"]: {"bid": 1.00, "ask": 1.06} for c in contracts}
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongPutSignal)
    # Spot 102, -1% OTM target is ~100.98; 101 should rank first.
    assert sig.strike == 101.0
    assert sig.dte == 0


def test_long_call_falls_forward_when_0dte_chain_is_too_wide(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        long_call_target_otm_pct=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    today = datetime.now(timezone.utc).date()
    exp0 = today.strftime("%Y-%m-%d")
    exp7 = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY0_C_103", "expiry": exp0, "type": "call", "strike": 103.0, "open_interest": 100},
        {"symbol": "SPY7_C_103", "expiry": exp7, "type": "call", "strike": 103.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {
        "SPY0_C_103": {"bid": 0.10, "ask": 0.30},  # median spread too wide => expiry skipped
        "SPY7_C_103": {"bid": 1.00, "ask": 1.06},
    }
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert sig.expiry == exp7
    assert sig.dte == 7


def test_long_put_falls_forward_when_0dte_chain_is_too_wide(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=True,
        long_equity_enabled=False,
        long_call_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        long_put_target_otm_pct=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(104.0, -0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    today = datetime.now(timezone.utc).date()
    exp0 = today.strftime("%Y-%m-%d")
    exp7 = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY0_P_101", "expiry": exp0, "type": "put", "strike": 101.0, "open_interest": 100},
        {"symbol": "SPY7_P_101", "expiry": exp7, "type": "put", "strike": 101.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {
        "SPY0_P_101": {"bid": 0.10, "ask": 0.30},
        "SPY7_P_101": {"bid": 1.00, "ask": 1.06},
    }
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongPutSignal)
    assert sig.expiry == exp7
    assert sig.dte == 7


def test_long_call_falls_forward_when_0dte_contract_is_over_cap(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        max_contract_dollar=500,
        long_call_target_otm_pct=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    today = datetime.now(timezone.utc).date()
    exp0 = today.strftime("%Y-%m-%d")
    exp7 = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY0_C_103", "expiry": exp0, "type": "call", "strike": 103.0, "open_interest": 100},
        {"symbol": "SPY7_C_103", "expiry": exp7, "type": "call", "strike": 103.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {
        "SPY0_C_103": {"bid": 6.00, "ask": 6.10},  # $607.50/contract, over cap
        "SPY7_C_103": {"bid": 1.00, "ask": 1.06},
    }
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert sig.expiry == exp7
    assert sig.limit_price == 1.04


def test_high_reward_mode_prefers_farther_otm_call(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        max_contract_dollar=500,
        long_option_high_reward_mode=True,
        long_option_otm_ladder_pct=(0.003, 0.006, 0.015),
        long_option_min_premium=0.15,
        long_option_max_spread_pct=0.35,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 99.99, "ask": 100.01}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_C_100_5", "expiry": expiry, "type": "call", "strike": 100.5, "open_interest": 100},
        {"symbol": "SPY_C_101_5", "expiry": expiry, "type": "call", "strike": 101.5, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {c["symbol"]: {"bid": 1.00, "ask": 1.06} for c in contracts}
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert sig.strike == 101.5
    assert "high_reward=True" in sig.notes


def test_high_reward_mode_rejects_dust_premium_call(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        max_contract_dollar=500,
        long_option_high_reward_mode=True,
        long_option_otm_ladder_pct=(0.003, 0.015),
        long_option_min_premium=0.15,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 99.99, "ask": 100.01}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_C_101_5", "expiry": expiry, "type": "call", "strike": 101.5, "open_interest": 100},
        {"symbol": "SPY_C_101_0", "expiry": expiry, "type": "call", "strike": 101.0, "open_interest": 100},
        {"symbol": "SPY_C_100_5", "expiry": expiry, "type": "call", "strike": 100.5, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {
        "SPY_C_101_5": {"bid": 0.04, "ask": 0.08},  # dust / untradeable
        "SPY_C_101_0": {"bid": 0.25, "ask": 0.29},
        "SPY_C_100_5": {"bid": 0.30, "ask": 0.34},
    }
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert sig.symbol == "SPY_C_101_0"


def test_high_reward_mode_prefers_farther_otm_put(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=True,
        long_equity_enabled=False,
        long_call_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        max_contract_dollar=500,
        long_option_high_reward_mode=True,
        long_option_otm_ladder_pct=(0.003, 0.006, 0.015),
        long_option_min_premium=0.15,
        long_option_max_spread_pct=0.35,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(104.0, -0.04)
    broker.get_stock_quote.return_value = {"bid": 99.99, "ask": 100.01}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_P_99_5", "expiry": expiry, "type": "put", "strike": 99.5, "open_interest": 100},
        {"symbol": "SPY_P_98_5", "expiry": expiry, "type": "put", "strike": 98.5, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {c["symbol"]: {"bid": 1.00, "ask": 1.06} for c in contracts}
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)

    result = Scanner(settings, broker, FunnelRecorder(tmp_path)).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongPutSignal)
    assert sig.strike == 98.5
    assert "high_reward=True" in sig.notes


def test_yfinance_1dte_bidask_confirmation_enriches_candidate(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=1,
        target_dte=1,
        max_dte=14,
        yfinance_confirm_1dte_enabled=True,
        yfinance_confirm_min_dte=1,
        yfinance_bidask_bonus=0.03,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = (datetime.now(timezone.utc).date() + timedelta(days=1)).strftime("%Y-%m-%d")
    contracts = [{"symbol": "SPY260815C00103500", "expiry": expiry, "type": "call", "strike": 103.5, "open_interest": 100}]
    broker.list_option_contracts.return_value = contracts
    broker.get_option_snapshot.return_value = {"bid": 1.00, "ask": 1.20}
    provider = FakeYFinanceProvider(YFinanceOptionQuote(
        contract_symbol="SPY260815C00103500",
        bid=1.05,
        ask=1.12,
        last_price=1.08,
        volume=250,
        open_interest=900,
        implied_volatility=0.42,
    ))

    result = Scanner(settings, broker, FunnelRecorder(tmp_path), yfinance_provider=provider).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert provider.calls == [("SPY", expiry, "SPY260815C00103500")]
    assert "yf_quote=bid_ask" in sig.notes
    assert "yf_confirm=bidask_tight" in sig.notes
    assert "yf_vol=250" in sig.notes
    assert sig.conviction > 0.45


def test_yfinance_confirmation_not_queried_for_0dte(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=0,
        yfinance_confirm_1dte_enabled=True,
        yfinance_confirm_min_dte=1,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    contracts = [{"symbol": "SPY260814C00103500", "expiry": expiry, "type": "call", "strike": 103.5, "open_interest": 100}]
    broker.list_option_contracts.return_value = contracts
    broker.get_option_snapshot.return_value = {"bid": 1.00, "ask": 1.06}

    result = Scanner(
        settings,
        broker,
        FunnelRecorder(tmp_path),
        yfinance_provider=ExplodingYFinanceProvider(),
    ).scan(["SPY"])

    assert result.candidates
    assert "yf_" not in result.candidates[0].notes


def test_yfinance_confirmation_applies_through_14dte_max(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=0,
        target_dte=0,
        max_dte=14,
        yfinance_confirm_1dte_enabled=True,
        yfinance_confirm_min_dte=1,
        yfinance_volume_bonus=0.01,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = (datetime.now(timezone.utc).date() + timedelta(days=14)).strftime("%Y-%m-%d")
    contracts = [{"symbol": "SPY260828C00103500", "expiry": expiry, "type": "call", "strike": 103.5, "open_interest": 100}]
    broker.list_option_contracts.return_value = contracts
    broker.get_option_snapshot.return_value = {"bid": 1.00, "ask": 1.06}
    provider = FakeYFinanceProvider(YFinanceOptionQuote(
        contract_symbol="SPY260828C00103500",
        last_price=1.03,
        last_trade_date=datetime.now(timezone.utc).isoformat(),
        volume=500,
    ))

    result = Scanner(settings, broker, FunnelRecorder(tmp_path), yfinance_provider=provider).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert isinstance(sig, LongCallSignal)
    assert sig.dte == 14
    assert provider.calls == [("SPY", expiry, "SPY260828C00103500")]
    assert "yf_confirm=volume_only" in sig.notes
    assert "yf_quote=last_price_proxy_non_executable" in sig.notes


def test_yfinance_stale_volume_does_not_boost_candidate(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=1,
        target_dte=1,
        max_dte=14,
        yfinance_confirm_1dte_enabled=True,
        yfinance_confirm_min_dte=1,
        yfinance_volume_bonus=0.03,
        yfinance_require_current_trade_date_for_volume_bonus=True,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = (datetime.now(timezone.utc).date() + timedelta(days=1)).strftime("%Y-%m-%d")
    contracts = [{"symbol": "SPY260815C00103500", "expiry": expiry, "type": "call", "strike": 103.5, "open_interest": 100}]
    broker.list_option_contracts.return_value = contracts
    broker.get_option_snapshot.return_value = {"bid": 1.00, "ask": 1.06}
    baseline = Scanner(settings, broker, FunnelRecorder(tmp_path / "base"), yfinance_provider=FakeYFinanceProvider(None)).scan(["SPY"]).candidates[0]
    provider = FakeYFinanceProvider(YFinanceOptionQuote(
        contract_symbol="SPY260815C00103500",
        last_price=1.03,
        last_trade_date="2026-08-13T16:57:05+00:00",
        volume=500,
    ))

    result = Scanner(settings, broker, FunnelRecorder(tmp_path), yfinance_provider=provider).scan(["SPY"])

    assert result.candidates
    sig = result.candidates[0]
    assert sig.conviction == baseline.conviction
    assert "yf_confirm=stale_volume_no_bonus" in sig.notes
    assert "yf_confirm=volume_only" not in sig.notes
    assert "yf_last_trade=2026-08-13" in sig.notes


def test_yfinance_strict_gate_rejects_wide_1dte_quote(tmp_path: Path):
    settings = Settings(
        run_dir=tmp_path,
        long_put_enabled=False,
        long_equity_enabled=False,
        min_dte=1,
        target_dte=1,
        max_dte=14,
        yfinance_confirm_1dte_enabled=True,
        yfinance_confirm_min_dte=1,
        yfinance_strict_gate=True,
        yfinance_max_spread_pct=0.35,
    )
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars(100.0, 0.04)
    broker.get_stock_quote.return_value = {"bid": 101.98, "ask": 102.02}
    expiry = (datetime.now(timezone.utc).date() + timedelta(days=1)).strftime("%Y-%m-%d")
    contracts = [{"symbol": "SPY260815C00103500", "expiry": expiry, "type": "call", "strike": 103.5, "open_interest": 100}]
    broker.list_option_contracts.return_value = contracts
    broker.get_option_snapshot.return_value = {"bid": 1.00, "ask": 1.06}
    provider = FakeYFinanceProvider(YFinanceOptionQuote(
        contract_symbol="SPY260815C00103500",
        bid=0.50,
        ask=1.50,
        volume=500,
    ))

    result = Scanner(settings, broker, FunnelRecorder(tmp_path), yfinance_provider=provider).scan(["SPY"])

    assert result.candidates == []


def test_directional_executor_blocks_same_underlying_stack(tmp_path: Path):
    settings = Settings(run_dir=tmp_path, max_contract_dollar=500, per_trade_risk_pct=10.0)
    broker = MagicMock(spec=BrokerClient)
    broker.submit_bracket_buy.side_effect = Exception("complex orders not supported")
    broker.submit_buy.return_value = SimpleNamespace(id="order-1", status="accepted", submitted_at="now", legs=[])
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    executor = Executor(settings, broker, journal, risk)

    first = LongCallSignal(symbol="SPY260813C00103000", expiry="2026-08-13", strike=103, limit_price=1.20, conviction=0.9, dte=0)
    second = LongPutSignal(symbol="SPY260813P00101000", expiry="2026-08-13", strike=101, limit_price=1.10, conviction=0.9, dte=0)
    assert executor.submit_long_call(first, equity=10_000, state=risk.load_state()).accepted
    denied = executor.submit_long_put(second, equity=10_000, state=risk.load_state())

    assert denied.accepted is False
    assert denied.reason == "duplicate_directional_underlying:SPY"


def test_reconcile_long_option_open_uses_actual_broker_fill(tmp_path: Path):
    settings = Settings(run_dir=tmp_path, max_contract_dollar=500, per_trade_risk_pct=10.0)
    broker = MagicMock(spec=BrokerClient)
    broker.submit_bracket_buy.side_effect = Exception("complex orders not supported")
    broker.submit_buy.return_value = SimpleNamespace(id="order-1", status="accepted", submitted_at="now", legs=[])
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    executor = Executor(settings, broker, journal, risk)

    sig = LongCallSignal(symbol="SPY260813C00103000", expiry="2026-08-13", strike=103, limit_price=1.20, conviction=0.9, dte=0)
    result = executor.submit_long_call(sig, equity=10_000, state=risk.load_state())
    assert result.accepted

    fill = MagicMock()
    fill.client_order_id = result.client_order_id
    fill.filled_avg_price = "1.17"
    fill.filled_qty = "1"
    fill.filled_at = "2026-08-13T15:00:00+00:00"
    fill.id = "order-1"
    fill.status = "FILLED"
    fill.side = "buy"
    fill.symbol = sig.symbol
    broker.list_filled_orders.return_value = [fill]

    assert executor.reconcile_fills() == 1
    event = journal.get_event(result.client_order_id)
    assert event is not None
    assert event["price"] == 1.17
    pos = journal.get_position_for_id(result.position_id)
    assert pos is not None
    assert pos["avg_entry_price"] == 1.17
    assert pos["qty_open"] == 1
