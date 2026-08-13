from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from flip_options_bot.broker import BrokerClient
from flip_options_bot.config import Settings
from flip_options_bot.execution import Closer
from flip_options_bot.execution.executor import Executor
from flip_options_bot.journal import Journal, TradeEvent
from flip_options_bot.monitor.position_monitor import PositionMonitor
from flip_options_bot.risk import RiskEngine, RiskState
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner
from flip_options_bot.strategies.long_equity import LongEquitySignal, compute_conviction, make_filters_from_settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        run_dir=tmp_path,
        long_equity_enabled=True,
        long_equity_max_position_dollar=500.0,
        min_tp_profit_dollar=1.0,
    )


def _bars(start=100.0, step=0.05, n=60):
    out = []
    for i in range(n):
        c = start + i * step
        out.append({"o": c - 0.02, "h": c + 0.05, "l": c - 0.05, "c": c, "v": 1000})
    return out


def test_long_equity_conviction_requires_bullish_tape(tmp_path: Path):
    f = make_filters_from_settings(_settings(tmp_path))
    assert compute_conviction(0.002, 0.001, 0.002, f) >= f.min_conviction
    assert compute_conviction(-0.002, 0.001, 0.002, f) == 0.0
    assert compute_conviction(0.002, 0.001, -0.002, f) == 0.0


def test_scanner_emits_long_equity_when_calls_have_no_candidate(tmp_path: Path):
    settings = _settings(tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _bars()
    broker.get_stock_quote.return_value = {"bid": 102.90, "ask": 102.92}
    broker.list_option_contracts.return_value = []  # force no call candidates
    scanner = Scanner(settings, broker, FunnelRecorder(tmp_path))

    result = scanner.scan(["SPY"])

    assert len(result.candidates) == 1
    sig = result.candidates[0]
    assert sig.strategy_id == "long_equity"
    assert isinstance(sig, LongEquitySignal)
    assert sig.symbol == "SPY"
    assert sig.qty > 0
    assert sig.limit_price > 0
    assert sig.stop_price < sig.limit_price < sig.take_profit_price


def test_executor_submits_long_equity_limit_and_records_stop_risk(tmp_path: Path):
    settings = _settings(tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.submit_stock_buy.return_value = SimpleNamespace(
        id="stock-order-1", status="accepted", submitted_at="now"
    )
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = RiskState()
    executor = Executor(settings, broker, journal, risk)
    sig = LongEquitySignal(
        symbol="SPY",
        qty=4,
        limit_price=100.0,
        stop_price=99.60,
        take_profit_price=100.80,
        conviction=0.7,
    )

    result = executor.submit_long_equity(sig, equity=10_000.0, state=state)

    assert result.accepted
    broker.submit_stock_buy.assert_called_once()
    kwargs = broker.submit_stock_buy.call_args.kwargs
    assert kwargs["symbol"] == "SPY"
    assert kwargs["qty"] == 4
    assert kwargs["limit_price"] == 100.0
    pos = journal.get_position_for_id(result.position_id)
    assert pos is not None
    assert pos["strategy_id"] == "long_equity"
    assert float(pos["sl_trigger_price"]) == 99.60
    assert float(pos["tp_price"]) == 100.80
    assert risk.load_state().open_position_count == 1


def test_monitor_closes_long_equity_at_take_profit(tmp_path: Path):
    settings = _settings(tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_quote.return_value = {"bid": 100.82, "ask": 100.84}
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    closer = MagicMock(spec=Closer)
    closer.flatten_position.return_value = MagicMock(accepted=True)
    monitor = PositionMonitor(settings, broker, journal, risk, closer)
    pid = journal.new_position_id()
    journal.append(TradeEvent(
        event_id="long-equity-open",
        ts=datetime.now(timezone.utc).isoformat(),
        kind="open",
        symbol="SPY",
        side="buy",
        qty=4,
        price=100.0,
        position_id=pid,
        strategy_id="long_equity",
    ))
    journal.set_bracket(
        position_id=pid,
        tp_order_id=None,
        sl_order_id=None,
        tp_price=100.80,
        sl_trigger_price=99.60,
        sl_limit_price=99.50,
    )

    tick = monitor.tick(RiskState())

    assert tick.closes_triggered == 1
    assert tick.reasons == {"equity_tp": 1}
    closer.flatten_position.assert_called_once()
    assert closer.flatten_position.call_args.kwargs["symbol"] == "SPY"
