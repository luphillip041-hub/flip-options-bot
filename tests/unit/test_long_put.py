from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from flip_options_bot.broker import BrokerClient
from flip_options_bot.config import Settings
from flip_options_bot.execution.executor import Executor
from flip_options_bot.journal import Journal
from flip_options_bot.risk import RiskEngine, RiskState
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner
from flip_options_bot.strategies.long_put import (
    LongPutSignal,
    compute_conviction,
    make_filters_from_settings,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        run_dir=tmp_path,
        long_put_enabled=True,
        long_equity_enabled=False,
        min_tp_profit_dollar=1.0,
    )


def _down_bars(start=100.0, step=-0.05, n=60):
    out = []
    base = datetime.now(UTC) - timedelta(minutes=n - 1)
    for i in range(n):
        c = start + i * step
        out.append({
            "t": (base + timedelta(minutes=i)).isoformat(),
            "o": c + 0.02,
            "h": c + 0.05,
            "l": c - 0.05,
            "c": c,
            "v": 1000,
        })
    return out


def test_long_put_conviction_requires_confirmed_bearish_tape(tmp_path: Path):
    f = make_filters_from_settings(_settings(tmp_path))
    assert compute_conviction(-0.002, 0.001, -0.002, 0.05, f) >= f.min_conviction
    assert compute_conviction(0.002, 0.001, -0.002, 0.05, f) == 0.0
    assert compute_conviction(-0.002, 0.001, 0.002, 0.05, f) == 0.0


def test_scanner_emits_long_put_on_downtrend(tmp_path: Path):
    settings = _settings(tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.get_stock_bars_minute.return_value = _down_bars()
    broker.get_stock_quote.return_value = {"bid": 97.03, "ask": 97.05}
    expiry = (datetime.now(UTC) + timedelta(days=settings.target_dte)).strftime("%Y-%m-%d")
    contracts = [
        {"symbol": "SPY_PUT_97", "expiry": expiry, "type": "put", "strike": 97.0, "open_interest": 100},
        {"symbol": "SPY_PUT_98", "expiry": expiry, "type": "put", "strike": 98.0, "open_interest": 100},
    ]
    broker.list_option_contracts.return_value = contracts
    snapshots = {
        "SPY_PUT_97": {"bid": 1.00, "ask": 1.06},
        "SPY_PUT_98": {"bid": 1.45, "ask": 1.53},
    }
    broker.get_option_snapshot.side_effect = lambda sym, expiry=None: snapshots.get(sym)
    scanner = Scanner(settings, broker, FunnelRecorder(tmp_path))

    result = scanner.scan(["SPY"])

    assert len(result.candidates) >= 1
    assert all(c.strategy_id == "long_put" for c in result.candidates)
    sig = result.candidates[0]
    assert isinstance(sig, LongPutSignal)
    assert sig.option_type == "put"
    assert sig.limit_price > 0
    assert sig.conviction >= settings.long_put_min_conviction


def test_executor_submits_long_put_limit_and_records_position(tmp_path: Path):
    settings = _settings(tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.submit_bracket_buy.side_effect = Exception("complex orders not supported")
    broker.submit_buy.return_value = SimpleNamespace(
        id="put-order-1", status="accepted", submitted_at="now", legs=[]
    )
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = RiskState()
    executor = Executor(settings, broker, journal, risk)
    sig = LongPutSignal(
        symbol="SPY260815P00750000",
        expiry="2026-08-15",
        strike=750.0,
        limit_price=1.25,
        conviction=0.8,
        dte=2,
    )

    result = executor.submit_long_put(sig, equity=10_000.0, state=state)

    assert result.accepted
    broker.submit_buy.assert_called_once()
    kwargs = broker.submit_buy.call_args.kwargs
    assert kwargs["contract_symbol"] == "SPY260815P00750000"
    assert kwargs["limit_price"] == 1.25
    pos = journal.get_position_for_id(result.position_id)
    assert pos is not None
    assert pos["strategy_id"] == "long_put"
    assert float(pos["tp_price"]) == round(1.25 * settings.tp_multiplier, 2)
    assert float(pos["sl_trigger_price"]) == round(1.25 * settings.sl_threshold_pct, 2)
    assert risk.load_state().open_position_count == 1
