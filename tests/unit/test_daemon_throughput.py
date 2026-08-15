from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from flip_options_bot import daemon
from flip_options_bot.config import Settings
from flip_options_bot.execution.executor import ExecutionResult
from flip_options_bot.risk import RiskState
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import ScanResult
from flip_options_bot.strategies.long_call import LongCallSignal
from flip_options_bot.strategies.long_put import LongPutSignal


class FakeBroker:
    def get_account(self):
        return {"equity": 10_000.0}

    def list_open_orders_or_raise(self):
        return []


class FakeRisk:
    def load_state(self):
        return RiskState()

    def tick_rollover(self, state):
        return state


class FakeExecutor:
    def __init__(self):
        self.submitted: list[str] = []

    def reconcile_fills(self):
        return 0

    def submit_long_call(self, sig, equity, state):
        self.submitted.append(sig.symbol)
        return ExecutionResult(accepted=True)

    def submit_long_put(self, sig, equity, state):
        self.submitted.append(sig.symbol)
        return ExecutionResult(accepted=True)

    def submit_long_equity(self, sig, equity, state):
        self.submitted.append(sig.symbol)
        return ExecutionResult(accepted=True)


class FakeScanner:
    def __init__(self, tmp_path: Path, candidates):
        self.tmp_path = tmp_path
        self.candidates = candidates

    def scan(self, watchlist):
        row = FunnelRecorder.new_row(watchlist_count=len(watchlist))
        row.raw_signal_count = len(self.candidates)
        row.sized_count = len(self.candidates)
        row.dominant_skip_reason = "ok"
        return ScanResult(funnel_row=row, candidates=self.candidates)


class FakeFunnel:
    def emit_skip(self, reason: str):
        self.reason = reason


class FakeMonitor:
    pass


def _call(symbol: str, conviction: float, dte: int = 0) -> LongCallSignal:
    return LongCallSignal(
        symbol=symbol,
        expiry="2026-08-14",
        strike=100,
        limit_price=1.0,
        conviction=conviction,
        dte=dte,
    )


def _put(symbol: str, conviction: float, dte: int = 0) -> LongPutSignal:
    return LongPutSignal(
        symbol=symbol,
        expiry="2026-08-14",
        strike=100,
        limit_price=1.0,
        conviction=conviction,
        dte=dte,
    )


def test_run_once_submits_best_candidates_before_watchlist_order(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "is_market_open", lambda: True)
    monkeypatch.setattr(daemon, "is_entry_window", lambda: True)
    settings = Settings(
        run_dir=tmp_path,
        max_positions=10,
        max_submissions_per_scan=2,
        long_option_high_reward_mode=True,
    )
    candidates = [
        _call("LOW_C", 0.46),
        _put("BEST_P", 0.92),
        _call("SECOND_C", 0.80),
    ]
    executor = FakeExecutor()

    status = daemon.run_once(
        settings=settings,
        broker=FakeBroker(),
        journal=object(),
        risk=FakeRisk(),
        funnel=FakeFunnel(),
        scanner=FakeScanner(tmp_path, candidates),
        executor=executor,
        monitor=FakeMonitor(),
        watchlist=["LOW", "BEST", "SECOND"],
    )

    assert executor.submitted == ["BEST_P", "SECOND_C"]
    assert status["submitted_count"] == 2
    assert status["ranked_candidate_count"] == 3
    assert status["max_submissions_per_scan"] == 2
    assert "scan_submission_cap:2/2" in status["denied"]


def test_rank_directional_candidates_prefers_target_dte_on_tie(tmp_path):
    settings = Settings(run_dir=tmp_path, target_dte=0)
    farther = _call("FARTHER_DTE", 0.70, dte=7)
    zero = _put("ZERO_DTE", 0.70, dte=0)

    ranked = daemon._rank_directional_candidates([farther, zero], settings)

    assert [s.symbol for s in ranked] == ["ZERO_DTE", "FARTHER_DTE"]


def test_stale_pending_cancel_close_holds_new_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "is_market_open", lambda: True)
    monkeypatch.setattr(daemon, "is_entry_window", lambda: True)

    class BrokerWithStaleClose(FakeBroker):
        def list_open_orders_or_raise(self):
            return [
                SimpleNamespace(
                    client_order_id="close-stale",
                    symbol="XLK260821C00192000",
                    side="SELL",
                    status="PENDING_CANCEL",
                    submitted_at=datetime.now(UTC) - timedelta(hours=18),
                )
            ]

    executor = FakeExecutor()

    status = daemon.run_once(
        settings=Settings(run_dir=tmp_path, max_positions=10, max_submissions_per_scan=2),
        broker=BrokerWithStaleClose(),
        journal=object(),
        risk=FakeRisk(),
        funnel=FakeFunnel(),
        scanner=FakeScanner(tmp_path, [_call("SPY260817C00650000", 0.90)]),
        executor=executor,
        monitor=FakeMonitor(),
        watchlist=["SPY"],
    )

    assert executor.submitted == []
    assert status["submitted_count"] == 0
    assert status["scan_id"] == "stale-close-hold"
    assert status["stale_close_entry_hold"] is True
    assert status["dominant_skip_reason"] == "unresolved_stale_close_orders"
