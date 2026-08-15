"""Tests for the risk engine — the critical structural-fix correctness checks.

These tests target the bug classes called out in paper-to-live-trading-bot-scaffold:
- tick_rollover must NOT zero daily_pnl on a fresh state
- _check_loss_caps must run BEFORE the kill_switch short-circuit
- record_close must be idempotent
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flip_options_bot.config import Settings
from flip_options_bot.risk import RiskEngine, RiskState


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        phase="paper",
        live_trade_enabled=False,
        equity_start=10_000.0,
        max_positions=3,
        per_trade_risk_pct=2.0,
        daily_loss_cap_pct=6.0,
        weekly_loss_cap_pct=12.0,
        max_contract_dollar=500,
        run_dir=tmp_path,
    )


@pytest.fixture
def engine(settings: Settings, tmp_path: Path) -> RiskEngine:
    return RiskEngine(settings, tmp_path)


def test_tick_rollover_does_not_zero_fresh_state(engine: RiskEngine):
    """Bug-class fix: fresh state (last_reset_day empty) must NOT zero daily_pnl.

    Reproduces the rollover bug from flip-alpaca-bot. If this test fails, a
    brand-new bot will silently nuke its daily P&L on the first tick.
    """
    state = RiskState(daily_pnl=-65.0)  # fresh state, prior day string empty
    state = engine.tick_rollover(state)
    assert state.daily_pnl == -65.0  # NOT zeroed
    assert state.last_reset_day != ""  # initialized


def test_tick_rollover_zeros_on_real_day_change(engine: RiskEngine):
    state = RiskState(
        daily_pnl=-65.0,
        last_reset_day="2020-01-01",  # prior day, will change
    )
    state = engine.tick_rollover(state)
    assert state.daily_pnl == 0.0
    assert state.last_reset_day != "2020-01-01"


def test_loss_cap_fires_before_kill_switch(engine: RiskEngine):
    """A fresh cap breach must escalate to KILL and block the trade.

    Reproduces the bug class where _check_loss_caps runs AFTER kill_switch
    short-circuit, causing silent cap-passes.
    """
    state = RiskState(daily_pnl=-700.0)  # 7% loss on $10k = over the 6% daily cap
    decision = engine.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=200.0)
    assert decision.allowed is False
    assert "daily_loss_cap_breach" in decision.reason
    # State must have been escalated to KILL
    reloaded = engine.load_state()
    assert reloaded.kill_switch is True
    assert "daily_loss_cap_breach" in reloaded.kill_reason


def test_kill_switch_blocks_subsequent_trades(engine: RiskEngine):
    state = engine.force_kill(RiskState(), "manual kill")
    decision = engine.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=200.0)
    assert decision.allowed is False
    assert "kill_switch" in decision.reason


def test_record_close_is_idempotent(engine: RiskEngine):
    """Structural fix: the same event_id must NOT produce two P&L impacts.

    Reproduces the duplicate close event artifact that produced the -$43k
    phantom in flip-alpaca-bot.
    """
    state = RiskState(daily_pnl=0.0, open_position_count=1)
    event_id = "close-abc-123"

    state = engine.record_close(state, pnl=-50.0, event_id=event_id)
    assert state.daily_pnl == -50.0
    assert state.open_position_count == 0

    # Replay the same event_id
    state = engine.record_close(state, pnl=-50.0, event_id=event_id)
    assert state.daily_pnl == -50.0  # NOT double-counted
    assert state.open_position_count == 0


def test_record_open_then_close_balances(engine: RiskEngine):
    state = RiskState()
    state = engine.record_open(state, symbol="SPY260815C00770000", debit=2.50, event_id="open-1")
    assert state.open_position_count == 1

    state = engine.record_close(state, pnl=0.50, event_id="close-1", payload="SPY260815C00770000")
    assert state.open_position_count == 0
    assert state.daily_pnl == 0.50
    assert state.realized_pnl_total == 0.50


def test_max_positions_enforced(engine: RiskEngine):
    # proposed_debit=2.0 → per_contract = 2.0*100 = $200 < $500 cap; passes that check.
    # open_position_count=3 hits the max_positions gate.
    state = RiskState(open_position_count=3)
    decision = engine.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=2.0)
    assert decision.allowed is False
    assert "max_positions" in decision.reason


def test_per_trade_risk_enforced(tmp_path: Path):
    """$2.50 option = $250/contract on $10k equity = 2.5% > 2% cap."""
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        equity_start=10_000.0,
        max_positions=3,
        per_trade_risk_pct=2.0,
        daily_loss_cap_pct=6.0,
        weekly_loss_cap_pct=12.0,
        max_contract_dollar=5000,
        run_dir=tmp_path,
    )
    engine = RiskEngine(settings, tmp_path)
    state = RiskState()
    decision = engine.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=2.50)
    assert decision.allowed is False
    assert "per_trade_risk" in decision.reason


def test_max_contract_dollar_enforced(engine: RiskEngine):
    """per_contract debit * 100 > max_contract_dollar."""
    state = RiskState()
    decision = engine.evaluate_pre_trade(
        state, equity=10_000.0, proposed_debit=6.0
    )  # 6*100 = 600 > 500
    assert decision.allowed is False
    assert "max_contract_dollar" in decision.reason


def test_event_log_persists(engine: RiskEngine, tmp_path: Path):
    """Verify the risk_events table is queryable and INSERT OR IGNORE works."""
    engine.record_event("ev-1", "open", pnl=-2.50, payload="SPY260815C00770000")
    engine.record_event("ev-1", "open", pnl=-2.50, payload="SPY260815C00770000")  # dup
    with sqlite3.connect(engine.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM risk_events WHERE event_id = ?", ("ev-1",)
        ).fetchone()[0]
    assert count == 1


def test_release_kill(engine: RiskEngine):
    state = engine.force_kill(RiskState(), "manual")
    state = engine.release_kill(state)
    # proposed_debit=2.0 fits under max_contract_dollar cap so it gets past that gate.
    decision = engine.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=2.0)
    assert decision.allowed is True
    assert decision.contracts >= 1
