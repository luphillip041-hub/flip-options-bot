"""Pre-trade evaluation: every gate active, end-to-end risk path.

Each test sets up a RiskEngine with a Settings that has all gates
tightened, then verifies that the expected gate blocks the trade.
"""

from pathlib import Path

import pytest

from flip_options_bot.config import Settings
from flip_options_bot.risk import RiskEngine


def _make_engine(tmp_path: Path, **overrides) -> tuple[RiskEngine, Settings]:
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        run_dir=tmp_path,
        per_trade_risk_pct=overrides.get("per_trade_risk_pct", 2.0),
        daily_loss_cap_pct=overrides.get("daily_loss_cap_pct", 6.0),
        weekly_loss_cap_pct=overrides.get("weekly_loss_cap_pct", 12.0),
        max_contract_dollar=overrides.get("max_contract_dollar", 500),
        max_positions=overrides.get("max_positions", 3),
    )
    return RiskEngine(settings, tmp_path), settings


def test_per_trade_risk_blocks_oversized_debit(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, per_trade_risk_pct=2.0, max_contract_dollar=5000)
    state = risk.load_state()
    # Equity $10k * 2% = $200 max debit; $2.50 option = $250/contract should be blocked.
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=2.50)
    assert decision.allowed is False
    assert "per_trade_risk" in decision.reason


def test_max_contract_dollar_blocks(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, per_trade_risk_pct=10.0, max_contract_dollar=500)
    state = risk.load_state()
    # 5.00 * 100 = $500, exactly at cap. Should pass.
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=5.00)
    assert decision.allowed is True
    # 5.01 * 100 = $501, over cap. Should fail.
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=5.01)
    assert decision.allowed is False
    assert "max_contract_dollar" in decision.reason


def test_max_positions_blocks(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, max_positions=2)
    state = risk.load_state()
    state.open_position_count = 2
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.0)
    assert decision.allowed is False
    assert "max_positions" in decision.reason


def test_daily_loss_cap_breach_trips_kill_switch(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, daily_loss_cap_pct=6.0)
    state = risk.load_state()
    state.daily_pnl = -700.0  # -7% on $10k > -6% cap
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.0)
    assert decision.allowed is False
    assert "daily_loss_cap" in decision.reason
    # Verify state was persisted as kill_switch=True
    state2 = risk.load_state()
    assert state2.kill_switch is True


def test_weekly_loss_cap_breach_trips_kill_switch(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, daily_loss_cap_pct=100.0, weekly_loss_cap_pct=12.0)
    state = risk.load_state()
    state.weekly_pnl = -1300.0  # -13% > -12%
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.0)
    assert decision.allowed is False
    assert "weekly_loss_cap" in decision.reason


def test_kill_switch_short_circuits_after_being_set(tmp_path: Path):
    risk, _ = _make_engine(tmp_path)
    state = risk.load_state()
    risk.force_kill(state, "manual test")
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.0)
    assert decision.allowed is False
    assert "kill_switch" in decision.reason
    assert "manual test" in decision.reason


def test_evaluate_caps_separate_from_pre_trade(tmp_path: Path):
    """evaluate_caps() returns True on breach; evaluate_pre_trade() never
    returns a True from caps (it returns False + reason)."""
    risk, _ = _make_engine(tmp_path, daily_loss_cap_pct=6.0)
    state = risk.load_state()
    state.daily_pnl = -700.0
    tripped = risk.evaluate_caps(state, equity=10_000.0)
    assert tripped is True
    assert state.kill_switch is True


def test_evaluate_caps_no_breach_returns_false(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, daily_loss_cap_pct=6.0)
    state = risk.load_state()
    state.daily_pnl = -100.0  # within cap
    tripped = risk.evaluate_caps(state, equity=10_000.0)
    assert tripped is False


def test_open_close_round_trip_updates_daily_pnl(tmp_path: Path):
    risk, _ = _make_engine(tmp_path)
    state = risk.load_state()
    initial_daily = state.daily_pnl

    risk.record_open(state, symbol="SPY260815C00770000", debit=200.0, event_id="open-1")
    assert state.open_position_count == 1
    assert state.realized_pnl_total == initial_daily  # not yet realized

    risk.record_close(state, pnl=85.50, event_id="close-1", payload="+85.50")
    assert state.open_position_count == 0
    assert state.daily_pnl == pytest.approx(initial_daily + 85.50)
    assert state.weekly_pnl == pytest.approx(85.50)
    assert state.realized_pnl_total == pytest.approx(initial_daily + 85.50)


def test_record_close_idempotent_on_event_id(tmp_path: Path):
    """Same event_id recorded twice = state unchanged the second time."""
    risk, _ = _make_engine(tmp_path)
    state = risk.load_state()
    risk.record_close(state, pnl=50.0, event_id="close-2")
    risk.record_close(state, pnl=50.0, event_id="close-2")  # second call
    # Both calls increment state. Wait — the structural fix should reject dupes.
    # Looking at record_close: it does check `existing` from DB. So the
    # second call should be a no-op.
    state2 = risk.load_state()
    # daily_pnl should reflect only ONE +50
    assert state2.daily_pnl == 50.0


def test_contracts_computed_from_max_contract_dollar(tmp_path: Path):
    risk, _ = _make_engine(
        tmp_path, max_contract_dollar=500, max_positions=3, per_trade_risk_pct=10.0
    )
    state = risk.load_state()
    # Proposed debit $1.50/contract → $150 contract risk; $500 cap allows 3 contracts.
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.50)
    assert decision.allowed is True
    assert decision.contracts == 3


def test_contracts_capped_at_max_positions_remaining(tmp_path: Path):
    risk, _ = _make_engine(tmp_path, max_contract_dollar=2000, max_positions=3)
    state = risk.load_state()
    state.open_position_count = 2  # only 1 slot remaining
    decision = risk.evaluate_pre_trade(state, equity=10_000.0, proposed_debit=1.0)
    # 2000 / (1.0*100) = 20 contracts → but only 1 slot left → 1 contract
    assert decision.contracts == 1
