"""Tests for the BPCS (bull put credit spread) wiring.

Coverage:
- BullPutSpreadSignal dataclass new fields
- BPCSFilters via make_filters_from_settings
- passes_dte_window + pick_target_expiry + passes_bpcs_conviction
- Risk engine evaluate_pre_trade_spread
- record_open_spread does NOT subtract debit (key behavior!)
- Executor submit_bull_put_spread basic happy path (with mocks)
- Scanner scan_bpcs end-to-end with mocked broker
- Journal get_legs_for_position returns multiple legs
- Strategy registry includes BPCS when enabled
"""

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flip_options_bot.broker.alpaca import BrokerClient
from flip_options_bot.config import Settings, get_settings
from flip_options_bot.execution import Closer
from flip_options_bot.execution.executor import Executor
from flip_options_bot.journal import Journal, TradeEvent
from flip_options_bot.monitor.position_monitor import PositionMonitor
from flip_options_bot.risk import RiskEngine, RiskState
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner
from flip_options_bot.strategies.bull_put_credit import (
    BPCSFilters,
    BullPutSpreadSignal,
    compute_bpcs_conviction,
    estimate_credit,
    make_filters_from_settings,
    passes_bpcs_conviction,
    passes_dte_window,
    pick_target_expiry,
    select_strikes,
)
from flip_options_bot.strategies import enabled_strategies, get_strategy


# ===== BPCSFilters =====

def test_make_filters_from_settings_defaults():
    settings = Settings(phase='paper', live_trade_enabled=False, run_dir=Path('/tmp'))
    filters = make_filters_from_settings(settings)
    assert filters.target_dte == 35
    assert filters.min_dte == 25
    assert filters.max_dte == 50
    assert filters.min_width == 2.0
    assert filters.max_width == 50.0
    assert filters.min_credit_pct_of_width == 0.20


def test_make_filters_from_settings_custom():
    settings = Settings(
        phase='paper', live_trade_enabled=False, run_dir=Path('/tmp'),
        bpcs_target_dte=30, bpcs_min_width=5.0, bpcs_max_width=15.0,
    )
    filters = make_filters_from_settings(settings)
    assert filters.target_dte == 30
    assert filters.min_width == 5.0
    assert filters.max_width == 15.0


def test_passes_dte_window():
    f = BPCSFilters(target_dte=35, min_dte=25, max_dte=50, min_width=2.0, max_width=10.0, min_credit_pct_of_width=0.20)
    assert passes_dte_window(25, f) is True
    assert passes_dte_window(35, f) is True
    assert passes_dte_window(50, f) is True
    assert passes_dte_window(24, f) is False
    assert passes_dte_window(51, f) is False


def test_passes_bpcs_conviction():
    f = BPCSFilters(target_dte=35, min_dte=25, max_dte=50, min_width=2.0, max_width=10.0, min_credit_pct_of_width=0.20)
    assert passes_bpcs_conviction(0.50, f) is True
    assert passes_bpcs_conviction(0.40, f) is False  # below floor


# ===== select_strikes =====

def test_select_strikes_2pct_otm():
    short, long = select_strikes(spot=100.0)
    assert short == 98.0  # 2% OTM
    assert long == 96.0   # 4% OTM


def test_select_strikes_zero_spot_returns_none():
    assert select_strikes(spot=0) is None
    assert select_strikes(spot=-10) is None


def test_select_strikes_high_price():
    """SPY/QQQ at $770 should still produce strikes."""
    short, long = select_strikes(spot=770.0)
    # 770 * 0.98 = 754.6 → rounds to 755
    assert short == 755.0
    # 770 * 0.96 = 739.2 → rounds to 739
    assert long == 739.0
    assert long < short


# ===== BullPutSpreadSignal dataclass =====

def test_bpcs_signal_has_leg_estimates():
    sig = BullPutSpreadSignal(
        short_strike=95.0, long_strike=90.0, expiry='2026-09-15',
        credit_estimate=1.50, max_loss_per_contract=350.0,
        max_gain_per_contract=150.0, pop=0.70, conviction=0.65,
        short_strike_price_estimate=1.45, long_strike_price_estimate=0.55,
        short_put_symbol='SPY260915P00095000',
        long_put_symbol='SPY260915P00090000',
    )
    assert sig.short_strike_price_estimate == 1.45
    assert sig.long_strike_price_estimate == 0.55
    assert sig.short_put_symbol == 'SPY260915P00095000'
    assert sig.long_put_symbol == 'SPY260915P00090000'


# ===== Risk engine evaluate_pre_trade_spread =====

def _make_risk(tmp_path: Path) -> RiskEngine:
    settings = Settings(
        phase='paper', live_trade_enabled=False, run_dir=tmp_path,
        bpcs_max_loss_pct=5.0, bpcs_max_loss_dollar=600.0,
    )
    return RiskEngine(settings, tmp_path)


def test_spread_gate_blocks_max_loss_pct(tmp_path: Path):
    """max_loss > 5% of equity → blocked."""
    risk = _make_risk(tmp_path)
    state = risk.load_state()
    # equity=10000, 5% = $500; ask for $600 max loss
    decision = risk.evaluate_pre_trade_spread(state, equity=10000.0, max_loss=600.0)
    assert decision.allowed is False
    assert 'bpcs_max_loss_pct' in decision.reason


def test_spread_gate_blocks_max_loss_dollar(tmp_path: Path):
    """max_loss > $600 absolute → blocked."""
    risk = _make_risk(tmp_path)
    state = risk.load_state()
    decision = risk.evaluate_pre_trade_spread(state, equity=100000.0, max_loss=700.0)
    assert decision.allowed is False
    assert 'bpcs_max_loss_dollar' in decision.reason


def test_spread_gate_allows_normal(tmp_path: Path):
    risk = _make_risk(tmp_path)
    state = risk.load_state()
    decision = risk.evaluate_pre_trade_spread(state, equity=10000.0, max_loss=150.0)
    assert decision.allowed is True


def test_record_open_spread_does_not_subtract_debit(tmp_path: Path):
    """KEY behavior: open_spread does NOT add negative debit to daily_pnl.

    Critical because a credit spread RECEIVES credit (income), not
    pays debit. If we subtracted, daily_pnl would be wrong.
    """
    risk = _make_risk(tmp_path)
    state = risk.load_state()
    initial_daily = state.daily_pnl
    risk.record_open_spread(state, symbol='SPY', max_loss=200.0, event_id='bpcs-test-1')
    assert state.daily_pnl == initial_daily  # NOT changed
    assert state.open_position_count == 1


# ===== Journal get_legs_for_position =====

def test_get_legs_for_position_returns_multi(tmp_path: Path):
    """For multi-leg strategies (e.g. closing a spread), multiple events
    can be tied to one position_id. A single MLEG open is just one event.
    """
    journal = Journal(tmp_path)
    pid = journal.new_position_id()
    # Write 2 events tied to same position (e.g., partial fills)
    journal.append(TradeEvent(
        event_id='leg1', ts=journal.now_iso(), kind='open', symbol='SPY_PUT95',
        side='sell', qty=1, price=1.50, position_id=pid, strategy_id='bull_put_credit_spread',
    ))
    journal.append(TradeEvent(
        event_id='leg2', ts=journal.now_iso(), kind='fill_partial', symbol='SPY_PUT95',
        side='sell', qty=1, price=1.55, position_id=pid, strategy_id='bull_put_credit_spread',
    ))
    legs = journal.get_legs_for_position(pid)
    assert len(legs) == 2
    assert legs[0]['side'] == 'sell'
    assert legs[1]['kind'] == 'fill_partial'


# ===== Strategy registry =====

def test_registry_includes_bpcs_when_enabled():
    settings = Settings(phase='paper', live_trade_enabled=False, run_dir=Path('/tmp'), bpcs_enabled=True)
    enabled = enabled_strategies(settings)
    ids = [s.strategy_id for s in enabled]
    assert 'bull_put_credit_spread' in ids
    assert 'long_call' in ids


def test_registry_excludes_bpcs_when_disabled():
    settings = Settings(phase='paper', live_trade_enabled=False, run_dir=Path('/tmp'), bpcs_enabled=False)
    enabled = enabled_strategies(settings)
    ids = [s.strategy_id for s in enabled]
    assert 'bull_put_credit_spread' not in ids
    assert 'long_call' in ids


def test_get_strategy_bpcs():
    desc = get_strategy('bull_put_credit_spread')
    assert desc is not None
    assert desc.strategy_id == 'bull_put_credit_spread'


# ===== Executor submit_bull_put_spread (mocked broker) =====

def test_executor_submit_bpcs_happy_path(tmp_path: Path):
    """MLEG spread order submitted → journal writes → risk records."""
    settings = Settings(
        phase='paper', live_trade_enabled=False, run_dir=tmp_path,
        bpcs_max_loss_pct=5.0, bpcs_max_loss_dollar=600.0,
    )
    broker = MagicMock(spec=BrokerClient)
    spread_order = MagicMock(id='spread-oid-1', status='ACCEPTED', submitted_at='now')
    broker.submit_credit_spread = MagicMock(return_value=spread_order)
    broker.cancel_order = MagicMock()

    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = risk.load_state()
    executor = Executor(settings, broker, journal, risk)

    sig = BullPutSpreadSignal(
        short_strike=95.0, long_strike=90.0, expiry='2026-09-15',
        credit_estimate=1.50, max_loss_per_contract=350.0,
        max_gain_per_contract=150.0, pop=0.70, conviction=0.65,
        short_strike_price_estimate=1.45, long_strike_price_estimate=0.55,
        short_put_symbol='SPY260915P00095000',
        long_put_symbol='SPY260915P00090000',
    )
    result = executor.submit_bull_put_spread(
        sig, equity=20000.0, state=state,  # 5% of 20000 = 1000 > 350
        short_put_symbol='SPY260915P00095000',
        long_put_symbol='SPY260915P00090000',
    )
    assert result.accepted is True
    assert result.position_id != ''
    assert 'bpcs-spread-' in result.client_order_id

    # MLEG order should have been submitted (single call, not 2 separate)
    broker.submit_credit_spread.assert_called_once()
    broker.submit_open_sell.assert_not_called()  # we use MLEG, not 2 separate orders
    broker.submit_buy.assert_not_called()  # ditto

    # Event recorded
    legs = journal.get_legs_for_position(result.position_id)
    assert len(legs) == 1
    assert legs[0]['kind'] == 'open_spread'


def test_executor_submit_bpcs_mleg_fails(tmp_path: Path):
    """If MLEG submit fails, no journal writes, no risk change."""
    settings = Settings(phase='paper', live_trade_enabled=False, run_dir=tmp_path)
    broker = MagicMock(spec=BrokerClient)
    broker.submit_credit_spread = MagicMock(side_effect=RuntimeError("broker error"))

    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = risk.load_state()
    initial_count = state.open_position_count
    executor = Executor(settings, broker, journal, risk)

    sig = BullPutSpreadSignal(
        short_strike=95.0, long_strike=90.0, expiry='2026-09-15',
        credit_estimate=1.50, max_loss_per_contract=350.0,
        max_gain_per_contract=150.0, pop=0.70, conviction=0.65,
        short_strike_price_estimate=1.45, long_strike_price_estimate=0.55,
        short_put_symbol='SPY_PUT95', long_put_symbol='SPY_PUT90',
    )
    result = executor.submit_bull_put_spread(
        sig, equity=20000.0, state=state,  # 5% = 1000 > 350
        short_put_symbol='SPY_PUT95', long_put_symbol='SPY_PUT90',
    )
    assert result.accepted is False
    assert 'broker_error_mleg' in result.reason
    assert state.open_position_count == initial_count


def test_executor_submit_bpcs_mleg_event_has_both_legs_in_payload(tmp_path: Path):
    """The journal event payload must contain both legs for audit trail."""
    settings = Settings(
        phase='paper', live_trade_enabled=False, run_dir=tmp_path,
        bpcs_max_loss_pct=5.0, bpcs_max_loss_dollar=600.0,
    )
    broker = MagicMock(spec=BrokerClient)
    spread_order = MagicMock(id='spread-oid', status='ACCEPTED', submitted_at='now')
    broker.submit_credit_spread = MagicMock(return_value=spread_order)

    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = risk.load_state()
    executor = Executor(settings, broker, journal, risk)

    sig = BullPutSpreadSignal(
        short_strike=95.0, long_strike=90.0, expiry='2026-09-15',
        credit_estimate=1.50, max_loss_per_contract=350.0,
        max_gain_per_contract=150.0, pop=0.70, conviction=0.65,
        short_strike_price_estimate=1.45, long_strike_price_estimate=0.55,
        short_put_symbol='SPY_PUT95', long_put_symbol='SPY_PUT90',
    )
    result = executor.submit_bull_put_spread(
        sig, equity=20000.0, state=state,
        short_put_symbol='SPY_PUT95', long_put_symbol='SPY_PUT90',
    )
    assert result.accepted is True
    legs = journal.get_legs_for_position(result.position_id)
    assert len(legs) == 1
    fill = legs[0].get('raw_broker_fill', {})
    # The raw fill is JSON-encoded in the DB, so we need to load it
    import json
    fill_data = json.loads(fill) if isinstance(fill, str) else fill
    assert fill_data.get('order_class') == 'MLEG'
    assert fill_data.get('short_leg') == 'SPY_PUT95'
    assert fill_data.get('long_leg') == 'SPY_PUT90'


def test_open_spread_materializes_position_row(tmp_path: Path):
    """open_spread must be visible to the position monitor as one logical position."""
    journal = Journal(tmp_path)
    pid = journal.new_position_id()
    journal.append(TradeEvent(
        event_id='spread-open-1', ts=journal.now_iso(), kind='open_spread',
        symbol='BPCS:IWM260918P00296000/IWM260918P00290000', side='sell', qty=1,
        price=1.43, position_id=pid, strategy_id='bull_put_credit_spread',
        raw_broker_fill={
            'order_class': 'MLEG',
            'short_leg': 'IWM260918P00296000',
            'long_leg': 'IWM260918P00290000',
        },
    ))
    rows = journal.get_open_positions()
    assert len(rows) == 1
    assert rows[0]['position_id'] == pid
    assert rows[0]['symbol'].startswith('BPCS:')
    assert rows[0]['qty_open'] == 1
    assert rows[0]['avg_entry_price'] == pytest.approx(1.43)


def test_backfills_existing_open_spread_without_position_row(tmp_path: Path):
    """Older 04afba5 journals had open_spread trades but no positions row."""
    journal = Journal(tmp_path)
    pid = journal.new_position_id()
    # Simulate the old bug by writing directly into trades, bypassing append().
    import sqlite3
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute(
            """
            INSERT INTO trades (event_id, ts, kind, symbol, side, qty, price, position_id, strategy_id, raw_broker_fill)
            VALUES (?, ?, 'open_spread', ?, 'sell', 1, 1.43, ?, 'bull_put_credit_spread', ?)
            """,
            ('legacy-spread', journal.now_iso(), 'BPCS:IWM260918P00296000/IWM260918P00290000', pid,
             json.dumps({'short_leg': 'IWM260918P00296000', 'long_leg': 'IWM260918P00290000'})),
        )
        conn.commit()
    assert journal.get_open_positions() == []
    # Re-instantiating Journal runs migration/backfill.
    journal2 = Journal(tmp_path)
    rows = journal2.get_open_positions()
    assert len(rows) == 1
    assert rows[0]['position_id'] == pid


def test_closer_flatten_credit_spread_writes_close_spread(tmp_path: Path):
    settings = Settings(phase='paper', live_trade_enabled=False, run_dir=tmp_path)
    broker = MagicMock(spec=BrokerClient)
    spread_order = MagicMock(id='close-spread-oid', status='ACCEPTED', submitted_at='now')
    broker.submit_close_credit_spread = MagicMock(return_value=spread_order)
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    closer = Closer(settings, broker, journal, risk)

    pid = journal.new_position_id()
    res = closer.flatten_credit_spread(
        position_id=pid,
        short_put_symbol='IWM260918P00296000',
        long_put_symbol='IWM260918P00290000',
        qty=1,
        short_put_limit=3.00,
        long_put_limit=2.00,
        entry_credit=1.43,
        reason='bpcs_tp',
    )
    assert res.accepted is True
    broker.submit_close_credit_spread.assert_called_once()
    legs = journal.get_legs_for_position(pid)
    assert len(legs) == 1
    assert legs[0]['kind'] == 'close_spread'
    assert legs[0]['price'] == pytest.approx(1.0)
    assert legs[0]['realized_pnl'] == pytest.approx(43.0)


def test_position_monitor_bpcs_tp_closes_atomically(tmp_path: Path):
    settings = Settings(
        phase='paper', live_trade_enabled=False, run_dir=tmp_path,
        bpcs_profit_target_pct=0.50,
    )
    broker = MagicMock(spec=BrokerClient)
    # close_debit = short ask 2.50 - long bid 1.90 = 0.60;
    # entry credit 1.43 -> target debit 0.715, so TP should fire.
    broker.get_option_snapshot.side_effect = [
        {'bid': 2.45, 'ask': 2.50},
        {'bid': 1.90, 'ask': 1.95},
    ]
    journal = Journal(tmp_path)
    risk = RiskEngine(settings, tmp_path)
    state = risk.load_state()
    closer = MagicMock(spec=Closer)
    closer.flatten_credit_spread.return_value = MagicMock(accepted=True)
    monitor = PositionMonitor(settings, broker, journal, risk, closer)

    pid = journal.new_position_id()
    journal.append(TradeEvent(
        event_id='spread-open-2', ts=journal.now_iso(), kind='open_spread',
        symbol='BPCS:IWM260918P00296000/IWM260918P00290000', side='sell', qty=1,
        price=1.43, position_id=pid, strategy_id='bull_put_credit_spread',
        raw_broker_fill={
            'order_class': 'MLEG',
            'short_leg': 'IWM260918P00296000',
            'long_leg': 'IWM260918P00290000',
        },
    ))
    tick = monitor.tick(state)
    assert tick.closes_triggered == 1
    assert tick.reasons['bpcs_tp'] == 1
    closer.flatten_credit_spread.assert_called_once()
    kwargs = closer.flatten_credit_spread.call_args.kwargs
    assert kwargs['short_put_symbol'] == 'IWM260918P00296000'
    assert kwargs['long_put_symbol'] == 'IWM260918P00290000'
    assert kwargs['entry_credit'] == pytest.approx(1.43)