"""Tests for the position monitor with gain-protection semantics.

These tests verify the cardinal rule: **hold onto gains, don't give them away.**

Each scenario covers a specific gain-protection scenario:

1. SL fires before any gain — full exit
2. TP partial fires at +50% — sell half, lock gain
3. TP full fires only if no partial AND mark is at +100%
4. Trailing floor never gives gains back to entry (profit_floor_pct = 1.10)
5. Trailing floor respects peak retention when peak is much higher
6. Min profit dollar prevents wash-trade exits
7. EOD flatten closes all remaining qty
8. After partial, the remaining half never gets a second partial
"""

from pathlib import Path
from unittest.mock import MagicMock

from flip_options_bot.config import Settings
from flip_options_bot.execution import Closer
from flip_options_bot.execution.executor import ExecutionResult
from flip_options_bot.journal import Journal
from flip_options_bot.monitor.position_monitor import PositionMonitor
from flip_options_bot.risk import RiskState


def _wire_monitor(tmp_path: Path):
    """Build a PositionMonitor with mocked broker and closer."""
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        run_dir=tmp_path,
        sl_threshold_pct=0.50,
        tp_multiplier=1.50,
        tp_full_multiplier=2.00,
        trailing_arm_pct=0.10,
        trailing_retention=0.50,
        profit_floor_pct=1.10,
        min_tp_profit_dollar=25.0,
        close_eod_minutes=15,
    )
    journal = Journal(tmp_path)
    risk = MagicMock()
    risk.load_state = MagicMock(return_value=RiskState())
    broker = MagicMock()
    broker.list_open_orders = MagicMock(return_value=[])
    closer = MagicMock(spec=Closer)
    closer.flatten_position = MagicMock(
        return_value=ExecutionResult(accepted=True, client_order_id="close-1")
    )
    monitor = PositionMonitor(settings, broker, journal, risk, closer)
    return monitor, journal, closer


def _open_position(journal: Journal, symbol: str, qty: int, avg_entry: float,
                   qty_closed: int = 0, peak_mark: float | None = None) -> str:
    """Helper: open a position with given entry, return position_id."""
    from flip_options_bot.journal.journal import TradeEvent
    pos_id = Journal.new_position_id()
    journal.upsert(TradeEvent(
        event_id=f"open-{pos_id}",
        ts="2026-08-11T15:00:00+00:00",
        kind="open",
        symbol=symbol,
        side="buy",
        qty=qty,
        price=avg_entry,
        position_id=pos_id,
        strategy_id="long_call",
        raw_broker_fill={"order_id": f"order-{pos_id}"},
    ))
    if qty_closed > 0 or peak_mark is not None:
        with __import__("sqlite3").connect(journal.db_path) as conn:
            conn.execute(
                "UPDATE positions SET qty_closed = ?, peak_mark = ? WHERE position_id = ?",
                (qty_closed, peak_mark, pos_id),
            )
            conn.commit()
    return pos_id


def _snap(bid: float, ask: float) -> dict:
    return {"bid": bid, "ask": ask}


def test_pending_close_order_blocks_duplicate_close_submit(tmp_path: Path):
    """A resting close SELL should stop the monitor from submitting duplicates."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    symbol = "SPY260815C00750000"
    _open_position(journal, symbol, qty=2, avg_entry=1.00)
    order = MagicMock()
    order.client_order_id = "close-existing"
    order.side = "sell"
    order.symbol = symbol
    monitor.broker.list_open_orders = MagicMock(return_value=[order])
    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=0.40, ask=0.40))

    tick = monitor.tick(RiskState())

    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


# ===== SL =====

def test_sl_fires_at_50pct_full_exit(tmp_path: Path):
    """SL: mark = 40% of entry → exit full position."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(journal, "SPY260815C00750000", qty=2, avg_entry=1.00)

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=0.40, ask=0.40))
    tick = monitor.tick(RiskState())

    assert tick.closes_triggered == 1
    assert tick.reasons == {"sl": 1}
    # Full exit: qty=2
    call = closer.flatten_position.call_args
    assert call.kwargs["qty"] == 2
    assert call.kwargs["reason"] == "sl"
    assert call.kwargs["symbol"] == "SPY260815C00750000"


# ===== TP partial =====

def test_tp_partial_at_50pct_sells_half(tmp_path: Path):
    """TP partial at +50%: entry $1, mark $1.50, qty=2 → sell 1, lock $50 gain."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(journal, "SPY260815C00750000", qty=2, avg_entry=1.00)

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())

    assert tick.closes_triggered == 1
    assert tick.reasons == {"tp_partial": 1}
    call = closer.flatten_position.call_args
    # half of 2 = 1
    assert call.kwargs["qty"] == 1
    assert call.kwargs["reason"] == "tp_partial"


def test_tp_partial_odd_qty_rounds_up(tmp_path: Path):
    """TP partial with qty=3 → close 2 (ceil(3/2))."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(journal, "SPY260815C00750000", qty=3, avg_entry=1.00)

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())
    assert tick.closes_triggered == 1
    call = closer.flatten_position.call_args
    # (qty + 1) // 2 = (3+1)//2 = 2
    assert call.kwargs["qty"] == 2


# ===== Min profit dollar =====

def test_min_profit_dollar_blocks_tp_at_wash(tmp_path: Path):
    """TP would fire at $1.50 but only $50 per contract — 1 contract = $50 profit.
    Set min_tp_profit to $100. Should NOT fire.
    """
    settings = Settings(
        phase="paper", live_trade_enabled=False, run_dir=tmp_path,
        min_tp_profit_dollar=100.0,
    )
    journal = Journal(tmp_path)
    risk = MagicMock()
    broker = MagicMock()
    closer = MagicMock()
    closer.flatten_position = MagicMock(
        return_value=ExecutionResult(accepted=True)
    )
    monitor = PositionMonitor(settings, broker, journal, risk, closer)

    _open_position(journal, "SPY260815C00750000", qty=1, avg_entry=1.00)
    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())

    # 1 contract * ($1.50 - $1.00) * 100 = $50 < $100 min → no TP
    assert tick.closes_triggered == 0
    closer.flatten_position.assert_not_called()


def test_min_profit_dollar_passes_with_qty_2(tmp_path: Path):
    """Same scenario but qty=2 → $100 profit at mark=$1.50 → TP fires."""
    settings = Settings(
        phase="paper", live_trade_enabled=False, run_dir=tmp_path,
        min_tp_profit_dollar=100.0,
    )
    journal = Journal(tmp_path)
    risk = MagicMock()
    broker = MagicMock()
    closer = MagicMock()
    closer.flatten_position = MagicMock(
        return_value=ExecutionResult(accepted=True)
    )
    monitor = PositionMonitor(settings, broker, journal, risk, closer)

    _open_position(journal, "SPY260815C00750000", qty=2, avg_entry=1.00)
    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())
    # 2 contracts * ($1.50 - $1.00) * 100 = $100 >= $100 min → TP fires
    assert tick.closes_triggered == 1


# ===== TP full =====

def test_single_contract_runner_uses_full_tp_not_partial(tmp_path: Path):
    """Single-contract winners are runners: no fake partial on a 1-lot."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(journal, "SPY260815C00750000", qty=1, avg_entry=1.00)

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=2.50, ask=2.50))
    tick = monitor.tick(RiskState())
    # tp_partial is multi-contract only. A 1-lot exits only at full TP.
    assert tick.reasons == {"tp": 1}
    call = closer.flatten_position.call_args
    assert call.kwargs["reason"] == "tp"
    assert call.kwargs["qty"] == 1


def test_tp_full_only_after_no_partial(tmp_path: Path):
    """After partial was taken (qty_closed=1, qty=1 left), full TP should NOT
    fire again — the remaining half is left to run with the trailing floor."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(journal, "SPY260815C00750000", qty=2, avg_entry=1.00, qty_closed=1)

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=2.50, ask=2.50))
    tick = monitor.tick(RiskState())
    # Neither tp_partial (qty_closed=1) nor tp_full (qty_closed=1) fires.
    # Trailing_floor would fire if peak>=1.10 and mark<=max(peak*0.50, 1.10).
    # peak defaults to avg_entry (1.00) since we didn't set it.
    # gain_pct = (1.00 - 1.00)/1.00 = 0 → arm not hit → no close.
    assert tick.closes_triggered == 0


# ===== Trailing floor =====

def test_trailing_floor_never_gives_gains_back_to_entry(tmp_path: Path):
    """Peak was +10% (peak_mark=1.10). Mark drops to entry (1.00).
    retention=0.50 means target=0.55, profit_floor=1.10 means floor=1.10.
    We use max(retention, profit_floor) = 1.10.
    Mark=1.00 < 1.10 → trailing_floor fires."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(
        journal, "SPY260815C00750000", qty=1, avg_entry=1.00, peak_mark=1.10
    )

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.00, ask=1.00))
    tick = monitor.tick(RiskState())

    # mark=1.00 ≤ max(peak*0.50=0.55, entry*1.10=1.10) = 1.10 → fires
    assert tick.closes_triggered == 1
    assert tick.reasons == {"trailing_floor": 1}


def test_trailing_floor_respects_higher_peak_retention(tmp_path: Path):
    """Peak was +200% (peak=3.00). Trailing floor target=1.50. But mark=1.50
    also hits +50% tp_partial → tp_partial wins (priority)."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(
        journal, "SPY260815C00750000", qty=2, avg_entry=1.00, peak_mark=3.00
    )

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())
    # tp_partial wins because +50% mark fires before trailing_floor (priority order)
    assert tick.reasons == {"tp_partial": 1}


def test_trailing_floor_after_partial_taken(tmp_path: Path):
    """After partial taken (qty_closed=1, qty=1 left), trailing_floor is the
    only gain-protection active. Peak was 3.00, mark=1.50 → fires."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(
        journal, "SPY260815C00750000", qty=2, avg_entry=1.00,
        qty_closed=1, peak_mark=3.00,
    )

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.50, ask=1.50))
    tick = monitor.tick(RiskState())
    # qty_closed=1 means tp_partial/tp_full won't fire again.
    # gain_pct = (3.00-1.00)/1.00 = 2.00 >= 0.10 arm → armed
    # retained_gain_floor = 1.00 + (3.00-1.00)*0.50 = 2.00
    # profit_floor = 1.00 * 1.10 = 1.10
    # exit_floor = max(2.00, 1.10) = 2.00
    # mark=1.50 ≤ 2.00 → trailing_floor fires
    assert tick.reasons == {"trailing_floor": 1}


def test_trailing_floor_retains_percent_of_gain_not_peak_price(tmp_path: Path):
    """Peak retention should retain gains, not multiply the whole option price."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(
        journal,
        "SPY260815C00750000",
        qty=1,
        avg_entry=1.00,
        peak_mark=1.50,
    )

    # Old formula peak*0.50 = 0.75 would not close here. New formula retains
    # 50% of the 0.50 gain -> floor 1.25, so the winner is protected.
    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=1.24, ask=1.24))
    tick = monitor.tick(RiskState())

    assert tick.reasons == {"trailing_floor": 1}
    call = closer.flatten_position.call_args
    assert call.kwargs["reason"] == "trailing_floor"


def test_fixed_profit_floor_never_exceeds_observed_peak(tmp_path: Path):
    """A lower arm than profit floor should not instantly close at arm tick."""
    monitor, _, _ = _wire_monitor(tmp_path)
    monitor.trailing_arm_pct = 0.06
    monitor.profit_floor_pct = 1.08
    monitor.trailing_retention = 0.70

    floor = monitor._trailing_exit_floor(avg_entry=1.00, peak_mark=1.06)

    assert floor == 1.06


def test_trailing_floor_not_armed_when_gain_below_threshold(tmp_path: Path):
    """Peak gain was only +5% (< arm_pct=10%). No trailing floor."""
    monitor, journal, closer = _wire_monitor(tmp_path)
    _open_position(
        journal, "SPY260815C00750000", qty=1, avg_entry=1.00, peak_mark=1.05
    )

    monitor.broker.get_option_snapshot = MagicMock(return_value=_snap(bid=0.80, ask=0.80))
    tick = monitor.tick(RiskState())
    # gain = (1.05-1.00)/1.00 = 0.05 < arm_pct=0.10 → not armed → no close
    # mark=0.80 is below entry but arm didn't fire → no SL either (sl at 0.50)
    assert tick.closes_triggered == 0


# ===== Exit price =====

def test_exit_price_tp_uses_mark_no_slippage(tmp_path: Path):
    """TP exit price = mark (we never accept below mark for gain-capture)."""
    monitor, _, _ = _wire_monitor(tmp_path)
    p = monitor._exit_price("tp_partial", mark=1.50, avg_entry=1.00)
    assert p == 1.50  # not 1.50 * 0.99

    p = monitor._exit_price("tp", mark=2.50, avg_entry=1.00)
    assert p == 2.50


def test_exit_price_sl_uses_97pct_mark(tmp_path: Path):
    """SL exit = mark * 0.97 (small slippage OK)."""
    monitor, _, _ = _wire_monitor(tmp_path)
    p = monitor._exit_price("sl", mark=0.50, avg_entry=1.00)
    assert p == 0.50 * 0.97


def test_exit_price_trail_uses_mark(tmp_path: Path):
    """Trailing floor exit = mark (we're capturing the floor, no slip)."""
    monitor, _, _ = _wire_monitor(tmp_path)
    p = monitor._exit_price("trailing_floor", mark=1.20, avg_entry=1.00)
    assert p == 1.20


def test_exit_price_floor_is_5_cents(tmp_path: Path):
    """Even if mark is tiny, exit at minimum 5 cents."""
    monitor, _, _ = _wire_monitor(tmp_path)
    p = monitor._exit_price("sl", mark=0.01, avg_entry=1.00)
    assert p == 0.05
