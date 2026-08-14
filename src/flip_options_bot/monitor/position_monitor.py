"""Position monitor — runs alongside the scanner loop.

The monitor watches open positions every `position_monitor_interval_s`
seconds and decides whether any position needs to be closed.

The cardinal rule: **hold onto gains, don't give them away.**

Close triggers, in priority order:

1. **Stop-loss (sl)**: mark <= entry * sl_threshold_pct (default 50%)
   — the absolute floor. Bad trade, get out.

2. **Take-profit partial (tp_partial)**: mark >= entry * tp_multiplier
   (default 1.50, i.e. +50%) — sell HALF the position. Lock the gain.
   The other half continues to run with the trailing floor.

3. **Take-profit full (tp)**: ONLY if no partial has been taken yet AND
   mark >= entry * tp_full_multiplier (default 2.00, i.e. +100%) —
   sell the full position. Big winner, take it.

4. **Trailing profit floor (trailing_floor)**: position gained >=
   arm_pct (default +10%), then mark drops below max(peak *
   retention_pct, entry * profit_floor_pct) → close the REMAINING
   half. **Never give gains back to entry.** profit_floor_pct (default
   1.10) means we keep at least +10% even after a peak.

5. **EOD flatten (eod)**: minutes_to_close < close_eod_minutes
   (default 15). No overnight risk.

Triggers fire `Closer.flatten_position(reason=...)` so the journal
records why. The Closer itself is idempotent on client_order_id.

Partial-fill state lives in journal.positions.qty_closed (incremented
on each partial close). After tp_partial, the monitor expects qty_closed
to be >= qty/2, and the remaining half is monitored against trailing_floor
instead of tp_full.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date

from ..broker import BrokerClient
from ..config import Settings
from ..execution import Closer
from ..journal import Journal
from ..market_time import (
    DEFAULT_EOD_FLATTEN_MINUTES,
    is_weekday,
    minutes_to_close,
    now_utc,
)
from ..risk import RiskEngine, RiskState

log = logging.getLogger("flip_options_bot.monitor")


@dataclass
class MonitorTick:
    positions_seen: int = 0
    closes_triggered: int = 0
    reasons: dict[str, int] | None = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = {}


class PositionMonitor:
    """One instance per daemon. Stateless; holds broker/journal/risk/closer."""

    def __init__(
        self,
        settings: Settings,
        broker: BrokerClient,
        journal: Journal,
        risk: RiskEngine,
        closer: Closer,
    ):
        self.settings = settings
        self.broker = broker
        self.journal = journal
        self.risk = risk
        self.closer = closer
        self.eod_minutes = getattr(settings, "close_eod_minutes", DEFAULT_EOD_FLATTEN_MINUTES)
        self.sl_threshold_pct = getattr(settings, "sl_threshold_pct", 0.50)
        self.tp_multiplier = getattr(settings, "tp_multiplier", 1.50)
        self.tp_full_multiplier = getattr(settings, "tp_full_multiplier", 2.00)
        self.trailing_arm_pct = getattr(settings, "trailing_arm_pct", 0.10)
        self.trailing_retention = getattr(settings, "trailing_retention", 0.50)
        self.profit_floor_pct = getattr(settings, "profit_floor_pct", 1.10)
        # Min dollar profit before TP fires — prevents exiting at a wash
        # when spread is so wide that even a +50% mark = small dollars.
        self.min_tp_profit_dollar = getattr(settings, "min_tp_profit_dollar", 25.0)

    def tick(self, state: RiskState) -> MonitorTick:
        """One monitoring pass. Returns a MonitorTick summary."""
        result = MonitorTick()
        positions = self.journal.get_open_positions()
        result.positions_seen = len(positions)
        pending_close_symbols = self._pending_close_symbols()

        for pos in positions:
            qty_open = pos.get("qty_open", 0) or 0
            qty_closed = pos.get("qty_closed", 0) or 0
            qty = qty_open - qty_closed
            if qty <= 0:
                continue

            symbol = pos.get("symbol", "")
            position_id = pos.get("position_id", "")
            strategy_id = pos.get("strategy_id", "") or ""

            if symbol in pending_close_symbols and not symbol.startswith("BPCS:"):
                log.info("monitor skip %s — pending close order already working", symbol)
                continue

            if strategy_id == "bull_put_credit_spread" or symbol.startswith("BPCS:"):
                spread_decision = self._evaluate_spread_position(pos, qty, state)
                if spread_decision is None:
                    continue
                trigger, short_sym, long_sym, short_limit, long_limit, entry_credit = spread_decision
                close = self.closer.flatten_credit_spread(
                    position_id=position_id,
                    short_put_symbol=short_sym,
                    long_put_symbol=long_sym,
                    qty=qty,
                    short_put_limit=short_limit,
                    long_put_limit=long_limit,
                    entry_credit=entry_credit,
                    reason=trigger,
                )
                if close.accepted:
                    result.closes_triggered += 1
                    result.reasons[trigger] = result.reasons.get(trigger, 0) + 1
                    log.info(
                        "monitor close spread %s/%s qty=%d trigger=%s debit=%.2f",
                        short_sym, long_sym, qty, trigger, max(short_limit - long_limit, 0.01),
                    )
                continue

            avg_entry_raw = pos.get("avg_entry_price")
            avg_entry = float(avg_entry_raw) if avg_entry_raw is not None else 0.0
            peak_mark_raw = pos.get("peak_mark")
            peak_mark = (
                float(peak_mark_raw)
                if peak_mark_raw is not None
                else max(avg_entry, 0.0)
            )

            if strategy_id == "long_equity":
                snap = self.broker.get_stock_quote(symbol) or {}
            else:
                snap = self.broker.get_option_snapshot(symbol) or {}
            bid = float(snap.get("bid") or 0.0) if isinstance(snap, dict) else 0.0
            ask = float(snap.get("ask") or 0.0) if isinstance(snap, dict) else 0.0
            if bid and ask:
                mark = (bid + ask) / 2.0
            elif bid:
                mark = bid
            elif ask:
                mark = ask
            else:
                # No fresh quote; use last cached peak (conservative — don't
                # trigger SL/TP on stale data).
                continue

            # Update peak in DB if mark is higher (idempotent — SET peak=MAX(peak, ?))
            if mark > peak_mark:
                self._update_peak(position_id, mark)
                peak_mark = mark

            if strategy_id == "long_equity":
                decision = self._evaluate_long_equity_position(pos, qty, avg_entry, peak_mark, mark, state)
            else:
                decision = self._evaluate_position(
                    pos, qty, qty_closed, avg_entry, peak_mark, mark, state
                )
            if decision is None:
                continue
            trigger, close_qty, limit_price = decision

            close = self.closer.flatten_position(
                symbol=symbol,
                qty=close_qty,
                position_id=position_id,
                limit_price=limit_price,
                reason=trigger,
            )
            if close.accepted:
                result.closes_triggered += 1
                result.reasons[trigger] = result.reasons.get(trigger, 0) + 1
                log.info(
                    "monitor close %s qty=%d trigger=%s mark=%.2f limit=%.2f",
                    symbol, close_qty, trigger, mark, limit_price,
                )

        if result.closes_triggered:
            log.warning(
                "monitor tick: %d positions, %d closes (%s)",
                result.positions_seen, result.closes_triggered, result.reasons,
            )
        return result

    def _pending_close_symbols(self) -> set[str]:
        """Return symbols that already have a bot-owned SELL close order working.

        A `close_attempt` keeps the journal position open until broker fill.
        Without this guard, the monitor can submit a duplicate SELL every tick
        while the first close limit is still resting.
        """
        try:
            orders = self.broker.list_open_orders()
        except Exception as exc:
            log.warning("could not inspect open orders for pending-close guard: %s", exc)
            return set()
        symbols: set[str] = set()
        for order in orders or []:
            coid = str(getattr(order, "client_order_id", "") or "")
            side = str(getattr(order, "side", "") or "").upper()
            symbol = str(getattr(order, "symbol", "") or "")
            if (
                coid.startswith("close-")
                and not coid.startswith("close-spread-")
                and side.endswith("SELL")
                and symbol
            ):
                symbols.add(symbol)
        return symbols

    def _evaluate_long_equity_position(
        self,
        pos: dict,
        qty: int,
        avg_entry: float,
        peak_mark: float,
        mark: float,
        state: RiskState,
    ) -> tuple[str, int, float] | None:
        """Long-share protection: full exits at stop, TP, trailing floor, or EOD."""
        if mark <= 0 or avg_entry <= 0:
            return None
        stop_raw = pos.get("sl_trigger_price")
        tp_raw = pos.get("tp_price")
        stop_price = float(stop_raw) if stop_raw is not None else avg_entry * (1.0 - self.settings.long_equity_stop_loss_pct)
        take_profit = float(tp_raw) if tp_raw is not None else avg_entry * (1.0 + self.settings.long_equity_take_profit_pct)

        if mark <= stop_price:
            return ("equity_sl", qty, round(max(mark, stop_price * 0.999), 2))
        if mark >= take_profit:
            return ("equity_tp", qty, round(mark, 2))

        gain_pct = (peak_mark - avg_entry) / avg_entry
        if gain_pct >= self.trailing_arm_pct:
            exit_floor = max(peak_mark * self.trailing_retention, avg_entry * self.profit_floor_pct)
            if mark <= exit_floor:
                return ("equity_trailing_floor", qty, round(mark, 2))

        now = now_utc()
        if is_weekday(now):
            minutes_left = minutes_to_close(now)
            if 0 <= minutes_left <= self.eod_minutes:
                return ("eod", qty, round(mark, 2))
        return None

    def _evaluate_position(
        self,
        pos: dict,
        qty: int,
        qty_closed: int,
        avg_entry: float,
        peak_mark: float,
        mark: float,
        state: RiskState,
        value_multiplier: int = 100,
    ) -> tuple[str, int, float] | None:
        """Decide what (if anything) to close. Returns (trigger, qty_to_close, limit_price) or None.

        Gain-protection semantics (priority order):
        1. SL always wins (no partial — full exit)
        2. tp_partial at +50%: sell half if no partial yet, IF minimum profit dollar met
        3. tp_full at +100% (if no partial was taken): exit all
        4. trailing_floor at arm+10% & mark < max(peak*retention, entry*profit_floor)
        5. EOD flatten — full exit, last 15 min of session
        """
        if mark <= 0 or avg_entry <= 0:
            return None

        # === 1. SL — full exit, no partial ===
        if mark <= avg_entry * self.sl_threshold_pct:
            limit_price = self._exit_price("sl", mark, avg_entry)
            return ("sl", qty, limit_price)

        # === 2. tp_partial — first time we hit +50%, sell half IF min profit $ ===
        # Compute the dollar profit if we exited at mark:
        # P&L per contract = (mark - avg_entry) * 100. For qty contracts:
        potential_profit_dollar = (mark - avg_entry) * value_multiplier * qty

        if (
            mark >= avg_entry * self.tp_multiplier
            and qty_closed == 0
            and potential_profit_dollar >= self.min_tp_profit_dollar
        ):
            # Close half (rounded up so odd-lots still lock gains)
            partial_qty = max(1, (qty + 1) // 2)
            limit_price = self._exit_price("tp_partial", mark, avg_entry)
            return ("tp_partial", partial_qty, limit_price)

        # === 3. tp_full — only if NO partial was taken AND mark is at +100% ===
        if (
            mark >= avg_entry * self.tp_full_multiplier
            and qty_closed == 0
            and potential_profit_dollar >= self.min_tp_profit_dollar
        ):
            limit_price = self._exit_price("tp", mark, avg_entry)
            return ("tp", qty, limit_price)

        # === 4. trailing_floor — only fires when armed (peak gain >= arm_pct) ===
        gain_pct = (peak_mark - avg_entry) / avg_entry
        if gain_pct >= self.trailing_arm_pct:
            retention_target = peak_mark * self.trailing_retention
            profit_floor = avg_entry * self.profit_floor_pct
            exit_floor = max(retention_target, profit_floor)
            if mark <= exit_floor:
                limit_price = self._exit_price("trailing_floor", mark, avg_entry)
                return ("trailing_floor", qty, limit_price)

        # === 5. EOD flatten — full exit, weekday only, last `eod_minutes` ===
        now = now_utc()
        if is_weekday(now):
            minutes_left = minutes_to_close(now)
            if 0 <= minutes_left <= self.eod_minutes:
                limit_price = self._exit_price("eod", mark, avg_entry)
                return ("eod", qty, limit_price)

        return None

    def _evaluate_spread_position(
        self,
        pos: dict,
        qty: int,
        state: RiskState,
    ) -> tuple[str, str, str, float, float, float] | None:
        """Evaluate a BPCS as ONE logical position.

        For a credit spread, entry is a credit and close is a debit:
          close_debit = short_put_ask - long_put_bid

        Profit target: buy back at 50% of credit.
        Stop: buy back at 2x credit (loss ~= credit received), bounded by
        spread width in reality. This is intentionally conservative.
        DTE exit: close at configured bpcs_close_at_dte to avoid gamma risk.
        """
        position_id = pos.get("position_id", "")
        legs = self.journal.get_legs_for_position(position_id)
        open_spread = next((leg for leg in legs if leg.get("kind") == "open_spread"), None)
        if not open_spread:
            return None

        raw = open_spread.get("raw_broker_fill") or "{}"
        if isinstance(raw, str):
            try:
                raw_payload = json.loads(raw)
            except json.JSONDecodeError:
                raw_payload = {}
        elif isinstance(raw, dict):
            raw_payload = raw
        else:
            raw_payload = {}

        short_sym = raw_payload.get("short_leg", "")
        long_sym = raw_payload.get("long_leg", "")
        if not short_sym or not long_sym:
            symbol = pos.get("symbol", "")
            if symbol.startswith("BPCS:") and "/" in symbol:
                pair = symbol.removeprefix("BPCS:")
                short_sym, long_sym = pair.split("/", 1)
        if not short_sym or not long_sym:
            return None

        expiry = self._occ_expiry(short_sym)
        short_snap = self.broker.get_option_snapshot(short_sym, expiry=expiry) or {}
        long_snap = self.broker.get_option_snapshot(long_sym, expiry=expiry) or {}
        short_ask = float(short_snap.get("ask") or 0.0) if isinstance(short_snap, dict) else 0.0
        long_bid = float(long_snap.get("bid") or 0.0) if isinstance(long_snap, dict) else 0.0
        if short_ask <= 0 or long_bid <= 0:
            # No fresh executable quote; fail closed by doing nothing.
            return None

        close_debit = max(short_ask - long_bid, 0.01)
        entry_credit = float(pos.get("avg_entry_price") or open_spread.get("price") or 0.0)
        if entry_credit <= 0:
            return None

        # 1) Profit target: capture configured % of received credit.
        target_debit = entry_credit * (1.0 - self.settings.bpcs_profit_target_pct)
        if close_debit <= target_debit:
            return ("bpcs_tp", short_sym, long_sym, short_ask, long_bid, entry_credit)

        # 2) Risk stop: debit has doubled vs entry credit.
        if close_debit >= entry_credit * 2.0:
            return ("bpcs_sl", short_sym, long_sym, short_ask, long_bid, entry_credit)

        # 3) DTE exit to avoid gamma risk.
        exp_date = self._occ_expiry_date(short_sym)
        if exp_date is not None:
            dte = (exp_date - date.today()).days
            if dte <= self.settings.bpcs_close_at_dte:
                return ("bpcs_dte_exit", short_sym, long_sym, short_ask, long_bid, entry_credit)

        return None

    @staticmethod
    def _occ_expiry(contract_symbol: str) -> str | None:
        exp = PositionMonitor._occ_expiry_date(contract_symbol)
        return exp.isoformat() if exp else None

    @staticmethod
    def _occ_expiry_date(contract_symbol: str) -> date | None:
        # Root length can vary. The OCC date is the 6 digits immediately before C/P + strike.
        import re
        m = re.match(r"^[A-Z]+(\d{6})[CP]\d{8}$", contract_symbol)
        if not m:
            return None
        yymmdd = m.group(1)
        try:
            return date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
        except ValueError:
            return None

    def _exit_price(self, trigger: str, mark: float, avg_entry: float) -> float:
        """Compute the limit price for the close order.

        Rule: **never exit below the bid**. For protective exits (SL, EOD)
        we accept mark*0.99 floor; for gain-capturing exits (TP,
        trailing_floor) we set the limit at mark so a paper fill at the
        bid still captures the gain.

        Gain-protection order of priority:
        1. Trailing floor / TP partial — exit at mark (paper fills at bid,
           but the bid IS our floor).
        2. TP full — exit at mark (we're capturing the full move, no
           compromise on price).
        3. SL — exit at bid or 97% of mark (whichever higher). Don't
           leave money on the table, but accept the slip.
        4. EOD — exit at mark (we want out, no haggling).
        """
        if trigger in ("tp", "tp_partial"):
            # Capture the gain; limit at mark so we don't leave money on
            # the table.
            return max(mark, 0.05)
        if trigger == "trailing_floor":
            # Floor based on mark — the bid is implicit floor.
            return max(mark, 0.05)
        if trigger == "sl":
            # Use mark (or slightly below) — the bracket stop will fire
            # if mark hits. Don't go below bid though.
            return max(mark * 0.97, 0.05)
        if trigger == "eod":
            # EOD: get out, accept slip — but still no lower than mark.
            return max(mark * 0.98, 0.05)
        # Default
        return max(mark, 0.05)

    def _update_peak(self, position_id: str, new_peak: float) -> None:
        """Update positions.peak_mark if `new_peak` is higher."""
        if not position_id:
            return
        try:
            with sqlite3.connect(self.journal.db_path) as conn:
                conn.execute(
                    "UPDATE positions SET peak_mark = ? "
                    "WHERE position_id = ? AND (peak_mark IS NULL OR peak_mark < ?)",
                    (new_peak, position_id, new_peak),
                )
        except Exception as e:
            log.debug("peak update failed for %s: %s", position_id, e)
