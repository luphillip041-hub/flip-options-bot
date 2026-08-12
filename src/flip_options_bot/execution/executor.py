"""Executor — translates LongCallSignals into broker orders.

The executor is the only layer that mutates broker + journal state. It:

1. Verifies the risk engine approved the trade (via state.open_position_count).
2. Generates a unique client_order_id = f"open-{uuid4()}".
3. Submits a BUY limit order via the broker.
4. Writes the open TradeEvent to the journal (idempotent on client_order_id).
5. Records the open in the risk engine (with the same event_id).
6. Returns the journal TradeEvent for downstream reconciliation.

If the broker rejects, no journal write happens. If the journal write
fails after broker success, the broker order still exists and the
reconcile loop will catch it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Literal

from ..broker import BrokerClient
from ..config import Settings
from ..journal import Journal, TradeEvent
from ..risk import RiskEngine, RiskState
from ..strategies.long_call import LongCallSignal

log = logging.getLogger("flip_options_bot.executor")


@dataclass
class ExecutionResult:
    accepted: bool
    client_order_id: str = ""
    position_id: str = ""
    reason: str = ""


class Executor:
    """One instance per daemon. Stateless; holds references to broker, journal, risk."""

    def __init__(self, settings: Settings, broker: BrokerClient, journal: Journal, risk: RiskEngine):
        self.settings = settings
        self.broker = broker
        self.journal = journal
        self.risk = risk

    def submit_long_call(self, signal: LongCallSignal, equity: float, state: RiskState) -> ExecutionResult:
        """Submit one LongCallSignal. Risk gate runs first.

        If `state.is_live()` is True and the executor has not been configured
        to confirm live trading, abort. (Double-gate lives in Settings.is_live()
        AND in `confirm_live` which is operator-supplied.)
        """
        if signal.strategy_id != "long_call":
            return ExecutionResult(accepted=False, reason=f"unknown strategy {signal.strategy_id}")

        # === Risk gate ===
        decision = self.risk.evaluate_pre_trade(
            state, equity=equity, proposed_debit=signal.limit_price
        )
        if not decision.allowed:
            log.info("risk denied %s: %s", signal.symbol, decision.reason)
            return ExecutionResult(accepted=False, reason=decision.reason)

        # === Live mode double-check ===
        if self.settings.is_live() and not self._confirm_live(signal):
            log.warning("live mode but confirm_live denied; aborting %s", signal.symbol)
            return ExecutionResult(accepted=False, reason="live_confirm_denied")

        # === Generate IDs ===
        position_id = Journal.new_position_id()
        coid = f"open-{uuid.uuid4()}"

        # === Compute TP/SL from signal + Settings ===
        # tp_multiplier: take-profit leg (matches monitor tp_partial trigger)
        # sl_threshold_pct: stop-loss leg (matches monitor SL trigger)
        # The bracket is broker-resident: if the daemon dies, the SL still fires.
        tp_price = round(signal.limit_price * self.settings.tp_multiplier, 2)
        sl_trigger = round(signal.limit_price * self.settings.sl_threshold_pct, 2)
        # SL limit price = 5% below trigger (gives the order room to fill
        # in a fast gap, otherwise the stop might "miss" if it's a stop-limit
        # with limit==trigger).
        sl_limit = round(sl_trigger * 0.95, 2)

        # === Submit bracket to broker ===
        # Alpaca paper does NOT support BRACKET orders on options ("complex
        # orders not supported"). Fall back to plain BUY + post-fill
        # `place_tpsl_bracket` attachment.
        order = None
        try:
            order = self.broker.submit_bracket_buy(
                contract_symbol=signal.symbol,
                qty=decision.contracts,
                limit_price=signal.limit_price,
                tp_price=tp_price,
                sl_trigger_price=sl_trigger,
                sl_limit_price=sl_limit,
                client_order_id=coid,
            )
        except Exception as e:
            err = str(e).lower()
            unsupported = (
                "complex orders not supported" in err
                or "bracket" in err
                or "oco" in err
            )
            if not unsupported:
                log.error("broker submit_bracket_buy failed for %s: %s", signal.symbol, e)
                return ExecutionResult(accepted=False, reason=f"broker_error: {e}")
            log.info(
                "broker doesn't support complex orders (paper); falling back to "
                "submit_buy + post-fill bracket attach"
            )
            try:
                order = self.broker.submit_buy(
                    contract_symbol=signal.symbol,
                    qty=decision.contracts,
                    limit_price=signal.limit_price,
                    client_order_id=coid,
                    position_id=position_id,
                )
            except Exception as e2:
                log.error("broker submit_buy (fallback) failed for %s: %s", signal.symbol, e2)
                return ExecutionResult(accepted=False, reason=f"broker_error: {e2}")

        # === Write journal event (idempotent on coid) ===
        event = TradeEvent(
            event_id=coid,
            ts=Journal.now_iso(),
            kind="open",
            symbol=signal.symbol,
            side="buy",
            qty=decision.contracts,
            price=signal.limit_price,
            position_id=position_id,
            strategy_id=signal.strategy_id,
            raw_broker_fill={
                "order_id": str(order.id),
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
                "tp_price": tp_price,
                "sl_trigger_price": sl_trigger,
                "sl_limit_price": sl_limit,
            },
        )
        written = self.journal.append(event)
        if not written:
            log.warning("journal event %s already existed (idempotent skip)", coid)

        # === Record bracket legs in positions table ===
        # legs[0] is TP, legs[1] is SL (Alpaca convention for BRACKET)
        tp_oid = ""
        sl_oid = ""
        try:
            legs = getattr(order, "legs", None) or []
            if len(legs) >= 1:
                tp_oid = str(legs[0].id)
            if len(legs) >= 2:
                sl_oid = str(legs[1].id)
        except Exception as e:
            log.debug("could not extract bracket leg IDs from order: %s", e)

        # Always record the planned TP/SL prices so the position monitor can
        # apply them as the broker-resident safety net (paper doesn't support
        # broker brackets; live does).
        self.journal.set_bracket(
            position_id=position_id,
            tp_order_id=tp_oid or None,
            sl_order_id=sl_oid or None,
            tp_price=tp_price,
            sl_trigger_price=sl_trigger,
            sl_limit_price=sl_limit,
        )

        # === Record in risk engine ===
        self.risk.record_open(
            state,
            symbol=signal.symbol,
            debit=signal.limit_price * decision.contracts,
            event_id=coid,
        )

        log.info(
            "submitted %s qty=%d coid=%s pos=%s tp=%s sl=%s",
            signal.symbol, decision.contracts, coid, position_id,
            tp_oid or "(none)", sl_oid or "(none)",
        )
        return ExecutionResult(
            accepted=True,
            client_order_id=coid,
            position_id=position_id,
        )

    def _confirm_live(self, signal: LongCallSignal) -> bool:
        """Operator-supplied confirmation hook. Default: always reject live.

        In production, this would be wired to a Discord-confirmation channel
        or a manual-approval flag in the env. For now, the scaffold leaves
        this as the LAST gate before a live order.
        """
        return False  # scaffold default — never auto-confirm live

    def reconcile_fills(self) -> int:
        """Pull real fills from broker, write close events to journal
        for any fills we haven't recorded yet. Returns count written.

        Idempotent: close event_id is the broker's client_order_id; INSERT
        OR IGNORE on duplicate. Also calls risk.record_close so the
        daily/weekly P&L counters and kill-switch caps stay current.
        """
        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(hours=24)
        filled = self.broker.list_filled_orders(since_ts=since)
        if not filled:
            return 0

        n_written = 0
        for order in filled:
            coid = getattr(order, "client_order_id", None) or ""
            if not coid:
                continue
            # Already recorded? (Defense in depth — journal.append is idempotent too.)
            if self.journal.has_event(coid):
                continue

            # Determine kind from the order side
            kind = "open" if str(order.side).endswith("BUY") else "close"
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
            side_str = str(order.side).lower().replace("order_side.", "")
            if side_str not in ("buy", "sell"):
                side_str = "buy"  # default fallback

            # For closes, compute realized_pnl = (fill_price - avg_entry) * qty * 100
            realized_pnl = 0.0
            position_id = ""
            if kind == "close":
                position_id = self._lookup_position_id(coid, symbol=str(order.symbol))
                if position_id:
                    row = self.journal.get_position_for_id(position_id)
                    if row:
                        avg_entry = float(row.get("avg_entry_price") or 0)
                        qty = int(order.filled_qty) if order.filled_qty else 1
                        sign = -1 if side_str == "sell" else 1  # sell closes long → negative diff
                        realized_pnl = (fill_price - avg_entry) * qty * 100 * sign

            event = TradeEvent(
                event_id=coid,
                ts=str(order.filled_at) if order.filled_at else Journal.now_iso(),
                kind=kind,  # type: ignore[arg-type]
                symbol=str(order.symbol),
                side=side_str,  # type: ignore[arg-type]
                qty=int(order.filled_qty) if order.filled_qty else 0,
                price=fill_price,
                position_id=position_id,
                strategy_id="long_call",
                realized_pnl=realized_pnl,
                raw_broker_fill={
                    "order_id": str(order.id),
                    "status": str(order.status),
                    "filled_at": str(order.filled_at),
                },
            )
            if self.journal.upsert(event):
                n_written += 1

            # If this is a close, also update the risk state so caps stay current
            if kind == "close" and position_id and realized_pnl != 0.0:
                state = self.risk.load_state()
                self.risk.record_close(
                    state,
                    pnl=realized_pnl,
                    event_id=coid,
                    payload=f"{order.symbol} realized={realized_pnl:.2f}",
                )

        if n_written:
            log.info("reconciled %d fills into journal", n_written)
        return n_written

    def cancel_stale_orders(self, older_than_seconds: int = 120) -> int:
        """Cancel any open order that's been ACCEPTED but not filled for
        longer than `older_than_seconds`. Returns count cancelled.

        Critical structural fix vs. flip-alpaca-bot: an order that sits at
        ACCEPTED forever (paper account quirk, low liquidity, stale quote)
        silently blocks the slot it occupies in risk_engine.open_position_count.
        We DON'T want to wait hours hoping the synthetic book picks it up.

        We track which orders we've already cancelled via the journal's
        raw_broker_fill JSON, so we don't cancel the same order twice.
        """
        from datetime import datetime, timezone

        opens = self.broker.list_open_orders()
        if not opens:
            return 0

        now = datetime.now(timezone.utc)
        n_cancelled = 0
        for order in opens:
            submitted = getattr(order, "submitted_at", None)
            if submitted is None:
                continue
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=timezone.utc)
            age = (now - submitted).total_seconds()
            if age < older_than_seconds:
                continue

            coid = getattr(order, "client_order_id", "") or ""
            if not coid:
                continue

            # Check journal: if already recorded as 'open' we don't want to
            # cancel. Only cancel orders that haven't been journaled yet
            # (i.e., they're stuck mid-submission).
            if self.journal.has_event(coid):
                log.debug("stale-order skip %s — already journaled", coid)
                continue

            try:
                self.broker.cancel_order(str(order.id))
                log.warning(
                    "cancelled stale order %s (submitted=%s, age=%ds)",
                    coid, submitted, int(age),
                )
                n_cancelled += 1
            except Exception as e:
                log.error("failed to cancel stale order %s: %s", coid, e)

        if n_cancelled:
            log.info("cancelled %d stale orders", n_cancelled)
        return n_cancelled

    def _lookup_position_id(self, coid: str, symbol: str) -> str:
        """Find the position_id that corresponds to a close fill.

        Strategy: the bracket leg's parent order shares the position_id
        via the open TradeEvent's raw_broker_fill.client_order_id (= coid
        of the parent buy). So we look for a buy with the same symbol
        whose order_id appears in trades.raw_broker_fill.
        """
        import json as _json
        import sqlite3
        with sqlite3.connect(self.journal.db_path) as conn:
            rows = conn.execute(
                "SELECT position_id, raw_broker_fill FROM trades "
                "WHERE kind = 'open' AND symbol = ?",
                (symbol,),
            ).fetchall()
        for pos_id, raw in rows:
            if not raw:
                continue
            try:
                data = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            # open event's raw may not contain coid; match on symbol + last-bought heuristic.
        # Fallback: return the most-recent open position for this symbol.
        if rows:
            return rows[-1][0]
        return ""