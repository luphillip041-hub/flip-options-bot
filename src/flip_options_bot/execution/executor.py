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

        # === Submit to broker ===
        try:
            order = self.broker.submit_buy(
                contract_symbol=signal.symbol,
                qty=decision.contracts,
                limit_price=signal.limit_price,
                client_order_id=coid,
                position_id=position_id,
            )
        except Exception as e:
            log.error("broker submit_buy failed for %s: %s", signal.symbol, e)
            return ExecutionResult(accepted=False, reason=f"broker_error: {e}")

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
            },
        )
        written = self.journal.append(event)
        if not written:
            log.warning("journal event %s already existed (idempotent skip)", coid)

        # === Record in risk engine ===
        self.risk.record_open(
            state,
            symbol=signal.symbol,
            debit=signal.limit_price * decision.contracts,
            event_id=coid,
        )

        log.info("submitted %s qty=%d coid=%s pos=%s",
                 signal.symbol, decision.contracts, coid, position_id)
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
        OR IGNORE on duplicate.
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
            event = TradeEvent(
                event_id=coid,
                ts=str(order.filled_at) if order.filled_at else Journal.now_iso(),
                kind=kind,  # type: ignore[arg-type]
                symbol=str(order.symbol),
                side=side_str,  # type: ignore[arg-type]
                qty=int(order.filled_qty) if order.filled_qty else 0,
                price=fill_price,
                position_id="",  # not propagated via broker; reconciled separately
                strategy_id="long_call",
                raw_broker_fill={
                    "order_id": str(order.id),
                    "status": str(order.status),
                    "filled_at": str(order.filled_at),
                },
            )
            if self.journal.append(event):
                n_written += 1
        if n_written:
            log.info("reconciled %d fills into journal", n_written)
        return n_written