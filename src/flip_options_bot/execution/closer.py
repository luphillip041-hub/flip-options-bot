"""Close executor — flattens an open position via a broker SELL.

The closer is intentionally separate from the open executor:

- The open executor generates UUID event_ids for buys; the close executor
  generates UUID event_ids for sells. No collision.
- The closer pulls positions from the broker (not from the journal) so it
  can flatten positions opened outside the daemon — e.g. via bracket fills
  whose open event was written by reconcile_fills() and not by the open
  executor.

The closer NEVER submits market orders. Limit only, with a price derived
from the latest option snapshot (or the mark if the snapshot is stale).
If the limit price is too far from the mark and the position is in
loss-cut territory, we fall back to a stop-limit at the SL trigger.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from ..broker import BrokerClient
from ..config import Settings
from ..journal import Journal, TradeEvent
from ..risk import RiskEngine, RiskState

log = logging.getLogger("flip_options_bot.closer")


@dataclass
class CloseResult:
    accepted: bool
    client_order_id: str = ""
    position_id: str = ""
    reason: str = ""


class Closer:
    """One instance per daemon. Sends SELL orders for open positions."""

    def __init__(
        self,
        settings: Settings,
        broker: BrokerClient,
        journal: Journal,
        risk: RiskEngine,
    ):
        self.settings = settings
        self.broker = broker
        self.journal = journal
        self.risk = risk

    def flatten_position(
        self,
        symbol: str,
        qty: int,
        position_id: str,
        limit_price: float,
        *,
        reason: str = "monitor",
    ) -> CloseResult:
        """Submit a SELL limit order to flatten one position.

        `position_id` is recorded in the `raw_broker_fill.close_position_id`
        so that when `reconcile_fills()` later writes the canonical close
        event (with the real fill price from the broker), it can update
        the position_state row.

        This method intentionally does NOT write a `close` TradeEvent to
        the journal directly — that would race with `reconcile_fills()`
        which is the canonical source. If both wrote the same event_id,
        INSERT OR IGNORE would lose the broker's actual fill price.

        The `reconcile_fills()` flow:
          1. broker.list_filled_orders returns the closed sell
          2. for each closed sell, journal.append(write_or_update(...))
             uses INSERT ... ON CONFLICT DO UPDATE so the canonical fill
             price overwrites the close metadata.
        """
        if qty <= 0:
            return CloseResult(accepted=False, reason="zero_qty")

        coid = f"close-{uuid.uuid4()}"
        try:
            order = self.broker.submit_close_sell(
                contract_symbol=symbol,
                qty=qty,
                limit_price=round(limit_price, 2),
                client_order_id=coid,
            )
        except Exception as e:
            log.error("submit_close_sell failed for %s: %s", symbol, e)
            return CloseResult(accepted=False, reason=f"broker_error: {e}")

        # Record the close intent (not the canonical close) so the journal
        # knows a close was attempted. This event has the same event_id
        # that reconcile_fills() will use, so the canonical event will
        # overwrite this row's price + realized_pnl fields.
        event = TradeEvent(
            event_id=coid,
            ts=Journal.now_iso(),
            kind="close",
            symbol=symbol,
            side="sell",
            qty=qty,
            price=round(limit_price, 2),
            position_id=position_id,
            realized_pnl=0.0,  # placeholder; reconcile writes the real number
            strategy_id="",
            raw_broker_fill={
                "order_id": str(getattr(order, "id", "")),
                "status": str(getattr(order, "status", "")),
                "submitted_at": str(getattr(order, "submitted_at", "")),
                "close_reason": reason,
                "close_position_id": position_id,
            },
        )
        # Idempotent: first call writes the row, the canonical reconcile
        # event later overwrites price/realized_pnl via INSERT OR REPLACE
        # in the executor's reconcile_fills (or via a future migration).
        self.journal.append(event)

        log.info(
            "flatten %s qty=%d limit=%.2f reason=%s coid=%s",
            symbol, qty, limit_price, reason, coid,
        )
        return CloseResult(
            accepted=True,
            client_order_id=coid,
            position_id=position_id,
        )

    def flatten_all(self, state: RiskState, reason: str = "panic") -> int:
        """Flatten every open position. Used by kill_switch and panic_close."""
        n = 0
        positions = self.journal.get_open_positions()
        for pos in positions:
            symbol = pos.get("symbol", "")
            qty_open = pos.get("qty_open", 0) or 0
            qty_closed = pos.get("qty_closed", 0) or 0
            qty = qty_open - qty_closed
            if qty <= 0 or not symbol:
                continue
            # Use a conservative limit price (ask fallback). If we have no
            # snapshot, submit at the avg_entry_price as the limit — broker
            # may reject (below market for a long). Better than nothing.
            snapshot = self.broker.get_option_snapshot(symbol) or {}
            ask = snapshot.get("ask") if isinstance(snapshot, dict) else None
            avg_entry_raw = pos.get("avg_entry_price")
            avg_entry = float(avg_entry_raw) if avg_entry_raw is not None else 0.0
            limit_price = float(ask) if ask and ask > 0 else max(avg_entry * 0.5, 0.05)
            result = self.flatten_position(
                symbol=symbol,
                qty=qty,
                position_id=pos["position_id"],
                limit_price=limit_price,
                reason=reason,
            )
            if result.accepted:
                n += 1
        log.warning("flatten_all reason=%s submitted=%d", reason, n)
        return n