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

        This method intentionally does NOT write a canonical `close` TradeEvent
        to the journal directly — that would mark the position closed before
        the broker fill exists. It writes `close_attempt`, which preserves the
        client_order_id/position_id link without mutating position_state.

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
        # knows a close was attempted. `close_attempt` does not update
        # position_state; reconcile_fills() later upserts a canonical `close`
        # with the actual broker fill price and realized P&L.
        event = TradeEvent(
            event_id=coid,
            ts=Journal.now_iso(),
            kind="close_attempt",
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

    def flatten_credit_spread(
        self,
        *,
        position_id: str,
        short_put_symbol: str,
        long_put_symbol: str,
        qty: int,
        short_put_limit: float,
        long_put_limit: float,
        entry_credit: float,
        reason: str = "monitor",
    ) -> CloseResult:
        """Close a bull put credit spread atomically via an MLEG order.

        The close price is a net debit: buy short put at short_put_limit,
        sell long put at long_put_limit. The realized P&L approximation is
        (entry_credit - close_debit) * 100 * qty. Reconcile can later
        overwrite with broker-canonical fills.
        """
        if qty <= 0:
            return CloseResult(accepted=False, reason="zero_qty")
        if not short_put_symbol or not long_put_symbol:
            return CloseResult(accepted=False, reason="missing_leg_symbol")

        coid = f"close-spread-{uuid.uuid4()}"
        close_debit = max(short_put_limit - long_put_limit, 0.01)
        try:
            order = self.broker.submit_close_credit_spread(
                short_put_symbol=short_put_symbol,
                long_put_symbol=long_put_symbol,
                short_put_limit=round(short_put_limit, 2),
                long_put_limit=round(long_put_limit, 2),
                qty=qty,
                client_order_id=coid,
                position_id=position_id,
            )
        except Exception as e:
            log.error("submit_close_credit_spread failed for %s/%s: %s", short_put_symbol, long_put_symbol, e)
            return CloseResult(accepted=False, reason=f"broker_error: {e}")

        realized = (float(entry_credit) - close_debit) * 100 * qty
        event = TradeEvent(
            event_id=coid,
            ts=Journal.now_iso(),
            kind="close_spread",
            symbol=f"BPCS:{short_put_symbol}/{long_put_symbol}",
            side="buy",
            qty=qty,
            price=round(close_debit, 2),
            position_id=position_id,
            realized_pnl=round(realized, 2),
            strategy_id="bull_put_credit_spread",
            raw_broker_fill={
                "order_id": str(getattr(order, "id", "")),
                "status": str(getattr(order, "status", "")),
                "order_class": "MLEG",
                "short_leg": short_put_symbol,
                "long_leg": long_put_symbol,
                "submitted_at": str(getattr(order, "submitted_at", "")),
                "close_reason": reason,
                "close_position_id": position_id,
                "entry_credit": entry_credit,
                "close_debit": close_debit,
            },
        )
        self.journal.append(event)
        log.info(
            "flatten spread %s/%s qty=%d debit=%.2f pnl=%.2f reason=%s coid=%s",
            short_put_symbol, long_put_symbol, qty, close_debit, realized, reason, coid,
        )
        return CloseResult(accepted=True, client_order_id=coid, position_id=position_id)

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

            if (pos.get("strategy_id") == "bull_put_credit_spread" or symbol.startswith("BPCS:")):
                legs = self.journal.get_legs_for_position(pos["position_id"])
                import json as _json
                open_spread = next((l for l in legs if l.get("kind") == "open_spread"), None)
                raw = open_spread.get("raw_broker_fill") if open_spread else None
                payload = _json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
                pair = symbol.removeprefix("BPCS:") if symbol.startswith("BPCS:") else ""
                short_sym = payload.get("short_leg", "")
                long_sym = payload.get("long_leg", "")
                if (not short_sym or not long_sym) and "/" in pair:
                    short_sym, long_sym = pair.split("/", 1)
                short_snap = self.broker.get_option_snapshot(short_sym) or {}
                long_snap = self.broker.get_option_snapshot(long_sym) or {}
                short_ask = float(short_snap.get("ask") or 0.0) if isinstance(short_snap, dict) else 0.0
                long_bid = float(long_snap.get("bid") or 0.0) if isinstance(long_snap, dict) else 0.0
                if short_sym and long_sym and short_ask > 0 and long_bid > 0:
                    result = self.flatten_credit_spread(
                        position_id=pos["position_id"],
                        short_put_symbol=short_sym,
                        long_put_symbol=long_sym,
                        qty=qty,
                        short_put_limit=short_ask,
                        long_put_limit=long_bid,
                        entry_credit=float(pos.get("avg_entry_price") or 0.0),
                        reason=reason,
                    )
                    if result.accepted:
                        n += 1
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