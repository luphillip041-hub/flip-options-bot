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

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC

from ..broker import BrokerClient
from ..config import Settings
from ..journal import Journal, TradeEvent
from ..risk import RiskEngine, RiskState
from ..strategies.bull_put_credit import BullPutSpreadSignal
from ..strategies.long_call import LongCallSignal
from ..strategies.long_equity import LongEquitySignal
from ..strategies.long_put import LongPutSignal

log = logging.getLogger("flip_options_bot.executor")


@dataclass
class ExecutionResult:
    accepted: bool
    client_order_id: str = ""
    position_id: str = ""
    reason: str = ""


class Executor:
    """One instance per daemon. Stateless; holds references to broker, journal, risk."""

    def __init__(
        self, settings: Settings, broker: BrokerClient, journal: Journal, risk: RiskEngine
    ):
        self.settings = settings
        self.broker = broker
        self.journal = journal
        self.risk = risk

    def submit_long_option(
        self, signal: LongCallSignal | LongPutSignal, equity: float, state: RiskState
    ) -> ExecutionResult:
        """Submit one long directional option signal (call or put). Risk gate runs first.

        If `state.is_live()` is True and the executor has not been configured
        to confirm live trading, abort. (Double-gate lives in Settings.is_live()
        AND in `confirm_live` which is operator-supplied.)
        """
        if signal.strategy_id not in {"long_call", "long_put"}:
            return ExecutionResult(accepted=False, reason=f"unknown strategy {signal.strategy_id}")
        underlying = self._occ_underlying(signal.symbol)
        if underlying and self._has_open_directional_underlying(underlying):
            reason = f"duplicate_directional_underlying:{underlying}"
            log.info("risk denied %s: %s", signal.symbol, reason)
            return ExecutionResult(accepted=False, reason=reason)

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
            unsupported = "complex orders not supported" in err or "bracket" in err or "oco" in err
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
                "strategy_id": signal.strategy_id,
                "option_type": signal.option_type,
                "expiry": signal.expiry,
                "strike": signal.strike,
                "dte": signal.dte,
                "conviction": signal.conviction,
                "submitted_limit_price": signal.limit_price,
                "target_otm_pct": (
                    self.settings.long_call_target_otm_pct
                    if signal.strategy_id == "long_call"
                    else self.settings.long_put_target_otm_pct
                ),
                "high_reward_mode": self.settings.long_option_high_reward_mode,
                "otm_ladder_pct": list(self.settings.long_option_otm_ladder_pct),
                "signal_notes": signal.notes,
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
            debit=signal.limit_price * 100 * decision.contracts,
            event_id=coid,
        )

        log.info(
            "submitted %s qty=%d coid=%s pos=%s tp=%s sl=%s",
            signal.symbol,
            decision.contracts,
            coid,
            position_id,
            tp_oid or "(none)",
            sl_oid or "(none)",
        )
        return ExecutionResult(
            accepted=True,
            client_order_id=coid,
            position_id=position_id,
        )

    def submit_long_call(
        self, signal: LongCallSignal, equity: float, state: RiskState
    ) -> ExecutionResult:
        """Backward-compatible wrapper for callers/tests that only know long_call."""
        return self.submit_long_option(signal, equity=equity, state=state)

    def submit_long_put(
        self, signal: LongPutSignal, equity: float, state: RiskState
    ) -> ExecutionResult:
        return self.submit_long_option(signal, equity=equity, state=state)

    def submit_long_equity(
        self, signal: LongEquitySignal, equity: float, state: RiskState
    ) -> ExecutionResult:
        """Submit one bullish long-equity fallback signal with limit orders only."""
        if signal.strategy_id != "long_equity":
            return ExecutionResult(accepted=False, reason=f"unknown strategy {signal.strategy_id}")
        notional = signal.limit_price * signal.qty
        stop_risk = max(signal.limit_price - signal.stop_price, 0.0) * signal.qty
        if notional > self.settings.long_equity_max_position_dollar:
            return ExecutionResult(
                accepted=False,
                reason=f"long_equity_notional: {notional:.2f} > {self.settings.long_equity_max_position_dollar:.2f}",
            )
        decision = self.risk.evaluate_pre_trade_stock(state, equity=equity, proposed_risk=stop_risk)
        if not decision.allowed:
            log.info("risk denied long_equity %s: %s", signal.symbol, decision.reason)
            return ExecutionResult(accepted=False, reason=decision.reason)
        if self.settings.is_live() and not self._confirm_live(signal):
            log.warning("live mode but confirm_live denied; aborting long_equity %s", signal.symbol)
            return ExecutionResult(accepted=False, reason="live_confirm_denied")

        position_id = Journal.new_position_id()
        coid = f"long-equity-{uuid.uuid4()}"
        try:
            order = self.broker.submit_stock_buy(
                symbol=signal.symbol,
                qty=signal.qty,
                limit_price=signal.limit_price,
                client_order_id=coid,
                position_id=position_id,
            )
        except Exception as e:
            log.error("broker submit_stock_buy failed for %s: %s", signal.symbol, e)
            return ExecutionResult(accepted=False, reason=f"broker_error: {e}")

        event = TradeEvent(
            event_id=coid,
            ts=Journal.now_iso(),
            kind="open",
            symbol=signal.symbol,
            side="buy",
            qty=signal.qty,
            price=signal.limit_price,
            position_id=position_id,
            strategy_id=signal.strategy_id,
            raw_broker_fill={
                "order_id": str(order.id),
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
                "stop_price": signal.stop_price,
                "take_profit_price": signal.take_profit_price,
                "notional": notional,
                "stop_risk": stop_risk,
            },
        )
        self.journal.append(event)
        self.journal.set_bracket(
            position_id=position_id,
            tp_order_id=None,
            sl_order_id=None,
            tp_price=signal.take_profit_price,
            sl_trigger_price=signal.stop_price,
            sl_limit_price=round(signal.stop_price * 0.999, 2),
        )
        self.risk.record_open(
            state,
            symbol=signal.symbol,
            debit=stop_risk,
            event_id=coid,
        )
        log.info(
            "submitted long_equity %s qty=%d coid=%s pos=%s entry=%.2f tp=%.2f stop=%.2f",
            signal.symbol,
            signal.qty,
            coid,
            position_id,
            signal.limit_price,
            signal.take_profit_price,
            signal.stop_price,
        )
        return ExecutionResult(accepted=True, client_order_id=coid, position_id=position_id)

    def submit_bull_put_spread(
        self,
        signal: BullPutSpreadSignal,
        equity: float,
        state: RiskState,
        short_put_symbol: str,
        long_put_symbol: str,
    ) -> ExecutionResult:
        """Submit a 2-leg bull put credit spread.

        Workflow:
          1. Risk gate: check max_loss (not debit!) against bpcs_max_loss_*
          2. Submit SHORT PUT sell-to-open at ask
          3. Submit LONG PUT buy-to-open at ask
          4. Write 2 journal events (open_sell + open_buy) linked by position_id
          5. Record open in risk engine with the MAX LOSS (not debit!)

        Both legs are submitted with GTC (BPCS targets 25-50 DTE).
        """
        if signal.strategy_id != "bull_put_credit_spread":
            return ExecutionResult(accepted=False, reason=f"unknown strategy {signal.strategy_id}")

        underlying = self._occ_underlying(short_put_symbol)
        if underlying and self._has_open_bpcs_underlying(underlying):
            reason = f"duplicate_bpcs_underlying:{underlying}"
            log.info("risk denied BPCS %s/%s: %s", short_put_symbol, long_put_symbol, reason)
            return ExecutionResult(accepted=False, reason=reason)

        # === Risk gate: check max_loss against bpcs_max_loss_* ===
        decision = self.risk.evaluate_pre_trade_spread(
            state,
            equity=equity,
            max_loss=signal.max_loss_per_contract,
        )
        if not decision.allowed:
            log.info(
                "risk denied BPCS %s/%s: %s", short_put_symbol, long_put_symbol, decision.reason
            )
            return ExecutionResult(accepted=False, reason=decision.reason)

        # === Live mode double-check ===
        if self.settings.is_live() and not self._confirm_live(signal):
            log.warning("live mode but confirm_live denied; aborting BPCS")
            return ExecutionResult(accepted=False, reason="live_confirm_denied")

        # === Generate IDs ===
        position_id = Journal.new_position_id()
        spread_coid = f"bpcs-spread-{uuid.uuid4()}"

        # === Submit the credit spread as a single MLEG order ===
        # Using multi-leg (MLEG) is the proper way to submit a spread —
        # Alpaca applies SPREAD margin rules (max loss only) instead of
        # cash-secured requirements on each leg. Submitting two separate
        # orders would treat each as a naked put and require 100x strike
        # in buying power per leg.
        spread_order = None
        try:
            spread_order = self.broker.submit_credit_spread(
                short_put_symbol=short_put_symbol,
                long_put_symbol=long_put_symbol,
                short_put_limit=signal.short_strike_price_estimate,
                long_put_limit=signal.long_strike_price_estimate,
                qty=decision.contracts,
                client_order_id=spread_coid,
                position_id=position_id,
            )
        except Exception as e:
            log.error("BPCS MLEG submit failed for %s/%s: %s", short_put_symbol, long_put_symbol, e)
            return ExecutionResult(accepted=False, reason=f"broker_error_mleg: {e}")

        # === Write journal event for the spread order ===
        # We record one event for the spread (filled as one instrument).
        # The audit trail can trace back to the MLEG via raw_broker_fill.
        spread_event = TradeEvent(
            event_id=spread_coid,
            ts=Journal.now_iso(),
            kind="open_spread",
            symbol=f"BPCS:{short_put_symbol}/{long_put_symbol}",
            side="sell",  # net side = credit received
            qty=decision.contracts,
            price=signal.credit_estimate,
            position_id=position_id,
            strategy_id=signal.strategy_id,
            raw_broker_fill={
                "order_id": str(spread_order.id),
                "status": str(spread_order.status),
                "order_class": "MLEG",
                "short_leg": short_put_symbol,
                "long_leg": long_put_symbol,
                "submitted_at": str(spread_order.submitted_at),
            },
        )
        self.journal.append(spread_event)

        # === Record in risk engine (track max_loss not debit!) ===
        self.risk.record_open_spread(
            state,
            symbol=f"BPCS:{short_put_symbol}/{long_put_symbol}",
            max_loss=signal.max_loss_per_contract,
            event_id=spread_coid,
        )

        log.info(
            "BPCS submitted %s/%s qty=%d pos=%s coid=%s max_loss=$%.0f credit=$%.2f status=%s",
            short_put_symbol,
            long_put_symbol,
            decision.contracts,
            position_id,
            spread_coid,
            signal.max_loss_per_contract,
            signal.credit_estimate,
            spread_order.status,
        )
        return ExecutionResult(
            accepted=True,
            client_order_id=spread_coid,
            position_id=position_id,
        )

    @staticmethod
    def _occ_underlying(contract_symbol: str) -> str:
        match = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", contract_symbol or "")
        return match.group(1) if match else ""

    def _has_open_directional_underlying(self, underlying: str) -> bool:
        """Avoid stacking same-underlying long premium positions.

        Long calls and long puts are high-gamma 0DTE exposures. One open
        directional option per underlying keeps the data clean and prevents
        accidental same-symbol lottery stacking.
        """
        for pos in self.journal.get_all_positions():
            if pos.get("strategy_id") not in {"long_call", "long_put"}:
                continue
            if pos.get("state") not in ("open", "partial"):
                continue
            pos_underlying = self._occ_underlying(str(pos.get("symbol") or ""))
            if pos_underlying == underlying:
                return True
        return False

    def _has_open_bpcs_underlying(self, underlying: str) -> bool:
        """One logical credit-spread exposure per underlying at a time.

        A submitted close_spread is only a placeholder. Treat the exposure as
        active until reconciliation marks that close with fill_source=broker.
        """
        prefix = f"BPCS:{underlying}"
        for pos in self.journal.get_all_positions():
            if pos.get("strategy_id") != "bull_put_credit_spread" or not str(
                pos.get("symbol") or ""
            ).startswith(prefix):
                continue
            if pos.get("state") in ("open", "partial"):
                return True
            closes = [
                event
                for event in self.journal.get_legs_for_position(pos["position_id"])
                if event.get("kind") == "close_spread"
            ]
            if not closes:
                return True
            raw = closes[-1].get("raw_broker_fill") or "{}"
            try:
                payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            if payload.get("fill_source") != "broker":
                return True
        return False

    def _confirm_live(self, signal) -> bool:
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
        from datetime import datetime, timedelta

        since = datetime.now(UTC) - timedelta(hours=24)
        filled = self.broker.list_filled_orders(since_ts=since)
        if not filled:
            return 0

        n_written = 0
        for order in filled:
            coid = getattr(order, "client_order_id", None) or ""
            if not coid:
                continue
            existing = self.journal.get_event(coid)
            if existing and existing.get("kind") in ("open", "close", "close_attempt", "open_spread", "close_spread"):
                raw = existing.get("raw_broker_fill") or "{}"
                try:
                    raw_payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw_payload = {}
                fill_price = abs(float(order.filled_avg_price or 0.0))
                if (
                    not math.isfinite(fill_price)
                    or fill_price <= 0
                    or raw_payload.get("fill_source") == "broker"
                ):
                    continue
                raw_payload.update(
                    {
                        "order_id": str(order.id),
                        "status": str(order.status),
                        "filled_at": str(order.filled_at),
                        "filled_avg_price": fill_price,
                        "fill_source": "broker",
                    }
                )
                kind = str(existing["kind"])
                if kind == "close_attempt":
                    kind = "close"
                realized_pnl = float(existing.get("realized_pnl") or 0.0)
                strategy_id = str(existing.get("strategy_id") or "")
                if kind == "close_spread":
                    entry_credit = float(raw_payload.get("entry_credit") or 0.0)
                    realized_pnl = (entry_credit - fill_price) * int(existing.get("qty") or 0) * 100
                    strategy_id = "bull_put_credit_spread"
                elif kind == "close":
                    pos = self.journal.get_position_for_id(str(existing.get("position_id") or ""))
                    avg_entry = float(pos.get("avg_entry_price") or 0.0) if pos else 0.0
                    realized_pnl = (fill_price - avg_entry) * int(existing.get("qty") or 0) * 100
                    if pos and not strategy_id:
                        strategy_id = str(pos.get("strategy_id") or "")

                canonical = TradeEvent(
                    event_id=coid,
                    ts=str(order.filled_at) if order.filled_at else Journal.now_iso(),
                    kind=kind,  # type: ignore[arg-type]
                    symbol=str(existing["symbol"]),
                    side=("buy" if existing["side"] == "buy" else "sell"),
                    qty=int(order.filled_qty or existing.get("qty") or 0),
                    price=fill_price,
                    position_id=str(existing.get("position_id") or ""),
                    realized_pnl=round(realized_pnl, 2),
                    strategy_id=strategy_id,
                    raw_broker_fill=raw_payload,
                )
                self.journal.upsert(canonical)
                n_written += 1
                if canonical.kind in ("close", "close_spread"):
                    state = self.risk.load_state()
                    self.risk.record_close(
                        state,
                        pnl=canonical.realized_pnl,
                        event_id=coid,
                        payload=f"{canonical.symbol} realized={canonical.realized_pnl:.2f}",
                    )
                continue

            # Already recorded unknown fill? Defense in depth.
            if existing:
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
        from datetime import datetime

        opens = self.broker.list_open_orders()
        if not opens:
            return 0

        now = datetime.now(UTC)
        n_cancelled = 0
        for order in opens:
            submitted = getattr(order, "submitted_at", None)
            if submitted is None:
                continue
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=UTC)
            age = (now - submitted).total_seconds()
            if age < older_than_seconds:
                continue

            coid = getattr(order, "client_order_id", "") or ""
            if not coid:
                continue

            event = self.journal.get_event(coid)
            if event:
                kind = str(event.get("kind") or "")
                side = str(getattr(order, "side", "")).upper()
                is_pending_close = kind == "close_attempt" and side.endswith("SELL")
                if not is_pending_close:
                    # Filled/journaled opens must not be cancelled here. A
                    # close_attempt is different: it is an unfilled exit intent,
                    # so canceling it lets the monitor reprice on the next tick.
                    log.debug("stale-order skip %s — journaled kind=%s", coid, kind)
                    continue

            try:
                self.broker.cancel_order(str(order.id))
                log.warning(
                    "cancelled stale order %s (submitted=%s, age=%ds)",
                    coid,
                    submitted,
                    int(age),
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
        import sqlite3

        with sqlite3.connect(self.journal.db_path) as conn:
            rows = conn.execute(
                "SELECT position_id, raw_broker_fill FROM trades "
                "WHERE kind = 'open' AND symbol = ?",
                (symbol,),
            ).fetchall()
        # Fallback: return the most-recent open position for this symbol.
        if rows:
            return rows[-1][0]
        return ""
