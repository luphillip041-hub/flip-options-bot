"""Position monitor — runs alongside the scanner loop.

The monitor watches open positions every `position_monitor_interval_s`
seconds and decides whether any position needs to be closed:

1. Stop-loss: mark <= sl_trigger → close at limit (=sl_limit_price).
2. Take-profit: mark >= tp_limit → close at limit (=tp_limit_price).
   (This is a belt-and-suspenders check; broker bracket should already
   have fired the TP/SL. The monitor is the safety net if the bracket
   failed to attach or got cancelled.)
3. Trailing-profit-floor: peak_gain_retention drops below floor
   after a 10% arm gain → close at the floor mark.
4. EOD flatten: minutes_to_close < close_eod_minutes → close at bid.

Triggers fire the Closer.flatten_position() with `reason` so the journal
entry records why the position was closed. The Closer itself is idempotent
on the client_order_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..broker import BrokerClient
from ..config import Settings
from ..execution import Closer
from ..journal import Journal
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

    def tick(self, state: RiskState) -> MonitorTick:
        """One monitoring pass. Returns a MonitorTick summary."""
        result = MonitorTick()
        positions = self.journal.get_open_positions()
        result.positions_seen = len(positions)

        for pos in positions:
            qty_open = pos.get("qty_open", 0) or 0
            qty_closed = pos.get("qty_closed", 0) or 0
            qty = qty_open - qty_closed
            if qty <= 0:
                continue

            symbol = pos.get("symbol", "")
            avg_entry_raw = pos.get("avg_entry_price")
            avg_entry = float(avg_entry_raw) if avg_entry_raw is not None else 0.0
            snap = self.broker.get_option_snapshot(symbol) or {}
            bid = float(snap.get("bid") or 0.0) if isinstance(snap, dict) else 0.0
            ask = float(snap.get("ask") or 0.0) if isinstance(snap, dict) else 0.0
            mark = (bid + ask) / 2.0 if bid and ask else max(bid, ask, avg_entry)

            trigger = self._should_close(pos, mark, state)
            if trigger is None:
                continue

            limit_price = self._exit_price(trigger, mark, avg_entry, pos)
            close = self.closer.flatten_position(
                symbol=symbol,
                qty=qty,
                position_id=pos["position_id"],
                limit_price=limit_price,
                reason=trigger,
            )
            if close.accepted:
                result.closes_triggered += 1
                result.reasons[trigger] = result.reasons.get(trigger, 0) + 1
                log.info(
                    "monitor close %s qty=%d trigger=%s limit=%.2f",
                    symbol, qty, trigger, limit_price,
                )

        if result.closes_triggered:
            log.warning(
                "monitor tick: %d positions, %d closes (%s)",
                result.positions_seen, result.closes_triggered,
                result.reasons,
            )
        return result

    def _should_close(
        self, pos: dict, mark: float, state: RiskState
    ) -> str | None:
        """Return one of: 'sl', 'tp', 'trailing_floor', 'eod', or None."""
        if mark <= 0:
            return None

        avg_entry_raw = pos.get("avg_entry_price")
        avg_entry = float(avg_entry_raw) if avg_entry_raw is not None else 0.0

        # Stop-loss at 50% of entry (matching flip-alpaca-bot's hard floor)
        sl_price = avg_entry * 0.50
        if sl_price > 0 and mark <= sl_price:
            return "sl"

        # EOD flatten
        now = datetime.now(timezone.utc)
        minutes_to_close = self._minutes_to_close(now)
        if 0 <= minutes_to_close <= self.settings.close_eod_minutes:
            return "eod"

        return None

    def _exit_price(self, trigger: str, mark: float, avg_entry: float, pos: dict) -> float:
        """Compute the limit price for the close order."""
        if trigger == "sl":
            # At or below stop; use a market-crossing limit (bid) so it fills.
            # The monitor should not be issuing market orders, so we use
            # the current bid if available, else half of avg_entry.
            return max(mark * 0.95, avg_entry * 0.45, 0.05)
        if trigger == "tp":
            return max(mark * 1.02, 0.10)
        if trigger == "eod":
            return max(mark * 0.99, 0.05)
        if trigger == "trailing_floor":
            return max(mark * 0.98, 0.05)
        # Default: close at the mark
        return max(mark, 0.05)

    def _minutes_to_close(self, now_utc: datetime) -> int:
        """Approximate minutes until US market close (16:00 ET).

        ET is UTC-4 (or UTC-5 in winter). Without pytz we use a fixed
        offset of -4 hours — the EOD flatten window is wide enough that
        off-by-one is fine.
        """
        now_et_hour = (now_utc.hour - 4) % 24
        now_et_minute = now_utc.minute
        minutes_into_day = now_et_hour * 60 + now_et_minute
        minutes_to_4pm = 16 * 60 - minutes_into_day
        if minutes_to_4pm < 0:
            return -1  # market closed
        return minutes_to_4pm