"""Position monitor — runs alongside the scanner loop.

The monitor watches open positions every `position_monitor_interval_s`
seconds and decides whether any position needs to be closed:

1. **Stop-loss (sl)**: mark <= entry * sl_threshold_pct (default 50%)
2. **Take-profit (tp)**: mark >= entry * tp_multiplier (default 1.50,
   i.e. +50% gain) — a safety net; broker bracket should fire first
3. **Trailing profit floor (trailing_floor)**: position gained >=
   arm_pct, then mark drops below peak_gain * retention_pct → close
   (default arm=10%, retention=50%). Mirrors the flip-alpaca-bot frozen
   knob `FLIP_TRAILING_RETENTION=0.50`.
4. **EOD flatten (eod)**: minutes_to_close < close_eod_minutes (default 15).

Triggers fire `Closer.flatten_position(reason=...)` so the journal
records why. The Closer itself is idempotent on client_order_id.

Peak gain tracking lives in journal.positions.peak_mark (set at entry,
updated each tick).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

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
        self.trailing_arm_pct = getattr(settings, "trailing_arm_pct", 0.10)
        self.trailing_retention = getattr(settings, "trailing_retention", 0.50)

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
            position_id = pos.get("position_id", "")
            avg_entry_raw = pos.get("avg_entry_price")
            avg_entry = float(avg_entry_raw) if avg_entry_raw is not None else 0.0
            peak_mark_raw = pos.get("peak_mark")
            peak_mark = (
                float(peak_mark_raw)
                if peak_mark_raw is not None
                else max(avg_entry, 0.0)
            )

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

            trigger = self._should_close(avg_entry, peak_mark, mark, state)
            if trigger is None:
                continue

            limit_price = self._exit_price(trigger, mark, avg_entry)
            close = self.closer.flatten_position(
                symbol=symbol,
                qty=qty,
                position_id=position_id,
                limit_price=limit_price,
                reason=trigger,
            )
            if close.accepted:
                result.closes_triggered += 1
                result.reasons[trigger] = result.reasons.get(trigger, 0) + 1
                log.info(
                    "monitor close %s qty=%d trigger=%s mark=%.2f limit=%.2f",
                    symbol, qty, trigger, mark, limit_price,
                )

        if result.closes_triggered:
            log.warning(
                "monitor tick: %d positions, %d closes (%s)",
                result.positions_seen, result.closes_triggered, result.reasons,
            )
        return result

    def _should_close(
        self, avg_entry: float, peak_mark: float, mark: float, state: RiskState
    ) -> str | None:
        """Return one of: 'sl', 'tp', 'trailing_floor', 'eod', or None."""
        if mark <= 0 or avg_entry <= 0:
            return None

        # Stop-loss at sl_threshold_pct of entry (default 50%)
        if mark <= avg_entry * self.sl_threshold_pct:
            return "sl"

        # Take-profit at tp_multiplier of entry (default +50%)
        if mark >= avg_entry * self.tp_multiplier:
            return "tp"

        # Trailing profit floor — only after a peak gain of trailing_arm_pct
        gain_pct = (peak_mark - avg_entry) / avg_entry
        if gain_pct >= self.trailing_arm_pct:
            retention_target = peak_mark * self.trailing_retention
            if mark <= retention_target:
                return "trailing_floor"

        # EOD flatten — only on a weekday, in the last `eod_minutes`
        now = now_utc()
        if is_weekday(now):
            minutes_left = minutes_to_close(now)
            if 0 <= minutes_left <= self.eod_minutes:
                return "eod"

        return None

    def _exit_price(self, trigger: str, mark: float, avg_entry: float) -> float:
        """Compute the limit price for the close order.

        The monitor NEVER submits market orders. For SL/EOD we use a
        limit BELOW the mark so a cross would happen if the price is
        moving against us. For TP/trailing we use a limit slightly below
        mark to ensure a fill while the bracket leg may already be gone.
        """
        if trigger == "sl":
            # Use mark (or slightly below) — the bracket stop will fire if mark hits
            return max(mark * 0.97, 0.05)
        if trigger == "tp":
            # Capture the gain; limit at mark
            return max(mark * 0.99, 0.10)
        if trigger == "trailing_floor":
            return max(mark * 0.99, 0.10)
        if trigger == "eod":
            # EOD: get out, even if a small slip — use mark (broker may reject)
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