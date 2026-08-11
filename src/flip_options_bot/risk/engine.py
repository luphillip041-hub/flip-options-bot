"""Risk engine — per-trade caps, daily/weekly caps, kill switch, idempotent state.

Pattern (from paper-to-live-trading-bot-scaffold + go-trader-prior-art):

- `tick_rollover` only zeros pnl if `last_reset_day` is non-empty AND
  different from today. Fresh state without a prior day string is initialized,
  not zeroed.

- `_check_loss_caps()` runs BEFORE the kill_switch short-circuit. A fresh cap
  breach always escalates to KILL, never silently passes.

- `record_close()` is a method (not a setter), so callers can't bypass
  invariants.

- State is persisted to SQLite at `run_dir/state.db`. Idempotent at the row
  level — never writes a duplicate entry for the same `event_id`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from ..config import Settings


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    last_reset_day: str = ""
    last_reset_week: str = ""
    kill_switch: bool = False
    kill_reason: str = ""
    open_position_count: int = 0
    realized_pnl_total: float = 0.0
    updated_at: str = ""


class RiskEngine:
    """Stateless risk evaluator over a persisted RiskState.

    Usage:
        state = risk_engine.load_state()
        decision = risk_engine.evaluate_pre_trade(state, equity=10_226.92)
        if decision.allowed:
            # submit order, then risk_engine.record_open(state)
        else:
            log(decision.reason)

        on fill, call risk_engine.record_close(state, pnl=..., event_id=...).
    """

    def __init__(self, settings: Settings, run_dir: Path):
        self.settings = settings
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "state.db"
        self._init_db()

    # ===== DB =====

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    daily_pnl REAL NOT NULL DEFAULT 0,
                    weekly_pnl REAL NOT NULL DEFAULT 0,
                    last_reset_day TEXT NOT NULL DEFAULT '',
                    last_reset_week TEXT NOT NULL DEFAULT '',
                    kill_switch INTEGER NOT NULL DEFAULT 0,
                    kill_reason TEXT NOT NULL DEFAULT '',
                    open_position_count INTEGER NOT NULL DEFAULT 0,
                    realized_pnl_total REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    pnl REAL,
                    payload TEXT
                )
                """
            )
            conn.commit()

    def load_state(self) -> RiskState:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()
        if row is None:
            state = RiskState()
            self._persist_state(state)
            return state
        return RiskState(
            daily_pnl=row[1],
            weekly_pnl=row[2],
            last_reset_day=row[3],
            last_reset_week=row[4],
            kill_switch=bool(row[5]),
            kill_reason=row[6],
            open_position_count=row[7],
            realized_pnl_total=row[8],
            updated_at=row[9],
        )

    def _persist_state(self, state: RiskState) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO risk_state (id, daily_pnl, weekly_pnl, last_reset_day, last_reset_week,
                                        kill_switch, kill_reason, open_position_count,
                                        realized_pnl_total, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    daily_pnl=excluded.daily_pnl,
                    weekly_pnl=excluded.weekly_pnl,
                    last_reset_day=excluded.last_reset_day,
                    last_reset_week=excluded.last_reset_week,
                    kill_switch=excluded.kill_switch,
                    kill_reason=excluded.kill_reason,
                    open_position_count=excluded.open_position_count,
                    realized_pnl_total=excluded.realized_pnl_total,
                    updated_at=excluded.updated_at
                """,
                (
                    state.daily_pnl,
                    state.weekly_pnl,
                    state.last_reset_day,
                    state.last_reset_week,
                    int(state.kill_switch),
                    state.kill_reason,
                    state.open_position_count,
                    state.realized_pnl_total,
                    state.updated_at,
                ),
            )
            conn.commit()

    def record_event(self, event_id: str, kind: str, pnl: float | None = None, payload: str = "") -> None:
        """Idempotent event log. INSERT OR IGNORE on event_id prevents dup-writes
        from a stuck reconcile loop. This is the structural fix for the
        'duplicate close events' artifact class that produced the -$43k phantom
        in flip-alpaca-bot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO risk_events (event_id, ts, kind, pnl, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    datetime.now(timezone.utc).isoformat(),
                    kind,
                    pnl,
                    payload,
                ),
            )
            conn.commit()

    # ===== Tick rollover =====

    def tick_rollover(self, state: RiskState) -> RiskState:
        """Called once per loop. Updates daily/weekly buckets.

        Bug-class fix: do NOT zero daily_pnl on a fresh state. Only zero
        if last_reset_day is non-empty AND different from today.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        iso_week = now.strftime("%Y-W%W")

        if state.last_reset_day and state.last_reset_day != today:
            state.daily_pnl = 0.0
            state.last_reset_day = today
        elif not state.last_reset_day:
            state.last_reset_day = today  # initialize, don't zero

        if state.last_reset_week and state.last_reset_week != iso_week:
            state.weekly_pnl = 0.0
            state.last_reset_week = iso_week
        elif not state.last_reset_week:
            state.last_reset_week = iso_week  # initialize, don't zero

        state.updated_at = now.isoformat()
        self._persist_state(state)
        return state

    # ===== Pre-trade evaluate =====

    @dataclass
    class Decision:
        allowed: bool
        reason: str = ""
        contracts: int = 0

    def evaluate_caps(self, state: RiskState, equity: float) -> bool:
        """Check daily + weekly loss caps. If breached, trip kill switch
        and return True (caller should flatten positions).

        This is called every cycle by the monitor loop, separate from
        `evaluate_pre_trade` (which is called only when entering a trade).
        Caps get checked on every tick so a single bad close triggers the
        trip immediately, not on the next scan.
        """
        daily_cap = equity * (self.settings.daily_loss_cap_pct / 100.0)
        weekly_cap = equity * (self.settings.weekly_loss_cap_pct / 100.0)
        if state.daily_pnl <= -daily_cap:
            state.kill_switch = True
            state.kill_reason = (
                f"daily_loss_cap: {state.daily_pnl:.2f} <= -{daily_cap:.2f}"
            )
            self._persist_state(state)
            return True
        if state.weekly_pnl <= -weekly_cap:
            state.kill_switch = True
            state.kill_reason = (
                f"weekly_loss_cap: {state.weekly_pnl:.2f} <= -{weekly_cap:.2f}"
            )
            self._persist_state(state)
            return True
        return False

    def evaluate_pre_trade(
        self,
        state: RiskState,
        equity: float,
        proposed_debit: float,
    ) -> "RiskEngine.Decision":
        """Pre-trade gate. Order of checks matters:

        1. Kill switch — short-circuit. (If the kill switch fires, no trade.)
        2. Loss caps — must run BEFORE the kill switch short-circuit. A fresh
           cap breach escalates to KILL and blocks the trade.
        3. Per-trade risk — debit vs equity.
        4. Position count — max open.
        """
        # === 1. Loss caps — runs BEFORE the kill-switch short-circuit ===
        daily_cap_dollar = equity * (self.settings.daily_loss_cap_pct / 100.0)
        weekly_cap_dollar = equity * (self.settings.weekly_loss_cap_pct / 100.0)
        if state.daily_pnl <= -daily_cap_dollar:
            state.kill_switch = True
            state.kill_reason = (
                f"daily_loss_cap_breach: {state.daily_pnl:.2f} <= -{daily_cap_dollar:.2f}"
            )
            self._persist_state(state)
            return self.Decision(allowed=False, reason=state.kill_reason)
        if state.weekly_pnl <= -weekly_cap_dollar:
            state.kill_switch = True
            state.kill_reason = (
                f"weekly_loss_cap_breach: {state.weekly_pnl:.2f} <= -{weekly_cap_dollar:.2f}"
            )
            self._persist_state(state)
            return self.Decision(allowed=False, reason=state.kill_reason)

        # === 2. Kill switch ===
        if state.kill_switch:
            return self.Decision(allowed=False, reason=f"kill_switch: {state.kill_reason}")

        # === 3. Per-trade risk ===
        max_debit = equity * (self.settings.per_trade_risk_pct / 100.0)
        if proposed_debit > max_debit:
            return self.Decision(
                allowed=False,
                reason=f"per_trade_risk: {proposed_debit:.2f} > {max_debit:.2f}",
            )

        # === 4. Max contract dollar cap ===
        if proposed_debit * 100 > self.settings.max_contract_dollar:
            return self.Decision(
                allowed=False,
                reason=(
                    f"max_contract_dollar: {proposed_debit * 100:.0f} > "
                    f"{self.settings.max_contract_dollar}"
                ),
            )

        # === 5. Position count ===
        if state.open_position_count >= self.settings.max_positions:
            return self.Decision(
                allowed=False,
                reason=(
                    f"max_positions: {state.open_position_count} >= "
                    f"{self.settings.max_positions}"
                ),
            )

        # All gates passed. Compute contracts.
        contracts = max(1, int(self.settings.max_contract_dollar / max(proposed_debit * 100, 1)))
        # Cap at max-positions remaining
        contracts = min(contracts, self.settings.max_positions - state.open_position_count)
        return self.Decision(allowed=True, contracts=contracts)

    # ===== Records =====

    def record_open(self, state: RiskState, symbol: str, debit: float, event_id: str) -> RiskState:
        state.open_position_count += 1
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_state(state)
        self.record_event(event_id, "open", pnl=-debit, payload=symbol)
        return state

    def record_close(
        self, state: RiskState, pnl: float, event_id: str, payload: str = ""
    ) -> RiskState:
        """Idempotent close event. INSERT OR IGNORE on event_id prevents
        duplicate close writes from the reconcile loop."""
        # Verify event isn't already recorded (defense in depth)
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM risk_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if existing:
            # Already recorded; do NOT mutate state. This is the structural fix.
            return state

        state.open_position_count = max(0, state.open_position_count - 1)
        state.daily_pnl += pnl
        state.weekly_pnl += pnl
        state.realized_pnl_total += pnl
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_state(state)
        self.record_event(event_id, "close", pnl=pnl, payload=payload)
        return state

    # ===== Kill switch =====

    def force_kill(self, state: RiskState, reason: str) -> RiskState:
        state.kill_switch = True
        state.kill_reason = reason
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_state(state)
        return state

    def release_kill(self, state: RiskState) -> RiskState:
        state.kill_switch = False
        state.kill_reason = ""
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_state(state)
        return state
