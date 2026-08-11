"""Trade journal — append-only log of fills and closes.

Structural fixes vs. flip-alpaca-bot:
1. Every event keyed by `client_order_id` (broker's order ID, which IS the
   trade_id). INSERT OR IGNORE means a stuck reconcile loop cannot create
   duplicate close writes — the canonical fix for the
   'duplicate close events' artifact that produced the -$43k phantom.

2. The `realized_pnl_original` field is NEVER used for ledger reconciliation.
   The canonical close ledger always prefers the broker's actual fill price
   from `list_filled_orders`, falling back to the WebSocket trade-update
   stream, falling back to the indicative quote — never the synthetic
   conservative-loss field. This is the fix for the
   'wrong field' artifact class.

3. Position_id is a UUID generated at entry, NEVER derived from symbol+date.
   The same (symbol, date) tuple can have multiple legitimate closes (multiple
   entry rounds); position_id disambiguates them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass
class TradeEvent:
    event_id: str  # client_order_id from broker
    ts: str
    kind: Literal["open", "close", "open_attempt", "close_attempt", "fill_partial"]
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    price: float
    position_id: str = ""  # UUID for opens; propagated to closes
    realized_pnl: float = 0.0
    fees: float = 0.0
    strategy_id: str = ""
    raw_broker_fill: dict | None = None


class Journal:
    """SQLite-backed append-only journal. Idempotent on event_id."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "journal.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    event_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    price REAL NOT NULL,
                    position_id TEXT,
                    realized_pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    strategy_id TEXT DEFAULT '',
                    raw_broker_fill TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_position ON trades (position_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts)"
            )
            # Map position_id → current state. One open event, then
            # one or more close events. position_state is updated on each event.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    qty_open INTEGER NOT NULL DEFAULT 0,
                    qty_closed INTEGER NOT NULL DEFAULT 0,
                    avg_entry_price REAL,
                    avg_exit_price REAL,
                    realized_pnl REAL DEFAULT 0,
                    strategy_id TEXT DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'open'
                )
                """
            )
            conn.commit()

    def append(self, event: TradeEvent) -> bool:
        """Idempotent append. Returns True if written, False if already existed.

        This is the structural fix: a duplicate event_id from a stuck reconcile
        loop returns False and is silently ignored. No double-write.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO trades (
                    event_id, ts, kind, symbol, side, qty, price,
                    position_id, realized_pnl, fees, strategy_id, raw_broker_fill
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.ts,
                    event.kind,
                    event.symbol,
                    event.side,
                    event.qty,
                    event.price,
                    event.position_id,
                    event.realized_pnl,
                    event.fees,
                    event.strategy_id,
                    json.dumps(event.raw_broker_fill) if event.raw_broker_fill else None,
                ),
            )
            written = cur.rowcount == 1
            if written:
                self._update_position_state(conn, event)
            conn.commit()
            return written

    def _update_position_state(self, conn: sqlite3.Connection, event: TradeEvent) -> None:
        if not event.position_id:
            return
        if event.kind == "open":
            conn.execute(
                """
                INSERT INTO positions (
                    position_id, symbol, opened_at, qty_open,
                    avg_entry_price, strategy_id, state
                ) VALUES (?, ?, ?, ?, ?, ?, 'open')
                ON CONFLICT(position_id) DO UPDATE SET
                    qty_open = qty_open + excluded.qty_open
                """,
                (
                    event.position_id,
                    event.symbol,
                    event.ts,
                    event.qty,
                    event.price,
                    event.strategy_id,
                ),
            )
        elif event.kind == "close":
            # Read current state first, then update.
            row = conn.execute(
                "SELECT qty_open, qty_closed, avg_exit_price FROM positions WHERE position_id = ?",
                (event.position_id,),
            ).fetchone()
            if row is None:
                # Closing an unknown position_id is a logic error; skip.
                return
            qty_open, qty_closed_prev, avg_exit_prev = row
            qty_closed_prev = qty_closed_prev or 0
            avg_exit_prev = avg_exit_prev or 0.0

            new_qty_closed = qty_closed_prev + event.qty
            # Weighted-average exit price
            if new_qty_closed > 0:
                new_avg_exit = (
                    (avg_exit_prev * qty_closed_prev) + (event.price * event.qty)
                ) / new_qty_closed
            else:
                new_avg_exit = 0.0

            new_state = "closed" if new_qty_closed >= (qty_open or 0) else "partial"

            conn.execute(
                """
                UPDATE positions SET
                    closed_at = COALESCE(closed_at, ?),
                    qty_closed = ?,
                    avg_exit_price = ?,
                    realized_pnl = realized_pnl + ?,
                    state = ?
                WHERE position_id = ?
                """,
                (
                    event.ts,
                    new_qty_closed,
                    new_avg_exit,
                    event.realized_pnl,
                    new_state,
                    event.position_id,
                ),
            )

    # ===== Queries =====

    def get_open_positions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE state = 'open' OR state = 'partial'"
            ).fetchall()
        return [dict(zip([c[0] for c in conn.execute("SELECT * FROM positions LIMIT 0").description], r)) for r in rows] if rows else []

    def get_all_positions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM positions ORDER BY opened_at"
            ).fetchall()
            if not rows:
                return []
            cols = [c[0] for c in conn.execute("SELECT * FROM positions LIMIT 0").description]
            return [dict(zip(cols, r)) for r in rows]

    def has_event(self, event_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT 1 FROM trades WHERE event_id = ?", (event_id,)
            ).fetchone() is not None

    def total_realized_pnl(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE kind = 'close'"
            ).fetchone()
        return float(row[0]) if row else 0.0

    # ===== Helpers =====

    @staticmethod
    def new_position_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()