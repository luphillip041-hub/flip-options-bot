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
    kind: Literal["open", "close", "open_attempt", "close_attempt", "fill_partial", "open_spread", "close_spread"]
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
                    state TEXT NOT NULL DEFAULT 'open',
                    peak_mark REAL,
                    tp_order_id TEXT,
                    sl_order_id TEXT,
                    tp_price REAL,
                    sl_trigger_price REAL,
                    sl_limit_price REAL
                )
                """
            )
            # Migration: add new columns if missing (older deployments).
            for col_ddl in (
                "ALTER TABLE positions ADD COLUMN peak_mark REAL",
                "ALTER TABLE positions ADD COLUMN tp_order_id TEXT",
                "ALTER TABLE positions ADD COLUMN sl_order_id TEXT",
                "ALTER TABLE positions ADD COLUMN tp_price REAL",
                "ALTER TABLE positions ADD COLUMN sl_trigger_price REAL",
                "ALTER TABLE positions ADD COLUMN sl_limit_price REAL",
            ):
                try:
                    conn.execute(col_ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._backfill_missing_positions(conn)
            conn.commit()

    def _backfill_missing_positions(self, conn: sqlite3.Connection) -> None:
        """Create position rows for older open/open_spread events.

        Commit 04afba5 introduced open_spread before positions-state knew
        how to materialize it. This migration makes the live filled spread
        visible to the monitor on restart, without duplicating existing rows.
        """
        rows = conn.execute(
            """
            SELECT DISTINCT position_id FROM trades
            WHERE position_id IS NOT NULL AND position_id != ''
              AND kind IN ('open', 'open_spread')
              AND position_id NOT IN (SELECT position_id FROM positions)
            """
        ).fetchall()
        for (position_id,) in rows:
            open_rows = conn.execute(
                """
                SELECT ts, symbol, qty, price, strategy_id FROM trades
                WHERE position_id = ? AND kind IN ('open', 'open_spread')
                ORDER BY ts
                """,
                (position_id,),
            ).fetchall()
            if not open_rows:
                continue
            qty_open = sum(int(r[2] or 0) for r in open_rows)
            avg_entry = (
                sum(float(r[2] or 0) * float(r[3] or 0.0) for r in open_rows) / qty_open
                if qty_open > 0 else 0.0
            )
            first = open_rows[0]
            conn.execute(
                """
                INSERT OR IGNORE INTO positions (
                    position_id, symbol, opened_at, qty_open,
                    avg_entry_price, strategy_id, state
                ) VALUES (?, ?, ?, ?, ?, ?, 'open')
                """,
                (position_id, first[1], first[0], qty_open, avg_entry, first[4] or ""),
            )

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

    def upsert(self, event: TradeEvent) -> bool:
        """Idempotent UPSERT. Returns True if a new row was created, False if updated.

        Use this for canonical reconciliation events (e.g., close fills from
        the broker). If the row already exists with the same event_id, the
        price + realized_pnl + raw_broker_fill fields are overwritten so the
        canonical broker fill wins over the placeholder written by the
        closer. The position_state is then recomputed.

        SQLite's INSERT...ON CONFLICT DO UPDATE always returns rowcount=1,
        so we explicitly check for existence beforehand to distinguish
        create vs update.
        """
        was_new = not self.has_event(event.event_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    event_id, ts, kind, symbol, side, qty, price,
                    position_id, realized_pnl, fees, strategy_id, raw_broker_fill
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    ts = excluded.ts,
                    price = excluded.price,
                    realized_pnl = excluded.realized_pnl,
                    fees = excluded.fees,
                    raw_broker_fill = excluded.raw_broker_fill
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
            # Always recompute position_state so qty_closed + state reflect the
            # latest close event (not the placeholder).
            self._update_position_state(conn, event)
            conn.commit()
            return was_new

    def _update_position_state(self, conn: sqlite3.Connection, event: TradeEvent) -> None:
        if not event.position_id:
            return
        if event.kind in ("open", "open_spread"):
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
        elif event.kind in ("close", "close_spread"):
            # Idempotent recompute from the canonical trades table.
            # This is safe for upserts (the same event_id arriving twice
            # produces the same totals).
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN kind IN ('open', 'open_spread') THEN qty ELSE 0 END), 0) AS qty_open_total,
                    COALESCE(SUM(CASE WHEN kind IN ('close', 'close_spread') THEN qty ELSE 0 END), 0) AS qty_closed_total,
                    COALESCE(SUM(CASE WHEN kind IN ('close', 'close_spread') THEN realized_pnl ELSE 0 END), 0) AS realized_total
                FROM trades WHERE position_id = ?
                """,
                (event.position_id,),
            ).fetchone()
            qty_open_total, qty_closed_total, realized_total = row

            # Weighted avg exit price from close rows only
            close_rows = conn.execute(
                """
                SELECT qty, price FROM trades
                WHERE position_id = ? AND kind IN ('close', 'close_spread')
                """,
                (event.position_id,),
            ).fetchall()
            total_qty_closed = sum(r[0] for r in close_rows)
            if total_qty_closed > 0:
                avg_exit = sum(r[0] * r[1] for r in close_rows) / total_qty_closed
            else:
                avg_exit = 0.0

            new_state = "closed" if qty_closed_total >= qty_open_total and qty_open_total > 0 else "open"

            conn.execute(
                """
                UPDATE positions SET
                    qty_open = ?,
                    qty_closed = ?,
                    avg_exit_price = ?,
                    realized_pnl = ?,
                    state = ?,
                    closed_at = COALESCE(closed_at, ?)
                WHERE position_id = ?
                """,
                (
                    qty_open_total,
                    qty_closed_total,
                    avg_exit,
                    realized_total,
                    new_state,
                    event.ts,
                    event.position_id,
                ),
            )

    # ===== Queries =====

    def get_open_positions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM positions WHERE state = 'open' OR state = 'partial'"
            )
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in rows]

    def get_all_positions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM positions ORDER BY opened_at")
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in rows]

    def get_position_for_id(self, position_id: str) -> dict | None:
        """Return the positions row for a single position_id, or None."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,))
            row = cur.fetchone()
        if not row:
            return None
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, row))

    def get_legs_for_position(self, position_id: str) -> list[dict]:
        """Return all trade events tied to a position_id.

        For BPCS this returns the 2 legs (short + long). For long_call
        this returns the open + (optionally) close.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM trades WHERE position_id = ? ORDER BY ts",
                (position_id,),
            )
            rows = cur.fetchall()
        if not rows:
            return []
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]

    def has_event(self, event_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT 1 FROM trades WHERE event_id = ?", (event_id,)
            ).fetchone() is not None

    def set_bracket(
        self,
        position_id: str,
        tp_order_id: str | None,
        sl_order_id: str | None,
        tp_price: float | None,
        sl_trigger_price: float | None,
        sl_limit_price: float | None,
    ) -> None:
        """Record the bracket leg order IDs + prices for a position.

        Called by the executor after `submit_bracket_buy` returns the
        bracket legs from the broker. Used by `reconcile_fills` to
        detect if a bracket leg filled and write the canonical close
        event with the correct realized_pnl.
        """
        if not position_id:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE positions SET
                    tp_order_id = COALESCE(?, tp_order_id),
                    sl_order_id = COALESCE(?, sl_order_id),
                    tp_price = COALESCE(?, tp_price),
                    sl_trigger_price = COALESCE(?, sl_trigger_price),
                    sl_limit_price = COALESCE(?, sl_limit_price)
                WHERE position_id = ?
                """,
                (
                    tp_order_id,
                    sl_order_id,
                    tp_price,
                    sl_trigger_price,
                    sl_limit_price,
                    position_id,
                ),
            )
            conn.commit()

    def total_realized_pnl(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE kind IN ('close', 'close_spread')"
            ).fetchone()
        return float(row[0]) if row else 0.0

    # ===== Helpers =====

    @staticmethod
    def new_position_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()