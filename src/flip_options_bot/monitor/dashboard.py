"""Streamlit dashboard — read-only views of the bot state.

Read-only by design. NO order submission from the dashboard. NO state
mutation. Just observability.

Layout (4 sections):
1. Status: heartbeat.json + last scan ts + phase + live flag
2. Risk state: daily P&L, weekly P&L, kill switch, open positions
3. Journal: open positions, recent closed positions, realized P&L total
4. Funnel: latest 10 funnel rows + dominant skip reason
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from ..config import get_settings


def _read_heartbeat(run_dir: Path) -> dict | None:
    p = run_dir / "heartbeat.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_risk_state(db_path: Path) -> dict | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()
    if row is None:
        return None
    return {
        "daily_pnl": row[1],
        "weekly_pnl": row[2],
        "last_reset_day": row[3],
        "last_reset_week": row[4],
        "kill_switch": bool(row[5]),
        "kill_reason": row[6],
        "open_position_count": row[7],
        "realized_pnl_total": row[8],
        "updated_at": row[9],
    }


def _read_positions(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT position_id, symbol, opened_at, closed_at, qty_open, qty_closed, "
            "avg_entry_price, avg_exit_price, realized_pnl, strategy_id, state "
            "FROM positions ORDER BY opened_at DESC",
            conn,
        )
    return df


def _read_funnel(run_dir: Path, n: int = 20) -> pd.DataFrame:
    p = run_dir / "funnel.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text().splitlines()[-n:]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="flip-options-bot", page_icon="📈", layout="wide")
    st.title("flip-options-bot")

    settings = get_settings()
    run_dir = settings.run_dir

    # ===== Section 1: Status =====
    st.header("Status")
    hb = _read_heartbeat(run_dir)
    if hb:
        col1, col2, col3 = st.columns(3)
        col1.metric("Phase", hb.get("phase", "?"))
        col2.metric("Live mode", "ON" if hb.get("is_live_mode") else "OFF")
        col3.metric("Last heartbeat", hb.get("ts", "?"))
    else:
        st.warning("No heartbeat.json found. Daemon hasn't run yet.")

    # ===== Section 2: Risk state =====
    st.header("Risk state")
    risk = _read_risk_state(run_dir / "state.db")
    if risk:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Daily P&L", f"${risk['daily_pnl']:.2f}")
        col2.metric("Weekly P&L", f"${risk['weekly_pnl']:.2f}")
        col3.metric("Realized total", f"${risk['realized_pnl_total']:.2f}")
        col4.metric("Open positions", risk["open_position_count"])
        if risk["kill_switch"]:
            st.error(f"KILL SWITCH: {risk['kill_reason']}")
    else:
        st.info("No risk state yet.")

    # ===== Section 3: Journal — positions =====
    st.header("Journal — positions")
    positions = _read_positions(run_dir / "journal.db")
    if not positions.empty:
        # Color-code by state
        def _state_color(s: str) -> str:
            return {
                "open": "🟢",
                "partial": "🟡",
                "closed": "⚪",
            }.get(s, "❓")

        positions["status"] = positions["state"].apply(_state_color)
        st.dataframe(positions, use_container_width=True)

        # Summary stats
        n_open = (positions["state"] == "open").sum()
        n_partial = (positions["state"] == "partial").sum()
        n_closed = (positions["state"] == "closed").sum()
        realized = positions[positions["state"] == "closed"]["realized_pnl"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open", n_open)
        c2.metric("Partial", n_partial)
        c3.metric("Closed", n_closed)
        c4.metric("Realized P&L (closed)", f"${realized:+.2f}")
    else:
        st.info("No positions recorded yet.")

    # ===== Section 4: Funnel =====
    st.header("Funnel — last 20 scans")
    funnel = _read_funnel(run_dir, n=20)
    if not funnel.empty:
        keep_cols = [c for c in [
            "ts", "watchlist_count", "eligible_count", "chains_fetched", "chains_failed",
            "move_pass_count", "momentum_pass_count", "contract_select_pass",
            "sized_count", "submitted_count", "dominant_skip_reason",
        ] if c in funnel.columns]
        st.dataframe(funnel[keep_cols], use_container_width=True)
    else:
        st.info("No funnel rows yet.")


if __name__ == "__main__":
    main()
