"""Paper-observation harness.

Tracks the paper-trading window so the bot is only eligible to be promoted
to live after N market days of clean paper trading. The harness is gated:

- Each market day, the operator (or a cron job) calls `record_market_day()`
  to mark a clean trading day.
- `promotion_gate()` returns True only when:
    - At least N market days are recorded
    - Win rate over closed positions > 50% (the WR floor is a placeholder;
      flip-alpaca-bot should set it based on the strategy's expected edge)
    - Max drawdown over the window < 10% of starting equity
    - Net realized P&L >= 0

The harness writes `runs/observation_state.json` with the current day count,
latest win-rate / drawdown, and a per-day history list. The next digest
cron reads this and reports the promotion readiness to Discord.

This is NOT auto-promotion. Promotion to live always requires an explicit
operator flip (`LIVETRADE_ENABLED=true`) AND `promotion_gate()` returning
True. The harness only reports readiness.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("flip_options_bot.observation")


@dataclass
class PromotionGate:
    eligible: bool
    reason: str
    market_days_recorded: int
    min_market_days: int
    win_rate: float
    min_win_rate: float
    max_drawdown_pct: float
    max_drawdown_allowed_pct: float
    net_realized_pnl: float


class ObservationHarness:
    """Tracks paper-trading observation window for promotion eligibility."""

    def __init__(
        self,
        run_dir: Path,
        min_market_days: int = 10,
        min_win_rate: float = 0.50,
        max_drawdown_allowed_pct: float = 10.0,
    ):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "observation_state.json"
        self.min_market_days = min_market_days
        self.min_win_rate = min_win_rate
        self.max_drawdown_allowed_pct = max_drawdown_allowed_pct
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {
                "market_days": [],          # list of YYYY-MM-DD strings
                "started_at": "",           # ISO timestamp
                "last_recorded_at": "",     # ISO timestamp
                "promoted_to_live": False,
                "promotion_blocked_reasons": [],
            }
        try:
            return json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            log.warning("observation_state.json corrupt; resetting")
            return {"market_days": [], "started_at": "", "last_recorded_at": "",
                    "promoted_to_live": False, "promotion_blocked_reasons": []}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True))

    def record_market_day(self, day_iso: str | None = None) -> bool:
        """Mark a market day as cleanly traded. Idempotent on the date."""
        day = day_iso or datetime.now(UTC).strftime("%Y-%m-%d")
        if day in self.state["market_days"]:
            log.info("market day %s already recorded; skipping", day)
            return False
        if not self.state["started_at"]:
            self.state["started_at"] = datetime.now(UTC).isoformat()
        self.state["market_days"].append(day)
        self.state["last_recorded_at"] = datetime.now(UTC).isoformat()
        self._save_state()
        log.info("recorded paper market day %s (total: %d)",
                 day, len(self.state["market_days"]))
        return True

    def compute_realized_stats(self, journal_db_path: Path) -> tuple[float, float, float]:
        """Returns (win_rate, max_drawdown_pct, net_realized_pnl)."""
        if not journal_db_path.exists():
            return 0.0, 0.0, 0.0
        with sqlite3.connect(journal_db_path) as conn:
            closes = conn.execute(
                "SELECT realized_pnl FROM trades WHERE kind='close' AND realized_pnl IS NOT NULL"
            ).fetchall()
        pnls = [r[0] for r in closes]
        if not pnls:
            return 0.0, 0.0, 0.0
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        net = sum(pnls)
        # Max drawdown as a percent of cumulative peak
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
        # Convert to percent of starting equity (assume $10k for the harness)
        max_dd_pct = (abs(max_dd) / 10_000.0) * 100.0
        return win_rate, max_dd_pct, net

    def promotion_gate(self, journal_db_path: Path) -> PromotionGate:
        """Evaluate whether the bot is eligible for promotion to live."""
        n_days = len(self.state["market_days"])
        win_rate, max_dd_pct, net_pnl = self.compute_realized_stats(journal_db_path)

        reasons = []
        if n_days < self.min_market_days:
            reasons.append(
                f"only {n_days} paper market days recorded; need {self.min_market_days}"
            )
        if win_rate < self.min_win_rate:
            reasons.append(
                f"win rate {win_rate:.1%} below {self.min_win_rate:.0%} floor"
            )
        if max_dd_pct > self.max_drawdown_allowed_pct:
            reasons.append(
                f"max drawdown {max_dd_pct:.1f}% > {self.max_drawdown_allowed_pct}% ceiling"
            )
        if net_pnl < 0:
            reasons.append(f"net realized P&L negative: ${net_pnl:+.2f}")

        eligible = not reasons
        return PromotionGate(
            eligible=eligible,
            reason="; ".join(reasons) if reasons else "all gates passed",
            market_days_recorded=n_days,
            min_market_days=self.min_market_days,
            win_rate=win_rate,
            min_win_rate=self.min_win_rate,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_allowed_pct=self.max_drawdown_allowed_pct,
            net_realized_pnl=net_pnl,
        )

    def render_digest(self, journal_db_path: Path) -> str:
        """Build a Discord-friendly summary of the observation window."""
        gate = self.promotion_gate(journal_db_path)
        n_days = gate.market_days_recorded
        days_left = max(0, gate.min_market_days - n_days)
        ready = "✅ READY" if gate.eligible else f"❌ {days_left}d to go"
        return (
            f"📋 **flip-options-bot paper observation** — {ready}\n"
            f"Days: {n_days}/{gate.min_market_days}\n"
            f"Win rate: {gate.win_rate:.1%} (floor {gate.min_win_rate:.0%})\n"
            f"Max drawdown: {gate.max_drawdown_pct:.1f}% (ceiling {gate.max_drawdown_allowed_pct}%)\n"
            f"Net realized P&L: ${gate.net_realized_pnl:+.2f}\n"
            f"Reason: {gate.reason}"
        )
