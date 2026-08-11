"""Daily paper-observation digest — runs at 08:05 ET, posts to Discord #flipphill.

Renders a markdown summary of the previous day's paper-trading session:

- Account equity / cash / buying power
- Open positions
- Closed positions from the last 24h
- Funnel stats: scans run, candidates, submitted, dominant skip
- Realized P&L total
- Promotion-gate verdict (per `observation.promotion_gate`)

Designed to be invoked by cron (or as a one-shot for testing).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the bot package importable when this script is run from cron
_BOT_ROOT = Path("/root/flip/projects/flip-options-bot")
if str(_BOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BOT_ROOT / "src"))

from flip_options_bot.config import get_settings  # noqa: E402
from flip_options_bot.journal import Journal  # noqa: E402
from flip_options_bot.observation import ObservationHarness  # noqa: E402
from flip_options_bot.risk import RiskEngine  # noqa: E402

CHANNEL_ID = "1531705542538690741"  # #flipphill


def _load_bot_token() -> str:
    """Load the Discord bot token from /root/.config/flip-options-bot/.env.

    We don't import get_settings() here because the .env loader only reads
    from cwd, not the bot's config dir. Use python-dotenv directly.
    """
    from dotenv import dotenv_values

    env_path = Path("/root/.config/flip-options-bot/.env")
    if env_path.exists():
        env = dotenv_values(env_path)
    else:
        env = {}
    # Process env wins (cron sets this explicitly via /etc/cron.d)
    token = os.environ.get("FOB_DISCORD_BOT_TOKEN") or env.get("FOB_DISCORD_BOT_TOKEN", "")
    if not token:
        raise RuntimeError(
            "FOB_DISCORD_BOT_TOKEN not set. Add to /root/.config/flip-options-bot/.env "
            "or set as process env before invoking this script."
        )
    return token


def _load_account_snapshot() -> dict:
    """Pull live account snapshot from the bot's state.db (no broker call)."""
    settings = get_settings()
    state_db = settings.run_dir / "state.db"
    if not state_db.exists():
        return {}
    risk = RiskEngine(settings, settings.run_dir)
    state = risk.load_state()
    return {
        "equity_estimate": round(
            float(settings.equity_start) + float(state.realized_pnl_total), 2
        ),
        "daily_pnl": round(state.daily_pnl, 2),
        "weekly_pnl": round(state.weekly_pnl, 2),
        "realized_total": round(state.realized_pnl_total, 2),
        "open_positions": state.open_position_count,
        "kill_switch": state.kill_switch,
        "kill_reason": state.kill_reason,
    }


def _load_recent_closes(hours: int = 24) -> list[dict]:
    """Trades with kind=close in the last N hours, from journal.db."""
    settings = get_settings()
    db = settings.run_dir / "journal.db"
    if not db.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    out = []
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT event_id, ts, symbol, qty, price, realized_pnl, strategy_id "
            "FROM trades WHERE kind = 'close' AND ts >= ? ORDER BY ts DESC",
            (cutoff,),
        ).fetchall()
    for r in rows:
        out.append({
            "event_id": r[0],
            "ts": r[1],
            "symbol": r[2],
            "qty": r[3],
            "price": r[4],
            "realized_pnl": r[5],
            "strategy_id": r[6],
        })
    return out


def _load_funnel_stats(hours: int = 24) -> dict:
    settings = get_settings()
    funnel_path = settings.run_dir / "funnel.jsonl"
    if not funnel_path.exists():
        return {"scans": 0, "candidates": 0, "submitted": 0, "dominant_skip": ""}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours))
    scans = 0
    candidates_total = 0
    submitted_total = 0
    skip_counts: dict[str, int] = {}
    for line in funnel_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = row.get("ts", "")
        if ts and datetime.fromisoformat(ts.replace("Z", "+00:00")) < cutoff:
            continue
        scans += 1
        candidates_total += row.get("sized_count", 0)
        submitted_total += row.get("submitted_count", 0)
        skip = row.get("dominant_skip_reason", "")
        if skip:
            skip_counts[skip] = skip_counts.get(skip, 0) + 1
    dominant_skip = (
        max(skip_counts.items(), key=lambda x: x[1])[0] if skip_counts else ""
    )
    return {
        "scans": scans,
        "candidates": candidates_total,
        "submitted": submitted_total,
        "dominant_skip": dominant_skip,
    }


def render_markdown(account: dict, closes: list[dict], funnel: dict, gate: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📊 **flip-options-bot daily digest** — {now}",
        "",
        "**Account**",
        f"- Equity (estimate): ${account.get('equity_estimate', '?')}",
        f"- Daily P&L: ${account.get('daily_pnl', 0):+.2f}",
        f"- Weekly P&L: ${account.get('weekly_pnl', 0):+.2f}",
        f"- Realized total: ${account.get('realized_total', 0):+.2f}",
        f"- Open positions: {account.get('open_positions', 0)}",
    ]
    if account.get("kill_switch"):
        lines.append(f"- **KILL SWITCH ACTIVE**: {account.get('kill_reason', '')}")

    lines.extend([
        "",
        "**Last 24h funnel**",
        f"- Scans: {funnel['scans']}",
        f"- Candidates: {funnel['candidates']}",
        f"- Submitted: {funnel['submitted']}",
        f"- Dominant skip: `{funnel['dominant_skip']}`",
    ])

    lines.extend(["", "**Last 24h closes**"])
    if closes:
        for c in closes[:10]:
            sign = "+" if c["realized_pnl"] >= 0 else ""
            lines.append(
                f"- `{c['symbol']}` qty={c['qty']} @ ${c['price']:.2f} → {sign}${c['realized_pnl']:.2f}"
            )
    else:
        lines.append("- (none)")

    lines.extend([
        "",
        "**Promotion gate**",
        f"- Eligible: **{'YES' if gate.get('eligible') else 'NO'}**",
    ])
    for k, v in gate.items():
        if k == "eligible":
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("_Live mode requires explicit operator action (set `LIVETRADE_ENABLED=true`). Auto-promotion is disabled._")
    return "\n".join(lines)


def post_to_discord(content: str) -> bool:
    """Post the digest to #flipphill via Discord API. Uses curl subprocess."""
    token = _load_bot_token()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"content": content}, f)
        payload_path = f.name
    try:
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "-X",
                "POST",
                f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                "-H",
                f"Authorization: Bot {token}",
                "-F",
                f"payload_json=@{payload_path}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"curl failed: {result.stderr}", file=sys.stderr)
            return False
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"non-JSON response: {result.stdout[:200]}", file=sys.stderr)
            return False
        if "id" not in data:
            print(f"unexpected response: {result.stdout[:200]}", file=sys.stderr)
            return False
        return True
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="flip-options-bot daily digest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the digest to stdout instead of posting to Discord",
    )
    args = parser.parse_args()

    account = _load_account_snapshot()
    closes = _load_recent_closes(hours=24)
    funnel = _load_funnel_stats(hours=24)

    # Promotion gate verdict
    settings = get_settings()
    try:
        harness = ObservationHarness(settings.run_dir)
        gate = harness.promotion_gate(settings.run_dir / "journal.db")
        gate = {
            "eligible": gate.eligible,
            **{k: getattr(gate, k) for k in dir(gate) if not k.startswith("_")},
        }
    except Exception as e:
        gate = {"eligible": False, "reason": f"observation error: {e}"}

    md = render_markdown(account, closes, funnel, gate)

    if args.dry_run:
        print(md)
        return 0

    ok = post_to_discord(md)
    if not ok:
        print(md, file=sys.stderr)
        return 1
    print("digest posted")
    return 0


if __name__ == "__main__":
    sys.exit(main())