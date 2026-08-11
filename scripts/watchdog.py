"""Daemon watchdog — restart flip-options-bot if it dies, alert Discord.

Runs every 5 minutes via cron. Checks `systemctl is-active`. If inactive:

1. `systemctl restart flip-options-bot.service`
2. Wait 5 seconds, check again.
3. If still inactive, post a Discord alert with the journalctl tail.
4. Always post a heartbeat line to /var/log/flip-options-bot-watchdog.log.

Idempotent: nothing changes if the daemon is already running.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CHANNEL_ID = "1531705542538690741"


def _load_bot_token() -> str:
    from dotenv import dotenv_values

    env_path = Path("/root/.config/flip-options-bot/.env")
    if env_path.exists():
        env = dotenv_values(env_path)
    else:
        env = {}
    return os.environ.get("FOB_DISCORD_BOT_TOKEN") or env.get("FOB_DISCORD_BOT_TOKEN", "")


def _post_to_discord(content: str, token: str) -> bool:
    if not token:
        print("FOB_DISCORD_BOT_TOKEN not set", file=sys.stderr)
        return False
    import json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"content": content}, f)
        payload_path = f.name
    try:
        r = subprocess.run(
            [
                "curl", "-sS", "-X", "POST",
                f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                "-H", f"Authorization: Bot {token}",
                "-F", f"payload_json=@{payload_path}",
            ],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0 and '"id"' in r.stdout
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass


def _service_active() -> bool:
    r = subprocess.run(
        ["systemctl", "is-active", "flip-options-bot.service"],
        capture_output=True, text=True, timeout=10,
    )
    return r.returncode == 0 and r.stdout.strip() == "active"


def _journal_tail(lines: int = 30) -> str:
    r = subprocess.run(
        ["journalctl", "-u", "flip-options-bot.service", "-n", str(lines), "--no-pager"],
        capture_output=True, text=True, timeout=10,
    )
    return r.stdout[:3500]  # cap


def main() -> int:
    parser = argparse.ArgumentParser(description="flip-options-bot watchdog")
    parser.add_argument("--no-restart", action="store_true",
                        help="Don't restart — only alert")
    args = parser.parse_args()

    log_path = Path("/var/log/flip-options-bot-watchdog.log")
    token = _load_bot_token()

    if _service_active():
        # Healthy — silent noop.
        with log_path.open("a") as f:
            f.write(f"OK active\n")
        return 0

    msg = "⚠️ **flip-options-bot daemon is INACTIVE**"
    print(msg, file=sys.stderr)

    if not args.no_restart:
        subprocess.run(
            ["systemctl", "restart", "flip-options-bot.service"],
            capture_output=True, text=True, timeout=30,
        )
        import time
        time.sleep(5)

    if _service_active():
        msg = "✅ **flip-options-bot daemon was DOWN but restarted successfully**"
    else:
        tail = _journal_tail(40)
        msg = (
            "🚨 **flip-options-bot daemon is DOWN and restart FAILED**\n\n"
            "Last 40 lines of journal:\n```\n"
            + tail
            + "\n```"
        )

    posted = _post_to_discord(msg, token)
    with log_path.open("a") as f:
        f.write(f"DEAD→{msg[:50]}... discord_posted={posted}\n")
    return 0 if posted else 1


if __name__ == "__main__":
    sys.exit(main())