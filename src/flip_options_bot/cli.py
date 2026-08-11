"""CLI: paper-trade status, heartbeat inspection, backtest runs.

The CLI is for iteration and smoke tests. After the daemon is installed as
a systemd service, use the CLI only for `status` / `heartbeat` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings


def cmd_status(args) -> int:
    settings = get_settings()
    hb_path = settings.run_dir / "heartbeat.json"
    if not hb_path.exists():
        print(f"no heartbeat at {hb_path}")
        return 1
    print(json.dumps(json.loads(hb_path.read_text()), indent=2))
    return 0


def cmd_heartbeat(args) -> int:
    settings = get_settings()
    print(f"phase:          {settings.phase}")
    print(f"live_enabled:   {settings.live_trade_enabled}")
    print(f"is_live_mode:   {settings.is_live()}")
    print(f"run_dir:        {settings.run_dir}")
    print(f"dashboard_port: {settings.dashboard_port}")
    print(f"paper_creds:    {'set' if settings.has_paper_creds() else 'MISSING'}")
    print(f"live_creds:     {'set' if settings.has_live_creds() else 'MISSING'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="flip-options-bot CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="print latest heartbeat.json").set_defaults(func=cmd_status)
    sub.add_parser("heartbeat", help="print settings snapshot").set_defaults(func=cmd_heartbeat)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())