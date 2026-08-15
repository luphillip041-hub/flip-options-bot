"""End-of-day market-day recorder for paper observation.

Runs at 16:30 ET every weekday. Records the day in the observation
harness if it was a clean trading day (no kill_switch, no broker
errors in the daemon log).

Idempotent: re-running for the same date is a no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BOT_ROOT = Path("/root/flip/projects/flip-options-bot")
if str(_BOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BOT_ROOT / "src"))

from flip_options_bot.config import get_settings  # noqa: E402
from flip_options_bot.market_time import now_utc, to_et  # noqa: E402
from flip_options_bot.observation import ObservationHarness  # noqa: E402
from flip_options_bot.risk import RiskEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="record paper market day")
    parser.add_argument(
        "--date",
        help="Override date (YYYY-MM-DD); default = today ET",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Record even if kill_switch tripped or non-weekday",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.date:
        target = args.date
    else:
        target = to_et(now_utc()).strftime("%Y-%m-%d")

    # Reject non-weekdays unless --force
    if not args.force:
        # Cheap weekday check from the date string itself
        from datetime import date

        wd = date.fromisoformat(target).weekday()
        if wd >= 5:
            print(f"{target} is a weekend — skipping (use --force to override)")
            return 0
        # Refuse if kill switch is active
        risk = RiskEngine(settings, settings.run_dir)
        state = risk.load_state()
        if state.kill_switch and not args.force:
            print(f"kill switch active ({state.kill_reason}) — skipping (use --force to override)")
            return 1

    harness = ObservationHarness(settings.run_dir)
    written = harness.record_market_day(target)
    print(f"recorded {target}: {'new' if written else 'already existed'}")
    print(f"total market days recorded: {len(harness.state['market_days'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
