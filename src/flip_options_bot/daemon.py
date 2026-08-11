"""Daemon entry point.

Usage:
    python -m flip_options_bot.daemon --once         # one scan cycle, then exit
    python -m flip_options_bot.daemon               # long-running loop

The daemon is NOT auto-started by the scaffold. systemctl enable is a
deliberate operator step after paper validation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Settings, get_settings
from .journal import Journal
from .risk import RiskEngine
from .signal import FunnelRecorder
from .strategies import enabled_strategies

log = logging.getLogger("flip_options_bot.daemon")


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "daemon.log"),
        ],
    )


def write_heartbeat(settings: Settings, status: dict) -> None:
    """Write the heartbeat file that the dashboard polls."""
    import json
    from datetime import datetime, timezone

    path = settings.run_dir / "heartbeat.json"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": settings.phase,
        "live_trade_enabled": settings.live_trade_enabled,
        "is_live_mode": settings.is_live(),
        "status": status,
    }
    path.write_text(json.dumps(payload, indent=2))


def run_once(settings: Settings) -> dict:
    """One scan cycle. Returns the heartbeat status dict."""
    # Placeholder: real implementation calls the scanner + funnel recorder.
    # For the scaffold, we just write the heartbeat so the dashboard can
    # verify wiring.
    strategies = enabled_strategies(settings)
    status = {
        "scan_id": "scaffold-placeholder",
        "strategies_enabled": [s.strategy_id for s in strategies],
        "watchlist_count": 0,
        "submitted_count": 0,
    }
    log.info("scaffold run_once: %s", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="flip-options-bot daemon")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle then exit (smoke-test mode)",
    )
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Load Settings from disk env, validate, print summary, exit",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.run_dir)

    if args.config_check:
        log.info("Settings loaded:")
        log.info("  phase=%s, live=%s, is_live_mode=%s",
                 settings.phase, settings.live_trade_enabled, settings.is_live())
        log.info("  paper creds: %s, live creds: %s",
                 "set" if settings.has_paper_creds() else "MISSING",
                 "set" if settings.has_live_creds() else "MISSING")
        log.info("  run_dir=%s, dashboard_port=%s",
                 settings.run_dir, settings.dashboard_port)
        log.info("  strategies enabled: %s",
                 [s.strategy_id for s in enabled_strategies(settings)])
        return 0

    if settings.is_live() and not settings.has_live_creds():
        log.error("phase=live but no live Alpaca creds. Aborting.")
        return 2

    if not settings.has_paper_creds():
        log.error("no paper Alpaca creds. Set APCA_API_KEY_ID_PAPER and APCA_API_SECRET_KEY_PAPER.")
        return 2

    log.info("flip-options-bot daemon starting")
    log.info("phase=%s, live=%s", settings.phase, settings.live_trade_enabled)

    # Scaffold components. The real wiring happens in a follow-up session.
    risk = RiskEngine(settings, settings.run_dir)
    journal = Journal(settings.run_dir)
    funnel = FunnelRecorder(settings.run_dir)

    state = risk.load_state()
    state = risk.tick_rollover(state)

    if args.once:
        status = run_once(settings)
        write_heartbeat(settings, status)
        return 0

    log.info("entering main loop, scan_interval=%ds", settings.scan_interval_s)
    try:
        while True:
            state = risk.tick_rollover(state)
            status = run_once(settings)
            write_heartbeat(settings, status)
            time.sleep(settings.scan_interval_s)
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
        return 0


if __name__ == "__main__":
    sys.exit(main())