"""Daemon entry point — wired loop: scan → risk → execute → monitor.

Usage:
    python -m flip_options_bot.daemon --once         # one cycle, then exit
    python -m flip_options_bot.daemon               # long-running loop
    python -m flip_options_bot.daemon --watchlist SYMBOLS  # comma-list override

The daemon is NOT auto-started by the scaffold. systemctl enable is a
deliberate operator step after paper validation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .broker import BrokerClient
from .config import Settings, get_settings
from .execution import Executor
from .journal import Journal
from .risk import RiskEngine
from .signal import FunnelRecorder
from .signal.scanner import Scanner
from .strategies import enabled_strategies

log = logging.getLogger("flip_options_bot.daemon")

DEFAULT_WATCHLIST = ["SPY", "QQQ", "IWM", "DIA"]


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


def run_once(
    settings: Settings,
    broker: BrokerClient,
    journal: Journal,
    risk: RiskEngine,
    funnel: FunnelRecorder,
    scanner: Scanner,
    executor: Executor,
    watchlist: list[str],
    reconcile_only: bool = False,
) -> dict:
    """One cycle: tick rollover → reconcile fills → scan → execute."""
    state = risk.load_state()
    state = risk.tick_rollover(state)

    # Step 1: reconcile real fills
    n_reconciled = executor.reconcile_fills()

    if reconcile_only:
        return {
            "scan_id": "reconcile-only",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": 0,
            "reconciled": n_reconciled,
            "submitted_count": 0,
        }

    # Step 2: scan
    result = scanner.scan(watchlist)
    log.info("scan %s: watchlist=%d candidates=%d skip=%s",
             result.funnel_row.scan_id[:8],
             result.funnel_row.watchlist_count,
             len(result.candidates),
             result.funnel_row.dominant_skip_reason)

    # Step 3: gate each candidate through risk + executor
    if settings.is_live():
        log.warning("LIVE MODE: scaffold executor will reject (confirm_live=False)")
    acct = broker.get_account()
    equity = acct["equity"]

    submitted = 0
    reasons: list[str] = []
    for sig in result.candidates:
        # Reload state inside the loop because record_open mutates open_position_count
        fresh_state = risk.load_state()
        exec_result = executor.submit_long_call(sig, equity=equity, state=fresh_state)
        if exec_result.accepted:
            submitted += 1
        else:
            reasons.append(f"{sig.symbol}:{exec_result.reason[:30]}")
            if "kill_switch" in exec_result.reason:
                log.warning("kill switch tripped — stopping scan cycle")
                break

    return {
        "scan_id": result.funnel_row.scan_id,
        "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
        "watchlist_count": result.funnel_row.watchlist_count,
        "eligible_count": result.funnel_row.eligible_count,
        "chains_fetched": len(result.funnel_row.chains_fetched),
        "chains_failed": len(result.funnel_row.chains_failed),
        "raw_signal_count": result.funnel_row.raw_signal_count,
        "submitted_count": submitted,
        "reconciled": n_reconciled,
        "denied": reasons[:5],
        "dominant_skip_reason": result.funnel_row.dominant_skip_reason,
    }


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
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Just reconcile fills from the broker, no scan",
    )
    parser.add_argument(
        "--watchlist",
        type=str,
        default=os.environ.get("FOB_WATCHLIST", ",".join(DEFAULT_WATCHLIST)),
        help="Comma-separated watchlist override",
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
    log.info("phase=%s, live=%s, watchlist=%s",
             settings.phase, settings.live_trade_enabled, args.watchlist)

    # Wire components
    try:
        broker = BrokerClient.from_settings(settings)
    except Exception as e:
        log.error("BrokerClient init failed: %s", e)
        return 3

    journal = Journal(settings.run_dir)
    risk = RiskEngine(settings, settings.run_dir)
    funnel = FunnelRecorder(settings.run_dir)
    scanner = Scanner(settings, broker, funnel)
    executor = Executor(settings, broker, journal, risk)

    watchlist = [s.strip() for s in args.watchlist.split(",") if s.strip()]
    log.info("watchlist: %s", watchlist)

    if args.once or args.reconcile_only:
        status = run_once(
            settings=settings,
            broker=broker,
            journal=journal,
            risk=risk,
            funnel=funnel,
            scanner=scanner,
            executor=executor,
            watchlist=watchlist,
            reconcile_only=args.reconcile_only,
        )
        write_heartbeat(settings, status)
        return 0

    log.info("entering main loop, scan_interval=%ds", settings.scan_interval_s)
    try:
        while True:
            status = run_once(
                settings=settings,
                broker=broker,
                journal=journal,
                risk=risk,
                funnel=funnel,
                scanner=scanner,
                executor=executor,
                watchlist=watchlist,
            )
            write_heartbeat(settings, status)
            time.sleep(settings.scan_interval_s)
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
        return 0


if __name__ == "__main__":
    sys.exit(main())