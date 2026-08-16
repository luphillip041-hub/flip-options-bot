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
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from .broker import BrokerClient
from .config import Settings, get_settings
from .execution import Closer, Executor
from .journal import Journal
from .logging_json import attach as attach_json_logger
from .market_time import is_entry_window, is_market_open
from .monitor.position_monitor import PositionMonitor
from .risk import RiskEngine
from .signal import FunnelRecorder
from .signal.candidate_gate import load_scanner_candidate_gate
from .signal.scanner import Scanner
from .strategies import enabled_strategies
from .strategies.long_call import LongCallSignal
from .strategies.long_equity import LongEquitySignal
from .strategies.long_put import LongPutSignal

log = logging.getLogger("flip_options_bot.daemon")

# Liquid option underlyings only: broad ETFs plus mega-cap names with deep chains.
# This is the default paper-forward universe; operators can still override via
# FOB_WATCHLIST for narrower studies.
DEFAULT_WATCHLIST = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "XLF",
    "XLE",
    "XLK",
    "SMH",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "AMD",
]


def unresolved_stale_close_gate(broker) -> tuple[bool, str]:
    """Return whether stale prior-session close orders must block new entries.

    Exits/monitoring should keep running, but new BUY entries should not stack
    onto an account that still has old bot-owned close SELLs stuck in
    PENDING_CANCEL. Monday's first job is clearing/repricing held positions,
    not adding fresh risk on top of unresolved exits.
    """
    try:
        if hasattr(broker, "list_open_orders_or_raise"):
            orders = broker.list_open_orders_or_raise()
        elif hasattr(broker, "list_open_orders"):
            orders = broker.list_open_orders()
        else:
            return False, ""
    except Exception as exc:
        return True, f"stale_close_lookup_failed:{exc}"

    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    blocked: list[str] = []
    for order in orders or []:
        coid = str(getattr(order, "client_order_id", "") or "")
        side = str(getattr(order, "side", "") or "").upper()
        status = str(getattr(order, "status", "") or "").upper()
        submitted = getattr(order, "submitted_at", None)
        if not coid.startswith("close-") or "SELL" not in side or "PENDING_CANCEL" not in status:
            continue
        if submitted is not None:
            if getattr(submitted, "tzinfo", None) is None:
                submitted = submitted.replace(tzinfo=UTC)
            if submitted.astimezone(UTC) > cutoff:
                continue
        symbol = str(getattr(order, "symbol", "") or "unknown")
        blocked.append(symbol)
    if blocked:
        return True, "unresolved_stale_close_orders:" + ",".join(sorted(set(blocked)))
    return False, ""


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
    # Also emit a structured JSON line per record. The JSON handler is the
    # authoritative machine-readable log; the text daemon.log above is the
    # human-readable companion.
    attach_json_logger(run_dir)


def write_heartbeat(settings: Settings, status: dict) -> None:
    import json
    from datetime import datetime

    path = settings.run_dir / "heartbeat.json"
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "phase": settings.phase,
        "live_trade_enabled": settings.live_trade_enabled,
        "is_live_mode": settings.is_live(),
        "status": status,
    }
    path.write_text(json.dumps(payload, indent=2))


def _candidate_quality_key(
    sig: LongCallSignal | LongPutSignal | LongEquitySignal, settings: Settings
) -> tuple[float, int, float]:
    """Global submission ordering: quality first, then preferred DTE.

    Scanner does per-underlying chain quality and OTM sorting; daemon uses this
    to avoid static watchlist order crowding out stronger later symbols.
    """
    conviction = float(getattr(sig, "conviction", 0.0))
    dte = int(getattr(sig, "dte", settings.target_dte))
    high_reward_bonus = 0.05 if settings.long_option_high_reward_mode else 0.0
    option_bonus = (
        high_reward_bonus if getattr(sig, "strategy_id", "") in {"long_call", "long_put"} else 0.0
    )
    return (conviction + option_bonus, -abs(dte - settings.target_dte), -dte)


def _rank_directional_candidates(
    candidates: list[LongCallSignal | LongPutSignal | LongEquitySignal],
    settings: Settings,
) -> list[LongCallSignal | LongPutSignal | LongEquitySignal]:
    return sorted(candidates, key=lambda sig: _candidate_quality_key(sig, settings), reverse=True)


def _scanner_gate_config_telemetry(settings: Settings) -> dict[str, object]:
    return {
        "scanner_candidate_gate_enabled": settings.scanner_candidate_gate_enabled,
        "scanner_candidate_gate_artifact_path": str(settings.scanner_candidate_artifact_path),
        "scanner_candidate_gate_max_age_s": settings.scanner_candidate_max_age_s,
        "scanner_candidate_gate_future_skew_s": settings.scanner_candidate_future_skew_s,
        "scanner_candidate_gate_strict_outside_mixed_chop": (
            settings.scanner_candidate_strict_outside_mixed_chop
        ),
    }


def run_once(
    settings: Settings,
    broker: BrokerClient,
    journal: Journal,
    risk: RiskEngine,
    funnel: FunnelRecorder,
    scanner: Scanner,
    executor: Executor,
    monitor: PositionMonitor,
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
            **_scanner_gate_config_telemetry(settings),
            "scan_id": "reconcile-only",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": len(watchlist),
            "reconciled": n_reconciled,
            "submitted_count": 0,
            "yfinance_confirm_1dte_enabled": settings.yfinance_confirm_1dte_enabled,
            "yfinance_confirm_min_dte": settings.yfinance_confirm_min_dte,
            "yfinance_strict_gate": settings.yfinance_strict_gate,
            "yfinance_require_current_trade_date_for_volume_bonus": settings.yfinance_require_current_trade_date_for_volume_bonus,
            "directional_underlying_loss_lockout_dollar": settings.directional_underlying_loss_lockout_dollar,
            "tp_multiplier": settings.tp_multiplier,
            "tp_full_multiplier": settings.tp_full_multiplier,
            "trailing_arm_pct": settings.trailing_arm_pct,
            "trailing_retention": settings.trailing_retention,
            "profit_floor_pct": settings.profit_floor_pct,
            "min_tp_profit_dollar": settings.min_tp_profit_dollar,
            "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
            "runner_trailing_retention": settings.runner_trailing_retention,
            "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        }

    # Step 2: scan
    # Only scan during market hours (09:30-16:00 ET) AND inside the entry
    # window (09:45-15:45 ET — first/last 15 min are too volatile).
    if not is_market_open():
        log.info("scan skipped: market closed")
        funnel.emit_skip(reason="market_closed")
        return {
            **_scanner_gate_config_telemetry(settings),
            "scan_id": "market-closed",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": len(watchlist),
            "submitted_count": 0,
            "reconciled": n_reconciled,
            "dominant_skip_reason": "market_closed",
            "yfinance_confirm_1dte_enabled": settings.yfinance_confirm_1dte_enabled,
            "yfinance_confirm_min_dte": settings.yfinance_confirm_min_dte,
            "yfinance_strict_gate": settings.yfinance_strict_gate,
            "yfinance_require_current_trade_date_for_volume_bonus": settings.yfinance_require_current_trade_date_for_volume_bonus,
            "directional_underlying_loss_lockout_dollar": settings.directional_underlying_loss_lockout_dollar,
            "tp_multiplier": settings.tp_multiplier,
            "tp_full_multiplier": settings.tp_full_multiplier,
            "trailing_arm_pct": settings.trailing_arm_pct,
            "trailing_retention": settings.trailing_retention,
            "profit_floor_pct": settings.profit_floor_pct,
            "min_tp_profit_dollar": settings.min_tp_profit_dollar,
            "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
            "runner_trailing_retention": settings.runner_trailing_retention,
            "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        }
    if not is_entry_window():
        log.info("scan skipped: outside entry window (09:45-15:45 ET)")
        funnel.emit_skip(reason="outside_entry_window")
        return {
            **_scanner_gate_config_telemetry(settings),
            "scan_id": "outside-entry",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": len(watchlist),
            "submitted_count": 0,
            "reconciled": n_reconciled,
            "dominant_skip_reason": "outside_entry_window",
            "yfinance_confirm_1dte_enabled": settings.yfinance_confirm_1dte_enabled,
            "yfinance_confirm_min_dte": settings.yfinance_confirm_min_dte,
            "yfinance_strict_gate": settings.yfinance_strict_gate,
            "yfinance_require_current_trade_date_for_volume_bonus": settings.yfinance_require_current_trade_date_for_volume_bonus,
            "directional_underlying_loss_lockout_dollar": settings.directional_underlying_loss_lockout_dollar,
            "tp_multiplier": settings.tp_multiplier,
            "tp_full_multiplier": settings.tp_full_multiplier,
            "trailing_arm_pct": settings.trailing_arm_pct,
            "trailing_retention": settings.trailing_retention,
            "profit_floor_pct": settings.profit_floor_pct,
            "min_tp_profit_dollar": settings.min_tp_profit_dollar,
            "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
            "runner_trailing_retention": settings.runner_trailing_retention,
            "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        }
    stale_close_blocked, stale_close_reason = unresolved_stale_close_gate(broker)
    if stale_close_blocked:
        log.warning("scan skipped: %s", stale_close_reason)
        funnel.emit_skip(reason="unresolved_stale_close_orders")
        return {
            **_scanner_gate_config_telemetry(settings),
            "scan_id": "stale-close-hold",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": len(watchlist),
            "submitted_count": 0,
            "reconciled": n_reconciled,
            "dominant_skip_reason": "unresolved_stale_close_orders",
            "denied": [stale_close_reason],
            "stale_close_entry_hold": True,
            "yfinance_confirm_1dte_enabled": settings.yfinance_confirm_1dte_enabled,
            "yfinance_confirm_min_dte": settings.yfinance_confirm_min_dte,
            "yfinance_strict_gate": settings.yfinance_strict_gate,
            "yfinance_require_current_trade_date_for_volume_bonus": settings.yfinance_require_current_trade_date_for_volume_bonus,
            "directional_underlying_loss_lockout_dollar": settings.directional_underlying_loss_lockout_dollar,
            "tp_multiplier": settings.tp_multiplier,
            "tp_full_multiplier": settings.tp_full_multiplier,
            "trailing_arm_pct": settings.trailing_arm_pct,
            "trailing_retention": settings.trailing_retention,
            "profit_floor_pct": settings.profit_floor_pct,
            "min_tp_profit_dollar": settings.min_tp_profit_dollar,
            "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
            "runner_trailing_retention": settings.runner_trailing_retention,
            "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        }

    candidate_gate = load_scanner_candidate_gate(settings)
    if candidate_gate.required and not candidate_gate.usable:
        log.warning("scan skipped: %s", candidate_gate.reason)
        funnel.emit_skip(reason=candidate_gate.reason)
        return {
            **_scanner_gate_config_telemetry(settings),
            **candidate_gate.telemetry(),
            "scan_id": "scanner-gate-hold",
            "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
            "watchlist_count": len(watchlist),
            "submitted_count": 0,
            "reconciled": n_reconciled,
            "dominant_skip_reason": candidate_gate.reason,
            "denied": [candidate_gate.reason],
            "tp_multiplier": settings.tp_multiplier,
            "tp_full_multiplier": settings.tp_full_multiplier,
            "trailing_arm_pct": settings.trailing_arm_pct,
            "trailing_retention": settings.trailing_retention,
            "profit_floor_pct": settings.profit_floor_pct,
            "min_tp_profit_dollar": settings.min_tp_profit_dollar,
            "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
            "runner_trailing_retention": settings.runner_trailing_retention,
            "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        }

    result = scanner.scan(watchlist, candidate_gate=candidate_gate)
    log.info(
        "scan %s: watchlist=%d candidates=%d skip=%s",
        result.funnel_row.scan_id[:8],
        result.funnel_row.watchlist_count,
        len(result.candidates),
        result.funnel_row.dominant_skip_reason,
    )

    # Step 2b: BPCS scan (parallel, only if enabled)
    bpcs_result = None
    if settings.bpcs_enabled:
        bpcs_result = scanner.scan_bpcs(watchlist, candidate_gate=candidate_gate)
        log.info(
            "bpcs scan %s: candidates=%d skip=%s",
            bpcs_result.funnel_row.scan_id[:8],
            len(bpcs_result.candidates),
            bpcs_result.funnel_row.dominant_skip_reason,
        )

    # Step 3: gate each candidate through risk + executor
    if settings.is_live():
        log.warning("LIVE MODE: scaffold executor will reject (confirm_live=False)")
    acct = broker.get_account()
    equity = acct["equity"]

    submitted = 0
    reasons: list[str] = []
    max_submissions = max(1, settings.max_submissions_per_scan)
    ranked_candidates = _rank_directional_candidates(result.candidates, settings)
    for sig in ranked_candidates:
        if submitted >= max_submissions:
            reasons.append(f"scan_submission_cap:{submitted}/{max_submissions}")
            break
        # Reload state inside the loop because record_open mutates open_position_count
        fresh_state = risk.load_state()
        if sig.strategy_id == "long_equity":
            exec_result = executor.submit_long_equity(
                cast(LongEquitySignal, sig), equity=equity, state=fresh_state
            )
        elif sig.strategy_id == "long_put":
            exec_result = executor.submit_long_put(
                cast(LongPutSignal, sig), equity=equity, state=fresh_state
            )
        else:
            exec_result = executor.submit_long_call(
                cast(LongCallSignal, sig), equity=equity, state=fresh_state
            )
        if exec_result.accepted:
            submitted += 1
        else:
            reasons.append(f"{sig.symbol}:{exec_result.reason[:30]}")
            if "kill_switch" in exec_result.reason:
                log.warning("kill switch tripped — stopping scan cycle")
                break

    # Step 3b: gate each BPCS candidate
    if bpcs_result is not None:
        for sig in bpcs_result.candidates:
            if submitted >= max_submissions:
                reasons.append(f"scan_submission_cap:{submitted}/{max_submissions}")
                break
            fresh_state = risk.load_state()
            exec_result = executor.submit_bull_put_spread(
                sig,
                equity=equity,
                state=fresh_state,
                short_put_symbol=sig.short_put_symbol,
                long_put_symbol=sig.long_put_symbol,
            )
            if exec_result.accepted:
                submitted += 1
            else:
                reasons.append(
                    f"BPCS {sig.short_strike}/{sig.long_strike}:{exec_result.reason[:30]}"
                )
                if "kill_switch" in exec_result.reason:
                    log.warning("kill switch tripped — stopping scan cycle")
                    break

    scanner_gate_denied_reasons: dict[str, int] = {}
    funnel_rows = [result.funnel_row]
    if bpcs_result is not None:
        funnel_rows.append(bpcs_result.funnel_row)
    for funnel_row in funnel_rows:
        raw_denials = funnel_row.extras.get("scanner_candidate_gate_denied_reasons", {})
        if isinstance(raw_denials, dict):
            for reason, count in raw_denials.items():
                if isinstance(reason, str) and isinstance(count, int):
                    scanner_gate_denied_reasons[reason] = (
                        scanner_gate_denied_reasons.get(reason, 0) + count
                    )
    dominant_skip_reason = result.funnel_row.dominant_skip_reason
    if (
        dominant_skip_reason == "no_candidates"
        and bpcs_result is not None
        and bpcs_result.funnel_row.dominant_skip_reason != "no_candidates"
    ):
        dominant_skip_reason = bpcs_result.funnel_row.dominant_skip_reason

    return {
        **_scanner_gate_config_telemetry(settings),
        **candidate_gate.telemetry(),
        "scan_id": result.funnel_row.scan_id,
        "strategies_enabled": [s.strategy_id for s in enabled_strategies(settings)],
        "watchlist_count": result.funnel_row.watchlist_count,
        "eligible_count": result.funnel_row.eligible_count,
        "chains_fetched": len(result.funnel_row.chains_fetched),
        "chains_failed": len(result.funnel_row.chains_failed),
        "raw_signal_count": result.funnel_row.raw_signal_count,
        "ranked_candidate_count": len(ranked_candidates),
        "max_submissions_per_scan": max_submissions,
        "yfinance_confirm_1dte_enabled": settings.yfinance_confirm_1dte_enabled,
        "yfinance_confirm_min_dte": settings.yfinance_confirm_min_dte,
        "yfinance_strict_gate": settings.yfinance_strict_gate,
        "yfinance_require_current_trade_date_for_volume_bonus": settings.yfinance_require_current_trade_date_for_volume_bonus,
        "directional_underlying_loss_lockout_dollar": settings.directional_underlying_loss_lockout_dollar,
        "tp_multiplier": settings.tp_multiplier,
        "tp_full_multiplier": settings.tp_full_multiplier,
        "trailing_arm_pct": settings.trailing_arm_pct,
        "trailing_retention": settings.trailing_retention,
        "profit_floor_pct": settings.profit_floor_pct,
        "min_tp_profit_dollar": settings.min_tp_profit_dollar,
        "runner_trailing_arm_pct": settings.runner_trailing_arm_pct,
        "runner_trailing_retention": settings.runner_trailing_retention,
        "runner_profit_floor_pct": settings.runner_profit_floor_pct,
        "submitted_count": submitted,
        "reconciled": n_reconciled,
        "denied": reasons[:5],
        "scanner_candidate_gate_denied_count": sum(scanner_gate_denied_reasons.values()),
        "scanner_candidate_gate_denied_reasons": dict(sorted(scanner_gate_denied_reasons.items())),
        "dominant_skip_reason": dominant_skip_reason,
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
        "--record-day",
        action="store_true",
        help="Mark today as a paper market day in the observation harness (cron-only)",
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
        log.info(
            "  phase=%s, live=%s, is_live_mode=%s",
            settings.phase,
            settings.live_trade_enabled,
            settings.is_live(),
        )
        log.info(
            "  paper creds: %s, live creds: %s",
            "set" if settings.has_paper_creds() else "MISSING",
            "set" if settings.has_live_creds() else "MISSING",
        )
        log.info("  run_dir=%s, dashboard_port=%s", settings.run_dir, settings.dashboard_port)
        log.info(
            "  throughput: max_positions=%s, max_submissions_per_scan=%s, scan_interval=%ss",
            settings.max_positions,
            settings.max_submissions_per_scan,
            settings.scan_interval_s,
        )
        log.info(
            "  high_reward=%s ladder=%s max_contract=$%s",
            settings.long_option_high_reward_mode,
            settings.long_option_otm_ladder_pct,
            settings.max_contract_dollar,
        )
        log.info(
            "  yfinance_1dte_confirm=%s min_dte=%s strict=%s current_trade_date_volume_bonus=%s",
            settings.yfinance_confirm_1dte_enabled,
            settings.yfinance_confirm_min_dte,
            settings.yfinance_strict_gate,
            settings.yfinance_require_current_trade_date_for_volume_bonus,
        )
        log.info(
            "  scanner_candidate_gate=%s path=%s max_age=%ss strict_outside_mixed=%s",
            settings.scanner_candidate_gate_enabled,
            settings.scanner_candidate_artifact_path,
            settings.scanner_candidate_max_age_s,
            settings.scanner_candidate_strict_outside_mixed_chop,
        )
        log.info("  strategies enabled: %s", [s.strategy_id for s in enabled_strategies(settings)])
        return 0

    if settings.is_live() and not settings.has_live_creds():
        log.error("phase=live but no live Alpaca creds. Aborting.")
        return 2

    if not settings.has_paper_creds():
        log.error("no paper Alpaca creds. Set APCA_API_KEY_ID_PAPER and APCA_API_SECRET_KEY_PAPER.")
        return 2

    log.info("flip-options-bot daemon starting")
    log.info(
        "phase=%s, live=%s, watchlist=%s",
        settings.phase,
        settings.live_trade_enabled,
        args.watchlist,
    )

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
    closer = Closer(settings, broker, journal, risk)
    monitor = PositionMonitor(settings, broker, journal, risk, closer)

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
            monitor=monitor,
            watchlist=watchlist,
            reconcile_only=args.reconcile_only,
        )
        write_heartbeat(settings, status)
        return 0

    if args.record_day:
        from .observation import ObservationHarness

        harness = ObservationHarness(settings.run_dir)
        recorded = harness.record_market_day()
        gate = harness.promotion_gate(settings.run_dir / "journal.db")
        log.info(
            "record-day: recorded=%s, days=%d/%d, eligible=%s reason=%s",
            recorded,
            gate.market_days_recorded,
            gate.min_market_days,
            gate.eligible,
            gate.reason,
        )
        print(harness.render_digest(settings.run_dir / "journal.db"))
        return 0

    # Spawn the position monitor in a separate thread. The monitor runs on
    # its own interval (`position_monitor_interval_s`) so the scan loop is
    # not blocked by per-second mark polls. The monitor is read-mostly — it
    # only mutates state via Closer.flatten_position() (which is idempotent
    # on client_order_id) and reconcile_fills() (also idempotent).
    stop_event = threading.Event()
    consecutive_failures = [0]
    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(settings, risk, broker, closer, executor, monitor, stop_event, consecutive_failures),
        name="position-monitor",
        daemon=True,
    )
    monitor_thread.start()

    log.info(
        "entering main loop, scan_interval=%ds, monitor_interval=%ds",
        settings.scan_interval_s,
        settings.position_monitor_interval_s,
    )
    try:
        while not stop_event.is_set():
            status = run_once(
                settings=settings,
                broker=broker,
                journal=journal,
                risk=risk,
                funnel=funnel,
                scanner=scanner,
                executor=executor,
                monitor=monitor,
                watchlist=watchlist,
            )
            write_heartbeat(settings, status)
            stop_event.wait(settings.scan_interval_s)
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)
    return 0


def _monitor_loop(
    settings: Settings,
    risk: RiskEngine,
    broker: BrokerClient,
    closer: Closer,
    executor: Executor,
    monitor: PositionMonitor,
    stop_event: threading.Event,
    consecutive_failures: list[int] | None = None,
) -> None:
    """Background thread: tick the position monitor on its own cadence.

    Each tick:
    1. tick_rollover
    2. evaluate caps → if tripped, flatten all + log
    3. cancel_stale_orders (orders ACCEPTED > 2 min that never filled)
    4. monitor.tick (SL/TP/trailing/EOD)
    5. reconcile_fills (canonical broker fills)
    """
    log.info("position-monitor thread started (interval=%ds)", settings.position_monitor_interval_s)
    while not stop_event.is_set():
        try:
            state = risk.load_state()
            state = risk.tick_rollover(state)

            # === Loss-cap watchdog ===
            acct = broker.get_account()
            equity = float(acct.get("equity", 0))
            tripped = risk.evaluate_caps(state, equity)
            if tripped:
                log.warning("loss cap tripped: %s — flattening all", state.kill_reason)
                closer.flatten_all(state, reason=f"kill_switch: {state.kill_reason}")

            # === Stale-order cleanup ===
            # Critical: orders that sit at ACCEPTED for >2 min without a fill
            # are dead paper-account limbo. Cancel them so we don't have
            # blocked position-count slots or phantom exposure.
            n_cancelled = executor.cancel_stale_orders(older_than_seconds=120)
            if n_cancelled:
                log.warning("stale-order cleanup: %d orders cancelled", n_cancelled)

            # === Broker-canonical fill reconciliation before monitor ===
            # Prevent a fast-filled close from disappearing from open orders
            # while the journal still looks open.
            executor.reconcile_fills()

            # === Per-position monitor ===
            tick = monitor.tick(state)
            if tick.closes_triggered:
                log.warning(
                    "monitor tick closed %d positions (%s)",
                    tick.closes_triggered,
                    tick.reasons,
                )
            executor.reconcile_fills()
            if consecutive_failures is not None:
                consecutive_failures[0] = 0
        except Exception as e:
            log.error("monitor tick failed: %s", e, exc_info=True)
            if consecutive_failures is not None:
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= 5:
                    log.critical("monitor loop failed 5 cycles in a row — requesting exit")
                    stop_event.set()
        stop_event.wait(settings.position_monitor_interval_s)
    log.info("position-monitor thread exiting")


if __name__ == "__main__":
    sys.exit(main())
