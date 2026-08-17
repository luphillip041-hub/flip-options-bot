"""Fail-closed scanner-approved candidate gate.

This module only removes candidates from the existing scanner-to-executor path.
It never creates signals or submits orders. The control-center artifact is an
advisory allow-list; broker, risk, kill, position, cooldown, and duplicate gates
remain authoritative after this gate passes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ..market_time import is_market_session_date

if TYPE_CHECKING:
    from ..config import Settings

ET = ZoneInfo("America/New_York")
EXPECTED_PURPOSE = "read_only_alpha_idea_feed_for_flip_options_bot"
VALID_REGIMES = {"mixed_chop", "risk_on", "risk_off_selective", "downtrend"}
REGIME_ENTRY_PERMISSION = {
    "mixed_chop": "A_setups_only",
    "risk_on": "calls_allowed",
    "risk_off_selective": "puts_only_or_stand_down",
    "downtrend": "puts_allowed_calls_restricted",
}
APPROVED_EARNINGS_STATUSES = {"exempt_etf", "clear_covered_calendar"}
OCC_CONTRACT_RE = re.compile(r"^([A-Z]+)\d{6}([CP])\d{8}$")
STRATEGY_DIRECTION = {
    "long_call": "bullish",
    "long_equity": "bullish",
    "long_put": "bearish",
    "bull_put_credit_spread": "bullish",
}


@dataclass(frozen=True)
class ScannerCandidateGate:
    """Immutable result of validating one alpha scanner artifact."""

    required: bool
    usable: bool
    reason: str
    artifact_path: Path
    regime: str = "unknown"
    generated_at_utc: str = ""
    artifact_age_s: float | None = None
    approved: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def approved_count(self) -> int:
        return sum(len(directions) for directions in self.approved.values())

    def allows(self, underlying: str, strategy_id: str) -> tuple[bool, str]:
        """Return whether an existing candidate may continue to normal controls."""
        if not self.required:
            return True, self.reason
        if not self.usable:
            return False, self.reason
        expected_direction = STRATEGY_DIRECTION.get(strategy_id)
        if expected_direction is None:
            return False, "scanner_gate_unknown_strategy"
        directions = self.approved.get(str(underlying).strip().upper())
        if not directions:
            return False, "scanner_gate_candidate_not_approved"
        if expected_direction not in directions:
            return False, "scanner_gate_direction_mismatch"
        return True, "scanner_gate_approved"

    def telemetry(self) -> dict[str, object]:
        return {
            "scanner_candidate_gate_required": self.required,
            "scanner_candidate_gate_usable": self.usable,
            "scanner_candidate_gate_reason": self.reason,
            "scanner_candidate_gate_regime": self.regime,
            "scanner_candidate_gate_generated_at_utc": self.generated_at_utc,
            "scanner_candidate_gate_artifact_age_s": (
                round(self.artifact_age_s, 3) if self.artifact_age_s is not None else None
            ),
            "scanner_candidate_gate_approved_count": self.approved_count,
            "scanner_candidate_gate_artifact_path": str(self.artifact_path),
        }


def _result(
    settings: Settings,
    *,
    required: bool,
    usable: bool,
    reason: str,
    regime: str = "unknown",
    generated_at_utc: str = "",
    artifact_age_s: float | None = None,
    approved: dict[str, frozenset[str]] | None = None,
) -> ScannerCandidateGate:
    return ScannerCandidateGate(
        required=required,
        usable=usable,
        reason=reason,
        artifact_path=settings.scanner_candidate_artifact_path,
        regime=regime,
        generated_at_utc=generated_at_utc,
        artifact_age_s=artifact_age_s,
        approved=approved or {},
    )


def _parse_aware_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _finite_float(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _is_regular_session_now(current: datetime) -> bool:
    current_et = current.astimezone(ET)
    return is_market_session_date(current_et.date()) and time(9, 30) <= current_et.time() < time(
        16, 0
    )


def _parse_candidates(
    top: object,
    session_date: str,
    evaluated_at: datetime,
    max_source_age_s: int,
    future_skew_s: int,
) -> dict[str, frozenset[str]] | None:
    if not isinstance(top, list):
        return None
    approved: dict[str, set[str]] = {}
    seen_symbols: set[str] = set()
    for item in top:
        if not isinstance(item, dict):
            return None
        symbol = item.get("symbol")
        direction = item.get("direction")
        tier = item.get("position_size_tier")
        confidence = item.get("confidence")
        quality = item.get("quality")
        suggested = item.get("suggested_contracts")
        no_trade_reason = item.get("no_trade_reason")
        signal_id = item.get("signal_id")
        intraday = item.get("intraday")
        option = item.get("option")
        sizing = item.get("sizing")
        earnings = item.get("earnings")
        portfolio_guard = item.get("portfolio_guard")
        data_quality = item.get("data_quality")
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
            or not isinstance(direction, str)
            or direction not in {"bullish", "bearish"}
            or not isinstance(tier, str)
            or not isinstance(confidence, str)
            or not isinstance(quality, str)
            or isinstance(suggested, bool)
            or not isinstance(suggested, int)
            or suggested < 0
            or not isinstance(no_trade_reason, str)
            or not isinstance(signal_id, str)
            or not isinstance(intraday, dict)
            or not isinstance(option, dict)
            or not isinstance(sizing, dict)
            or not isinstance(earnings, dict)
            or not isinstance(portfolio_guard, dict)
            or not isinstance(data_quality, str)
        ):
            return None
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol in seen_symbols:
            # The producer emits one directional view per underlying. Conflicting
            # or duplicate rows are ambiguous and must not broaden authorization.
            return None
        seen_symbols.add(normalized_symbol)
        maybe_actionable = (
            tier == "A"
            and confidence == "high"
            and quality == "actionable-watch"
            and data_quality == "alpaca_confirmed_read_only"
            and suggested > 0
            and not no_trade_reason.strip()
            and intraday.get("confirmed") is True
        )
        if not maybe_actionable:
            # Watch-only rows are allowed to preserve stale/failed evidence for
            # the UI, but they must never make the whole artifact unusable or
            # authorize an entry. Strict execution checks below are reserved for
            # rows that claim to be actionable.
            continue
        if not signal_id.startswith(f"{session_date}:"):
            return None
        bar_asof = _parse_aware_datetime(intraday.get("bar_asof_utc"))
        quote_asof = _parse_aware_datetime(option.get("quote_asof_utc"))
        daily_source_session = intraday.get("daily_source_session")
        reference_daily_session = intraday.get("reference_daily_session")
        daily_source_valid = (
            isinstance(daily_source_session, str)
            and bool(daily_source_session)
            and isinstance(reference_daily_session, str)
            and daily_source_session == reference_daily_session
        )
        checks = intraday.get("checks")
        required_checks = {
            "market_open",
            "fresh_bar",
            "vwap_aligned",
            "opening_range_break",
            "relative_volume",
            "trend_5m_aligned",
        }
        minute_coverage = _finite_float(intraday.get("minute_coverage"))
        intraday_complete = (
            isinstance(checks, dict)
            and required_checks.issubset(checks)
            and all(checks.get(name) is True for name in required_checks)
            and minute_coverage == 1.0
        )
        source_fresh = (
            bar_asof is not None
            and quote_asof is not None
            and -future_skew_s <= (evaluated_at - bar_asof).total_seconds() <= max_source_age_s
            and -future_skew_s <= (evaluated_at - quote_asof).total_seconds() <= max_source_age_s
            and bar_asof.astimezone(ET).date().isoformat() == session_date
            and quote_asof.astimezone(ET).date().isoformat() == session_date
            and daily_source_valid
        )
        delta = _finite_float(option.get("delta_estimate"))
        implied_volatility = _finite_float(option.get("implied_volatility"))
        expected_move = _finite_float(option.get("expected_move"))
        bid = _finite_float(option.get("bid"))
        ask = _finite_float(option.get("ask"))
        target = _finite_float(item.get("target"))
        stop = _finite_float(item.get("stop"))
        contract = option.get("contract")
        contract_match = (
            OCC_CONTRACT_RE.fullmatch(contract.strip().upper())
            if isinstance(contract, str) and contract.strip()
            else None
        )
        contract_direction_valid = (
            contract_match is not None
            and contract_match.group(1) == normalized_symbol.replace(".", "")
            and (
                (
                    direction == "bullish"
                    and contract_match.group(2) == "C"
                    and delta is not None
                    and delta > 0
                )
                or (
                    direction == "bearish"
                    and contract_match.group(2) == "P"
                    and delta is not None
                    and delta < 0
                )
            )
        )
        price_plan_valid = (
            target is not None
            and stop is not None
            and (
                (direction == "bullish" and target > stop)
                or (direction == "bearish" and target < stop)
            )
        )
        sized_contracts = sizing.get("suggested_contracts")
        execution_option_complete = (
            contract_direction_valid
            and delta is not None
            and 0 < abs(delta) <= 1
            and implied_volatility is not None
            and implied_volatility > 0
            and expected_move is not None
            and expected_move > 0
            and bid is not None
            and bid > 0
            and ask is not None
            and ask >= bid
            and price_plan_valid
            and isinstance(sized_contracts, int)
            and not isinstance(sized_contracts, bool)
            and sized_contracts == suggested
            and sized_contracts > 0
        )
        is_approved = (
            tier == "A"
            and confidence == "high"
            and quality == "actionable-watch"
            and data_quality == "alpaca_confirmed_read_only"
            and suggested > 0
            and not no_trade_reason.strip()
            and intraday.get("confirmed") is True
            and intraday_complete
            and option.get("data_quality") == "alpaca_indicative_option_snapshot"
            and sizing.get("target_inside_expected_move") is True
            and earnings.get("blocked") is False
            and earnings.get("status") in APPROVED_EARNINGS_STATUSES
            and portfolio_guard.get("allowed") is True
            and portfolio_guard.get("blocked") is False
            and not str(portfolio_guard.get("blocked_by") or "").strip()
            and execution_option_complete
            and source_fresh
        )
        if is_approved:
            approved.setdefault(normalized_symbol, set()).add(direction)
    return {symbol: frozenset(directions) for symbol, directions in approved.items()}


def load_scanner_candidate_gate(
    settings: Settings, *, now: datetime | None = None
) -> ScannerCandidateGate:
    """Load and validate the configured artifact exactly once for a scan cycle.

    Missing or unreadable data cannot prove that the current regime is outside
    mixed chop, so an enabled gate fails closed. A valid non-mixed artifact
    bypasses candidate filtering unless strict-outside mode is explicitly on.
    """
    if not settings.scanner_candidate_gate_enabled:
        return _result(
            settings,
            required=False,
            usable=True,
            reason="scanner_gate_disabled",
        )

    path = settings.scanner_candidate_artifact_path
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_missing",
        )
    except OSError:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_unreadable",
        )

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
        )
    if not isinstance(payload, dict):
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
        )

    market_regime = payload.get("market_regime")
    regime = market_regime.get("label") if isinstance(market_regime, dict) else None
    if not isinstance(regime, str) or not regime.strip():
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
        )
    regime = regime.strip()
    if regime not in VALID_REGIMES:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_unknown_regime",
            regime=regime,
        )

    if (
        payload.get("schema_version") != 2
        or payload.get("purpose") != EXPECTED_PURPOSE
        or not isinstance(market_regime, dict)
        or market_regime.get("entry_permission") != REGIME_ENTRY_PERMISSION[regime]
        or re.fullmatch(r"[0-9a-f]{20}", str(payload.get("cycle_id") or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_fingerprint") or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("policy_fingerprint") or "")) is None
    ):
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
            regime=regime,
        )
    generated = _parse_aware_datetime(payload.get("generated_at_utc"))
    if generated is None:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
            regime=regime,
        )
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age_s = (current - generated).total_seconds()
    generated_raw = str(payload["generated_at_utc"])
    if generated.astimezone(ET).date() != current.astimezone(ET).date():
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_wrong_session",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )
    if not math.isfinite(age_s) or age_s < -settings.scanner_candidate_future_skew_s:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_from_future",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )
    if age_s > settings.scanner_candidate_max_age_s:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_stale",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )

    market_state = payload.get("market_state")
    if (
        not isinstance(market_state, dict)
        or market_state.get("is_open") is not True
        or market_state.get("status") != "regular_session"
        or not _is_regular_session_now(current)
    ):
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_market_not_open",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )

    universe = payload.get("universe")
    universe_dict = universe if isinstance(universe, dict) else {}
    universe_asof = _parse_aware_datetime(universe_dict.get("data_asof_utc"))
    universe_source = universe_dict.get("source")
    universe_accepted = universe_dict.get("accepted_count")
    universe_raw = universe_dict.get("raw_count")
    universe_counts_valid = (
        isinstance(universe_accepted, int)
        and not isinstance(universe_accepted, bool)
        and 0 < universe_accepted <= 100
        and isinstance(universe_raw, int)
        and not isinstance(universe_raw, bool)
        and universe_raw >= universe_accepted
    )
    universe_fresh = (
        universe_counts_valid
        and isinstance(universe_source, str)
        and universe_source.startswith("alpaca_most_actives")
        and universe_dict.get("degraded") is False
        and universe_asof is not None
        and -settings.scanner_candidate_future_skew_s
        <= (current - universe_asof).total_seconds()
        <= settings.scanner_candidate_max_age_s
        and universe_asof.astimezone(ET).date() == generated.astimezone(ET).date()
    )
    if not universe_fresh:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_universe_stale_or_degraded",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )

    approved = _parse_candidates(
        payload.get("top"),
        generated.astimezone(ET).date().isoformat(),
        current,
        settings.scanner_candidate_max_age_s,
        settings.scanner_candidate_future_skew_s,
    )
    if approved is None:
        return _result(
            settings,
            required=True,
            usable=False,
            reason="scanner_gate_artifact_malformed",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
        )
    required = regime == "mixed_chop" or settings.scanner_candidate_strict_outside_mixed_chop
    if not required:
        return _result(
            settings,
            required=False,
            usable=True,
            reason="scanner_gate_bypass_non_mixed_chop",
            regime=regime,
            generated_at_utc=generated_raw,
            artifact_age_s=age_s,
            approved=approved,
        )
    return _result(
        settings,
        required=True,
        usable=True,
        reason="scanner_gate_ok",
        regime=regime,
        generated_at_utc=generated_raw,
        artifact_age_s=age_s,
        approved=approved,
    )
