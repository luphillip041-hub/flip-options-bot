from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from flip_options_bot.config import Settings
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.candidate_gate import load_scanner_candidate_gate
from flip_options_bot.signal.scanner import Scanner
from flip_options_bot.strategies.bull_put_credit import BullPutSpreadSignal
from flip_options_bot.strategies.long_call import LongCallSignal
from flip_options_bot.strategies.long_put import LongPutSignal

NOW = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


def _approved_candidate(symbol: str = "SPY", direction: str = "bullish") -> dict:
    return {
        "signal_id": f"2026-08-17:{symbol}:{direction}:100.0",
        "symbol": symbol,
        "direction": direction,
        "position_size_tier": "A",
        "confidence": "high",
        "quality": "actionable-watch",
        "data_quality": "alpaca_confirmed_read_only",
        "suggested_contracts": 2,
        "target": 104.0 if direction == "bullish" else 96.0,
        "stop": 98.0 if direction == "bullish" else 102.0,
        "no_trade_reason": "",
        "intraday": {
            "confirmed": True,
            "bar_asof_utc": (NOW - timedelta(minutes=3)).isoformat(),
            "daily_source_session": "2026-08-14",
            "reference_daily_session": "2026-08-14",
            "minute_coverage": 1.0,
            "checks": {
                "market_open": True,
                "fresh_bar": True,
                "vwap_aligned": True,
                "opening_range_break": True,
                "relative_volume": True,
                "trend_5m_aligned": True,
            },
        },
        "option": {
            "contract": f"{symbol}260821{'C' if direction == 'bullish' else 'P'}00100000",
            "data_quality": "alpaca_indicative_option_snapshot",
            "quote_asof_utc": (NOW - timedelta(minutes=3)).isoformat(),
            "bid": 1.0,
            "ask": 1.1,
            "expected_move": 8.0,
            "delta_estimate": 0.42 if direction == "bullish" else -0.42,
            "implied_volatility": 0.24,
        },
        "sizing": {
            "suggested_contracts": 2,
            "target_inside_expected_move": True,
        },
        "earnings": {"status": "clear_covered_calendar", "blocked": False},
        "portfolio_guard": {"allowed": True, "blocked": False, "blocked_by": ""},
    }


def _write_artifact(
    path: Path,
    *,
    regime: str = "mixed_chop",
    generated_at: datetime = NOW - timedelta(minutes=2),
    top: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "purpose": "read_only_alpha_idea_feed_for_flip_options_bot",
                "cycle_id": "0123456789abcdefabcd",
                "source_fingerprint": "a" * 64,
                "policy_fingerprint": "b" * 64,
                "generated_at_utc": generated_at.isoformat(),
                "market_regime": {
                    "label": regime,
                    "entry_permission": {
                        "mixed_chop": "A_setups_only",
                        "risk_on": "calls_allowed",
                        "risk_off_selective": "puts_only_or_stand_down",
                        "downtrend": "puts_allowed_calls_restricted",
                    }.get(regime, "A_setups_only"),
                },
                "market_state": {"is_open": True, "status": "regular_session"},
                "universe": {
                    "source": "alpaca_most_actives_top_100",
                    "data_asof_utc": (generated_at - timedelta(minutes=1)).isoformat(),
                    "raw_count": 40,
                    "accepted_count": 40,
                    "degraded": False,
                },
                "top": top if top is not None else [_approved_candidate()],
            }
        )
    )


def _settings(tmp_path: Path, artifact: Path, **overrides) -> Settings:
    return Settings(
        run_dir=tmp_path,
        scanner_candidate_artifact_path=artifact,
        scanner_candidate_max_age_s=900,
        **overrides,
    )


def test_mixed_chop_requires_approved_a_confirmed_matching_direction(tmp_path: Path):
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact)
    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is True
    assert gate.reason == "scanner_gate_ok"
    assert gate.allows("SPY", "long_call") == (True, "scanner_gate_approved")
    assert gate.allows("SPY", "long_equity") == (True, "scanner_gate_approved")
    assert gate.allows("SPY", "long_put") == (False, "scanner_gate_direction_mismatch")
    assert gate.allows("SPY", "bull_put_credit_spread") == (True, "scanner_gate_approved")
    assert gate.allows("QQQ", "long_call") == (False, "scanner_gate_candidate_not_approved")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "scanner_gate_artifact_missing"),
        ("malformed_json", "scanner_gate_artifact_malformed"),
        ("malformed_candidate", "scanner_gate_artifact_malformed"),
        ("stale", "scanner_gate_artifact_stale"),
        ("wrong_session", "scanner_gate_artifact_wrong_session"),
        ("future", "scanner_gate_artifact_from_future"),
    ],
)
def test_required_gate_fails_closed_for_bad_artifacts(tmp_path: Path, mutation: str, reason: str):
    artifact = tmp_path / "alpha.json"
    if mutation == "malformed_json":
        artifact.write_text("{")
    elif mutation == "malformed_candidate":
        candidate = _approved_candidate()
        candidate.pop("confidence")
        _write_artifact(artifact, top=[candidate])
    elif mutation == "stale":
        _write_artifact(artifact, generated_at=NOW - timedelta(minutes=30))
    elif mutation == "wrong_session":
        _write_artifact(artifact, generated_at=NOW - timedelta(days=1))
    elif mutation == "future":
        _write_artifact(artifact, generated_at=NOW + timedelta(minutes=1))

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == reason
    assert gate.allows("SPY", "long_call") == (False, reason)


def test_non_mixed_regime_requires_fresh_valid_artifact_before_bypass(tmp_path: Path):
    artifact = tmp_path / "alpha.json"
    _write_artifact(
        artifact,
        regime="risk_on",
        generated_at=NOW - timedelta(minutes=30),
        top=[_approved_candidate("QQQ", "bearish")],
    )
    stale = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)
    assert stale.required is True
    assert stale.usable is False
    assert stale.reason == "scanner_gate_artifact_stale"

    _write_artifact(
        artifact,
        regime="risk_on",
        generated_at=NOW - timedelta(minutes=2),
        top=[_approved_candidate("QQQ", "bearish")],
    )
    ordinary = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)
    strict = load_scanner_candidate_gate(
        _settings(tmp_path, artifact, scanner_candidate_strict_outside_mixed_chop=True),
        now=NOW,
    )

    assert ordinary.required is False
    assert ordinary.usable is True
    assert ordinary.allows("SPY", "long_call") == (True, "scanner_gate_bypass_non_mixed_chop")
    assert strict.required is True
    assert strict.usable is True
    assert strict.reason == "scanner_gate_ok"


def test_only_a_high_confidence_actionable_candidates_are_approved(tmp_path: Path):
    artifact = tmp_path / "alpha.json"
    candidates = []
    for symbol, field, value in (
        ("B_TIER", "position_size_tier", "B"),
        ("MED_CONF", "confidence", "medium"),
        ("WATCH", "quality", "watch-only"),
        ("ZERO_SIZE", "suggested_contracts", 0),
        ("NO_TRADE", "no_trade_reason", "blocked"),
    ):
        candidate = _approved_candidate(symbol)
        candidate[field] = value
        candidates.append(candidate)
    _write_artifact(artifact, top=candidates)

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.usable is True
    assert gate.approved_count == 0
    for candidate in candidates:
        assert gate.allows(candidate["symbol"], "long_call") == (
            False,
            "scanner_gate_candidate_not_approved",
        )


def test_scanner_filters_direction_before_funnel_emit(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact, top=[_approved_candidate("SPY", "bullish")])
    settings = _settings(tmp_path, artifact, long_call_enabled=True, long_put_enabled=True)
    gate = load_scanner_candidate_gate(settings, now=NOW)
    funnel = FunnelRecorder(tmp_path)
    scanner = Scanner(settings, broker=object(), funnel=funnel)
    call = LongCallSignal(symbol="SPY260821C00780000", expiry="2026-08-21", strike=780)
    put = LongPutSignal(symbol="SPY260821P00770000", expiry="2026-08-21", strike=770)
    monkeypatch.setattr(scanner, "_scan_symbol", lambda symbol, filters, now: [call])
    monkeypatch.setattr(scanner, "_scan_long_put_symbol", lambda symbol, filters, now: [put])

    result = scanner.scan(["SPY"], candidate_gate=gate)

    assert result.candidates == [call]
    persisted = json.loads((tmp_path / "funnel.jsonl").read_text().splitlines()[0])
    assert persisted["extras"]["scanner_candidate_gate_denied_count"] == 1
    assert persisted["extras"]["scanner_candidate_gate_denied_reasons"] == {
        "scanner_gate_direction_mismatch": 1
    }
    assert persisted["dominant_skip_reason"] == "ok"


def test_bpcs_uses_bullish_candidate_semantics_and_reports_funnel_denial(
    tmp_path: Path, monkeypatch
):
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact, top=[_approved_candidate("SPY", "bearish")])
    settings = _settings(tmp_path, artifact, bpcs_enabled=True)
    gate = load_scanner_candidate_gate(settings, now=NOW)
    funnel = FunnelRecorder(tmp_path)
    scanner = Scanner(settings, broker=object(), funnel=funnel)
    spread = BullPutSpreadSignal(
        short_strike=760,
        long_strike=755,
        expiry="2026-09-18",
        credit_estimate=1,
        max_loss_per_contract=400,
        max_gain_per_contract=100,
        pop=0.7,
        conviction=0.8,
    )
    monkeypatch.setattr(scanner, "_scan_bpcs_symbol", lambda symbol, filters, now: [spread])

    result = scanner.scan_bpcs(["SPY"], candidate_gate=gate)

    assert result.candidates == []
    assert result.funnel_row.dominant_skip_reason == "scanner_gate_direction_mismatch"
    persisted = json.loads((tmp_path / "funnel.jsonl").read_text().splitlines()[0])
    assert persisted["extras"]["scanner_candidate_gate_denied_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "unconfirmed",
        "stale_quote",
        "fallback_option",
        "outside_move",
        "missing_delta",
        "missing_iv",
        "stale_daily_source",
        "earnings_blocked",
        "portfolio_blocked",
        "future_quote",
        "failed_intraday_check",
        "incomplete_minute_grid",
        "unknown_earnings_status",
        "portfolio_blocked_by",
        "missing_contract",
        "missing_bid",
        "missing_ask",
        "missing_target",
        "missing_stop",
        "wrong_option_right",
        "wrong_delta_sign",
        "size_mismatch",
        "numeric_strings",
    ],
)
def test_nested_evidence_must_be_confirmed_and_fresh(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "alpha.json"
    candidate = _approved_candidate()
    if mutation == "unconfirmed":
        candidate["intraday"]["confirmed"] = False
    elif mutation == "stale_quote":
        candidate["option"]["quote_asof_utc"] = (NOW - timedelta(minutes=20)).isoformat()
    elif mutation == "fallback_option":
        candidate["option"]["data_quality"] = "yfinance_delayed_option_chain"
    elif mutation == "outside_move":
        candidate["sizing"]["target_inside_expected_move"] = False
    elif mutation == "missing_delta":
        candidate["option"]["delta_estimate"] = None
    elif mutation == "missing_iv":
        candidate["option"]["implied_volatility"] = None
    elif mutation == "stale_daily_source":
        candidate["intraday"]["daily_source_session"] = "2026-08-13"
    elif mutation == "earnings_blocked":
        candidate["earnings"]["blocked"] = True
    elif mutation == "portfolio_blocked":
        candidate["portfolio_guard"]["blocked"] = True
    elif mutation == "future_quote":
        candidate["option"]["quote_asof_utc"] = (NOW + timedelta(minutes=1)).isoformat()
    elif mutation == "failed_intraday_check":
        candidate["intraday"]["checks"]["vwap_aligned"] = False
    elif mutation == "incomplete_minute_grid":
        candidate["intraday"]["minute_coverage"] = 0.95
    elif mutation == "unknown_earnings_status":
        candidate["earnings"] = {"status": "unknown_fail_closed", "blocked": False}
    elif mutation == "portfolio_blocked_by":
        candidate["portfolio_guard"] = {
            "allowed": True,
            "blocked": False,
            "blocked_by": "correlation_cap",
        }
    elif mutation == "missing_contract":
        candidate["option"].pop("contract")
    elif mutation == "missing_bid":
        candidate["option"].pop("bid")
    elif mutation == "missing_ask":
        candidate["option"].pop("ask")
    elif mutation == "missing_target":
        candidate.pop("target")
    elif mutation == "missing_stop":
        candidate.pop("stop")
    elif mutation == "wrong_option_right":
        candidate["option"]["contract"] = "SPY260821P00100000"
    elif mutation == "wrong_delta_sign":
        candidate["option"]["delta_estimate"] = -0.42
    elif mutation == "size_mismatch":
        candidate["sizing"]["suggested_contracts"] = 1
    elif mutation == "numeric_strings":
        for key in ("bid", "ask", "delta_estimate", "implied_volatility", "expected_move"):
            candidate["option"][key] = str(candidate["option"][key])
        candidate["target"] = str(candidate["target"])
        candidate["stop"] = str(candidate["stop"])
    _write_artifact(artifact, top=[candidate])

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is True
    assert gate.approved_count == 0
    assert gate.allows("SPY", "long_call") == (
        False,
        "scanner_gate_candidate_not_approved",
    )


def test_closed_market_artifact_cannot_authorize(tmp_path: Path) -> None:
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact)
    payload = json.loads(artifact.read_text())
    payload["market_state"] = {"is_open": False, "status": "pre_market"}
    artifact.write_text(json.dumps(payload))

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_market_not_open"


def test_unknown_regime_fails_closed_instead_of_bypassing(tmp_path: Path) -> None:
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact, regime="banana")

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_unknown_regime"


def test_source_age_is_measured_from_evaluation_time_not_generation(tmp_path: Path) -> None:
    artifact = tmp_path / "alpha.json"
    candidate = _approved_candidate()
    candidate["intraday"]["bar_asof_utc"] = (NOW - timedelta(minutes=16)).isoformat()
    candidate["option"]["quote_asof_utc"] = (NOW - timedelta(minutes=16)).isoformat()
    _write_artifact(
        artifact,
        generated_at=NOW - timedelta(minutes=14),
        top=[candidate],
    )

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.usable is True
    assert gate.approved_count == 0


@pytest.mark.parametrize("mutation", ["degraded", "stale", "empty", "inconsistent"])
def test_dynamic_universe_must_be_fresh_and_non_degraded(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact)
    payload = json.loads(artifact.read_text())
    if mutation == "degraded":
        payload["universe"]["degraded"] = True
    elif mutation == "stale":
        payload["universe"]["data_asof_utc"] = (NOW - timedelta(minutes=20)).isoformat()
    elif mutation == "empty":
        payload["universe"]["accepted_count"] = 0
    else:
        payload["universe"]["raw_count"] = 1
    artifact.write_text(json.dumps(payload))

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_universe_stale_or_degraded"


@pytest.mark.parametrize(
    "mutation",
    ["cycle_id", "source_fingerprint", "policy_fingerprint", "entry_permission"],
)
def test_artifact_identity_and_regime_permission_are_mandatory(
    tmp_path: Path, mutation: str
) -> None:
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact)
    payload = json.loads(artifact.read_text())
    if mutation == "entry_permission":
        payload["market_regime"]["entry_permission"] = "puts_only"
    else:
        payload.pop(mutation)
    artifact.write_text(json.dumps(payload))

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=NOW)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_artifact_malformed"


def test_artifact_cannot_replay_after_current_regular_session(tmp_path: Path) -> None:
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact, generated_at=NOW + timedelta(hours=5))

    gate = load_scanner_candidate_gate(
        _settings(tmp_path, artifact), now=NOW + timedelta(hours=5, minutes=2)
    )

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_market_not_open"


def test_artifact_cannot_authorize_on_market_holiday(tmp_path: Path) -> None:
    holiday = datetime(2026, 9, 7, 15, 0, tzinfo=UTC)
    candidate = _approved_candidate()
    candidate["signal_id"] = "2026-09-07:SPY:bullish:100.0"
    candidate["intraday"]["bar_asof_utc"] = (holiday - timedelta(minutes=3)).isoformat()
    candidate["intraday"]["daily_source_session"] = "2026-09-04"
    candidate["intraday"]["reference_daily_session"] = "2026-09-04"
    candidate["option"]["quote_asof_utc"] = (holiday - timedelta(minutes=3)).isoformat()
    artifact = tmp_path / "alpha.json"
    _write_artifact(artifact, generated_at=holiday - timedelta(minutes=2), top=[candidate])

    gate = load_scanner_candidate_gate(_settings(tmp_path, artifact), now=holiday)

    assert gate.required is True
    assert gate.usable is False
    assert gate.reason == "scanner_gate_market_not_open"
