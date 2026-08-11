"""Tests for the paper-observation harness."""

from __future__ import annotations

from pathlib import Path

from flip_options_bot.journal import Journal, TradeEvent
from flip_options_bot.observation import ObservationHarness


def _record_win(journal: Journal, symbol: str, pnl: float) -> None:
    pos_id = Journal.new_position_id()
    journal.append(TradeEvent(
        event_id=f"open-{pos_id}",
        ts=Journal.now_iso(),
        kind="open",
        symbol=symbol,
        side="buy",
        qty=1,
        price=2.50,
        position_id=pos_id,
    ))
    journal.append(TradeEvent(
        event_id=f"close-{pos_id}",
        ts=Journal.now_iso(),
        kind="close",
        symbol=symbol,
        side="sell",
        qty=1,
        price=2.50 + pnl,
        position_id=pos_id,
        realized_pnl=pnl,
    ))


def test_record_market_day_appends(tmp_path: Path) -> None:
    h = ObservationHarness(tmp_path)
    assert h.record_market_day("2026-08-12") is True
    assert h.record_market_day("2026-08-13") is True
    assert h.record_market_day("2026-08-12") is False  # idempotent
    assert len(h.state["market_days"]) == 2


def test_promotion_gate_blocks_before_min_days(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    _record_win(journal, "SPY260815C00770000", pnl=10.0)
    h = ObservationHarness(tmp_path, min_market_days=10)
    gate = h.promotion_gate(tmp_path / "runs" / "journal.db")
    assert gate.eligible is False
    assert "paper market days" in gate.reason
    assert gate.market_days_recorded == 0


def test_promotion_gate_blocks_negative_pnl(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    _record_win(journal, "SPY260815C00770000", pnl=-50.0)
    h = ObservationHarness(tmp_path, min_market_days=1)
    # Record enough market days to pass the day count
    h.record_market_day("2026-08-12")
    gate = h.promotion_gate(tmp_path / "runs" / "journal.db")
    assert gate.eligible is False
    assert "negative" in gate.reason or "P&L" in gate.reason


def test_promotion_gate_passes_all_gates(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "runs")
    # 5 wins, 1 loss = 83% WR, net positive
    for i, pnl in enumerate([10.0, 20.0, 5.0, 15.0, 8.0, -3.0]):
        _record_win(journal, f"SPY260815C0077000{i}", pnl=pnl)
    h = ObservationHarness(tmp_path, min_market_days=1, min_win_rate=0.5)
    h.record_market_day("2026-08-12")
    gate = h.promotion_gate(tmp_path / "runs" / "journal.db")
    assert gate.eligible is True, gate.reason
    assert gate.win_rate == 5.0 / 6.0


def test_render_digest_includes_key_fields(tmp_path: Path) -> None:
    h = ObservationHarness(tmp_path)
    h.record_market_day("2026-08-12")
    journal = Journal(tmp_path / "runs")
    _record_win(journal, "SPY260815C00770000", pnl=10.0)
    text = h.render_digest(tmp_path / "runs" / "journal.db")
    assert "Days: 1/10" in text
    assert "Win rate:" in text
    assert "Net realized P&L:" in text


def test_state_persists_across_instances(tmp_path: Path) -> None:
    h1 = ObservationHarness(tmp_path)
    h1.record_market_day("2026-08-12")
    h1.record_market_day("2026-08-13")

    h2 = ObservationHarness(tmp_path)  # fresh instance, same dir
    assert len(h2.state["market_days"]) == 2
    assert h2.state["market_days"] == ["2026-08-12", "2026-08-13"]