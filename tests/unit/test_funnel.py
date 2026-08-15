"""Tests for the FunnelRecorder — one row per scan, no duplicates."""

from __future__ import annotations

from pathlib import Path

from flip_options_bot.signal import FunnelRecorder


def test_emit_writes_one_row(tmp_path: Path):
    recorder = FunnelRecorder(tmp_path)
    row = FunnelRecorder.new_row(watchlist_count=130)
    row.eligible_count = 38
    row.chains_fetched = ["SPY", "QQQ"]
    assert recorder.emit(row) is True

    rows = recorder.all_rows()
    assert len(rows) == 1
    assert rows[0].eligible_count == 38


def test_duplicate_scan_id_returns_false(tmp_path: Path):
    """Structural fix: duplicate scan_id returns False on emit (already-recorded)."""
    recorder = FunnelRecorder(tmp_path)
    row = FunnelRecorder.new_row(watchlist_count=130)
    assert recorder.emit(row) is True
    assert recorder.emit(row) is False  # second emit of same scan_id rejected

    rows = recorder.all_rows()
    assert len(rows) == 1


def test_new_row_has_unique_scan_id():
    r1 = FunnelRecorder.new_row(watchlist_count=130)
    r2 = FunnelRecorder.new_row(watchlist_count=130)
    assert r1.scan_id != r2.scan_id


def test_all_rows_preserves_fields(tmp_path: Path):
    recorder = FunnelRecorder(tmp_path)
    row = FunnelRecorder.new_row(watchlist_count=130, eligible_count=38)
    row.chains_fetched = ["SPY", "QQQ"]
    row.chains_failed = ["IWM"]
    row.move_pass_count = 12
    row.momentum_pass_count = 0
    row.dominant_skip_reason = "momentum_filter"
    row.contract_select_none = {"quote_stale": 5}
    recorder.emit(row)

    rows = recorder.all_rows()
    assert rows[0].chains_fetched == ["SPY", "QQQ"]
    assert rows[0].chains_failed == ["IWM"]
    assert rows[0].dominant_skip_reason == "momentum_filter"
    assert rows[0].contract_select_none == {"quote_stale": 5}


def test_malformed_json_lines_are_skipped(tmp_path: Path):
    """Real-world funnel.jsonl might have malformed lines from a bot crash."""
    p = tmp_path / "funnel.jsonl"
    p.write_text(
        '{"scan_id": "good", "ts": "2026-08-11", "watchlist_count": 1, "eligible_count": 0}\n'
        "this is not valid json\n"
        '{"scan_id": "also-good", "ts": "2026-08-11", "watchlist_count": 2, "eligible_count": 0}\n'
    )

    recorder = FunnelRecorder(tmp_path)
    rows = recorder.all_rows()
    assert len(rows) == 2  # malformed line skipped, not crashed
