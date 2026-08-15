"""Funnel recorder — one row per scan cycle, all stage counters.

Carried forward from flip-alpaca-bot with the structural fix:
- Each funnel row carries a `scan_id` (UUID) so duplicate emits are
  detectable upstream.
- `funnel.jsonl` is APPEND-ONLY but a duplicate `scan_id` is flagged by
  the reconciler in `monitor/funnel_health.py`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class FunnelRow:
    scan_id: str
    ts: str
    watchlist_count: int = 0
    eligible_count: int = 0
    chains_fetched: list[str] = field(default_factory=list)
    chains_failed: list[str] = field(default_factory=list)
    raw_signal_count: int = 0
    move_pass_count: int = 0
    momentum_pass_count: int = 0
    contract_select_pass: int = 0
    contract_select_none: dict[str, int] = field(default_factory=dict)
    conviction_distribution: list[float] = field(default_factory=list)
    sized_count: int = 0
    submitted_count: int = 0
    dominant_skip_reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class FunnelRecorder:
    """Writes one FunnelRow per scan to `run_dir/funnel.jsonl`.

    Usage:
        recorder = FunnelRecorder(run_dir)
        row = FunnelRecorder.new_row(watchlist_count=130)
        row.eligible_count = 38
        row.chains_fetched = [...]
        ...
        recorder.emit(row)
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "funnel.jsonl"

    def emit(self, row: FunnelRow) -> bool:
        """Append the row to funnel.jsonl. Returns True if written, False if
        duplicate scan_id (idempotency check via dedup scan)."""
        existing_scan_ids = self._load_scan_ids()
        if row.scan_id in existing_scan_ids:
            return False
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        return True

    def emit_skip(self, reason: str = "no_candidates") -> bool:
        """Convenience: emit a one-row skip funnel entry without scanning.

        Used by the daemon when the scan loop is short-circuited (e.g.,
        market closed, kill switch active). Keeps the diagnostic
        instrument running even when there's no candidate work.
        """
        row = FunnelRow(
            scan_id=str(uuid.uuid4()),
            ts=datetime.now(UTC).isoformat(),
            watchlist_count=0,
            eligible_count=0,
            chains_fetched=[],
            chains_failed=[],
            raw_signal_count=0,
            move_pass_count=0,
            momentum_pass_count=0,
            contract_select_pass=0,
            contract_select_none={},
            conviction_distribution=[],
            sized_count=0,
            submitted_count=0,
            dominant_skip_reason=reason,
            extras={},
        )
        return self.emit(row)

    def _load_scan_ids(self) -> set[str]:
        ids = set()
        if not self.path.exists():
            return ids
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                ids.add(obj.get("scan_id", ""))
            except json.JSONDecodeError:
                continue
        return ids

    def all_rows(self) -> list[FunnelRow]:
        rows = []
        if not self.path.exists():
            return rows
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # Filter internal fields
                obj.pop("extras", None)
                rows.append(FunnelRow(**obj))
            except (json.JSONDecodeError, TypeError):
                continue
        return rows

    @staticmethod
    def new_row(watchlist_count: int, eligible_count: int = 0) -> FunnelRow:
        return FunnelRow(
            scan_id=str(uuid.uuid4()),
            ts=datetime.now(UTC).isoformat(),
            watchlist_count=watchlist_count,
            eligible_count=eligible_count,
        )
