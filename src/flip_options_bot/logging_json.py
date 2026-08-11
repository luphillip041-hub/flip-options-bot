"""Structured (JSON) logger.

Writes newline-delimited JSON to a file, while keeping the human-readable
text on stdout. Used by the daemon so we can ingest logs into a downstream
analyzer (or grep with `jq`).

Each record:
- ts (ISO 8601 UTC)
- level
- name (logger name)
- message
- ... extras (the kwargs passed at the call site)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

JSON_RECORDER_ATTR = "_json_handler"


class JsonFileHandler(logging.Handler):
    """One JSON object per line. Adds ts/level/name on every emit."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
            with self.path.open("a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            self.handleError(record)


def attach(run_dir: Path, logger_name: str = "flip_options_bot") -> JsonFileHandler:
    """Attach a JSON file handler to the named logger. Returns the handler.

    Idempotent: re-attaching returns the existing handler if one is already
    attached, so the file doesn't get duplicated lines.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "structured.jsonl"
    target = logging.getLogger(logger_name)
    for h in target.handlers:
        if getattr(h, JSON_RECORDER_ATTR, False):
            return h  # type: ignore[return-value]
    handler = JsonFileHandler(log_path)
    setattr(handler, JSON_RECORDER_ATTR, True)
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    return handler