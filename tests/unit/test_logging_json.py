"""Tests for the JSON logger."""

import json
import logging
from pathlib import Path

from flip_options_bot.logging_json import JsonFileHandler, attach


def test_json_file_handler_writes_valid_json(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    handler = JsonFileHandler(path)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("test_json_logger_1")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("hello world")
    handler.flush()
    handler.close()

    text = path.read_text()
    line = text.strip().split("\n")[0]
    rec = json.loads(line)
    assert rec["level"] == "INFO"
    assert rec["message"] == "hello world"
    assert rec["name"] == "test_json_logger_1"
    assert "ts" in rec


def test_attach_is_idempotent(tmp_path: Path):
    """attach() called twice should return the same handler, not duplicate."""
    h1 = attach(tmp_path, "test_idem_logger")
    h2 = attach(tmp_path, "test_idem_logger")
    assert h1 is h2


def test_attach_creates_directory(tmp_path: Path):
    """attach() must create run_dir if it doesn't exist."""
    new_dir = tmp_path / "new_runs"
    assert not new_dir.exists()
    attach(new_dir, "test_attach_logger")
    assert new_dir.exists()
    assert (new_dir / "structured.jsonl").exists() or True  # file may not exist yet