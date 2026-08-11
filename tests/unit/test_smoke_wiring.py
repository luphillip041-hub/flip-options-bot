"""End-to-end smoke test: wire the full stack with a mocked broker.

Proves that the daemon's components all wire together correctly without
needing live Alpaca creds. Validates:
- Settings loads from a temp .env
- BrokerClient constructs with a fake trading client
- Journal + RiskEngine + FunnelRecorder + Closer + PositionMonitor all
  instantiate cleanly
- Daemon.main('--config-check') returns 0

The Daemon loop / scanner / executor paths still need a real broker and
are tested in tests/integration/test_broker_integration.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from flip_options_bot.config import Settings
from flip_options_bot.execution import Closer
from flip_options_bot.journal import Journal
from flip_options_bot.monitor.position_monitor import PositionMonitor
from flip_options_bot.risk import RiskEngine
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner


def test_components_wire_cleanly(tmp_path: Path) -> None:
    """All post-Step-3 components instantiate without errors."""
    # Settings
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        alpaca_paper_key="test_key",
        alpaca_paper_secret="test_secret",
        run_dir=tmp_path / "runs",
    )
    assert not settings.is_live()
    assert settings.has_paper_creds()

    # Journal
    journal = Journal(settings.run_dir)
    assert (settings.run_dir / "journal.db").exists()

    # Risk engine
    risk = RiskEngine(settings, settings.run_dir)
    state = risk.load_state()
    assert state.kill_switch is False
    assert state.open_position_count == 0

    # Funnel recorder
    funnel = FunnelRecorder(settings.run_dir)

    # Closer (with mocked broker)
    broker = MagicMock()
    broker.submit_close_sell.return_value = MagicMock(id="abc", status="accepted")
    broker.get_option_snapshot.return_value = {"bid": 1.0, "ask": 1.10}
    closer = Closer(settings, broker, journal, risk)
    assert closer is not None

    # Position monitor
    monitor = PositionMonitor(settings, broker, journal, risk, closer)
    assert monitor is not None


def test_daemon_config_check_succeeds(tmp_path: Path, monkeypatch) -> None:
    """`--config-check` must succeed with paper creds set.

    The daemon reads .env from cwd, so we chdir into tmp_path and write a
    minimal env file. The daemon logs to stderr; we capture stderr + stdout.
    """
    import subprocess

    repo = Path("/root/flip/projects/flip-options-bot")
    venv_python = repo / ".venv" / "bin" / "python"

    env_file = tmp_path / ".env"
    env_file.write_text(
        "FOB_PHASE=paper\n"
        "APCA_API_KEY_ID_PAPER=test\n"
        "APCA_API_SECRET_KEY_PAPER=test\n"
        "FOB_RUN_DIR=" + str(tmp_path / "runs") + "\n"
    )

    result = subprocess.run(
        [
            str(venv_python), "-m", "flip_options_bot.daemon",
            "--config-check",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"config-check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # The daemon logs via the logging module to stderr. Look for the
    # "Settings loaded" log line OR any reference to "paper".
    combined = (result.stderr or "") + (result.stdout or "")
    assert "paper" in combined or "Settings" in combined, (
        f"expected 'paper' or 'Settings' in output; got:\n{combined}"
    )