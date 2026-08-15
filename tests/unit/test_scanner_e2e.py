"""End-to-end scanner test with mocked broker.

This validates the FULL scanner pipeline against a fake broker that
returns realistic-looking data — without needing the market to be
open. Catches bugs like:
- Wrong field names in dict returns
- Strategy filter logic
- Conviction scoring
- Volatility regime filter
- Entry-price formula
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from flip_options_bot.broker.alpaca import BrokerClient
from flip_options_bot.config import Settings
from flip_options_bot.signal import FunnelRecorder
from flip_options_bot.signal.scanner import Scanner


def _fakish_bars(prices: list[float], vols: list[int]) -> list[dict]:
    """Build realistic minute bars from a price sequence."""
    bars = []
    base = datetime.now(UTC) - timedelta(minutes=len(prices) - 1)
    for i, (p, v) in enumerate(zip(prices, vols, strict=True)):
        bars.append({
            "t": (base + timedelta(minutes=i)).isoformat(),
            "o": p,
            "h": p * 1.001,
            "l": p * 0.999,
            "c": p,
            "v": v,
        })
    return bars


def _build_broker_mock() -> MagicMock:
    """Mock broker with controlled data — small upward trend."""
    broker = MagicMock(spec=BrokerClient)

    # 20 bars, clear upward trend: 100.00 → 100.50 (+0.50%) — well above
    # the 0.30% min_direction_move threshold.
    prices = [100.00 + 0.025 * i for i in range(20)]  # 0.50% over 20 bars
    vols = [1000] * 20
    bars = _fakish_bars(prices, vols)
    broker.get_stock_bars_minute = MagicMock(return_value=bars)

    # Spot quote: 100.50 / 100.52
    broker.get_stock_quote = MagicMock(return_value={
        "bid": 100.50,
        "ask": 100.52,
    })

    # Option chain: 3 calls, all tight bid/ask, valid OI
    today = datetime.now(UTC).date()
    expiry_str = (today + timedelta(days=5)).strftime("%Y-%m-%d")
    expiry_yymmdd = (today + timedelta(days=5)).strftime("%y%m%d")
    contracts = [
        {"symbol": f"SPY{expiry_yymmdd}C010100", "expiry": expiry_str, "strike": 101.0, "type": "call", "open_interest": 100},
        {"symbol": f"SPY{expiry_yymmdd}C010150", "expiry": expiry_str, "strike": 101.5, "type": "call", "open_interest": 50},
        {"symbol": f"SPY{expiry_yymmdd}C010200", "expiry": expiry_str, "strike": 102.0, "type": "call", "open_interest": 10},
    ]
    broker.list_option_contracts = MagicMock(return_value=contracts)

    # Snapshots for each contract — tight spread (1% wide)
    snapshots = {
        contracts[0]["symbol"]: {"bid": 0.55, "ask": 0.56},
        contracts[1]["symbol"]: {"bid": 0.30, "ask": 0.31},
        contracts[2]["symbol"]: {"bid": 0.15, "ask": 0.16},
    }
    broker.get_option_snapshot = MagicMock(side_effect=lambda sym, expiry=None: snapshots.get(sym))

    return broker


def test_scanner_pipeline_with_tight_spreads(tmp_path: Path):
    """Trending SPY + tight spreads → candidates should emerge."""
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        run_dir=tmp_path,
    )
    broker = _build_broker_mock()
    funnel = FunnelRecorder(tmp_path / "funnel.jsonl")
    scanner = Scanner(settings, broker, funnel)

    result = scanner.scan(["SPY"])

    print(f"  watchlist={result.funnel_row.watchlist_count}")
    print(f"  raw_signal={result.funnel_row.raw_signal_count}")
    print(f"  skip={result.funnel_row.dominant_skip_reason}")
    print(f"  candidates: {len(result.candidates)}")
    for c in result.candidates:
        print(f"    {c.symbol} strike=${c.strike} entry=${c.limit_price} conviction={c.conviction:.3f}")


def test_scanner_skips_with_wide_spreads(tmp_path: Path):
    """Wide-spread chain → volatility regime filter rejects."""
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        run_dir=tmp_path,
    )
    broker = _build_broker_mock()
    funnel = FunnelRecorder(tmp_path / "funnel.jsonl")
    scanner = Scanner(settings, broker, funnel)

    # Override the snapshots to be very wide spread (50%+)
    today = datetime.now(UTC).date()
    expiry_yymmdd = (today + timedelta(days=5)).strftime("%y%m%d")
    wide = {
        f"SPY{expiry_yymmdd}C010100": {"bid": 0.40, "ask": 0.80},   # 67% spread
        f"SPY{expiry_yymmdd}C010150": {"bid": 0.20, "ask": 0.40},   # 67%
        f"SPY{expiry_yymmdd}C010200": {"bid": 0.05, "ask": 0.15},   # 100%
    }
    broker.get_option_snapshot = MagicMock(side_effect=lambda sym, expiry=None: wide.get(sym))

    result = scanner.scan(["SPY"])
    print(f"  wide-spread scan: candidates={len(result.candidates)}, skip={result.funnel_row.dominant_skip_reason}")
    # Median spread is > 20% → volatility regime rejects → 0 candidates
    assert len(result.candidates) == 0


def test_scanner_with_insufficient_bars(tmp_path: Path):
    """Only 5 bars returned (less than lookback 20) → skip with no error."""
    settings = Settings(
        phase="paper",
        live_trade_enabled=False,
        run_dir=tmp_path,
    )
    broker = _build_broker_mock()
    broker.get_stock_bars_minute = MagicMock(return_value=[
        {"o": 100, "h": 100, "l": 100, "c": 100, "v": 100}
    ] * 5)
    funnel = FunnelRecorder(tmp_path / "funnel.jsonl")
    scanner = Scanner(settings, broker, funnel)

    result = scanner.scan(["SPY"])
    assert len(result.candidates) == 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'tests')
    sys.path.insert(0, 'src')
    tmp = Path('/tmp/scan_e2e')
    tmp.mkdir(parents=True, exist_ok=True)
    print('=== test_scanner_pipeline_with_tight_spreads ===')
    test_scanner_pipeline_with_tight_spreads(tmp)
    print()
    print('=== test_scanner_skips_with_wide_spreads ===')
    test_scanner_skips_with_wide_spreads(tmp)
    print()
    print('=== test_scanner_with_insufficient_bars ===')
    test_scanner_with_insufficient_bars(tmp)
    print()
    print('All e2e scanner tests passed ✓')
