#!/usr/bin/env python3
"""Free yfinance option-chain snapshot for flip-options-bot.

This is a forward-data helper, not a historical backtest source. yfinance can
usually provide current/delayed chains with bid/ask/OI/volume/IV, but it does
not provide historical OPRA quote streams. Use it to sanity-check today's
contracts and collect free chain snapshots while paper trading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from flip_options_bot.config import get_settings

ET = ZoneInfo("America/New_York")


@dataclass
class ChainCandidate:
    snapshot_ts: str
    underlying: str
    side: str
    expiry: str
    dte: int
    contract_symbol: str
    strike: float
    spot: float
    otm_pct: float
    bid: float
    ask: float
    mid: float
    entry_limit: float
    spread_pct: float
    last_price: float | None
    volume: int | None
    open_interest: int | None
    implied_volatility: float | None
    score: float
    quote_mode: str
    source: str = "yfinance_current_chain"


def _float_or_none(value) -> float | None:
    try:
        if value is None:
            return None
        if value != value:  # NaN
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value) -> int | None:
    try:
        if value is None or value != value:
            return None
        return int(value)
    except Exception:
        return None


def _spot(ticker) -> float | None:
    for attr in ("fast_info", "info"):
        try:
            data = getattr(ticker, attr)
            if callable(data):
                data = data()
            for key in ("last_price", "lastPrice", "regularMarketPrice", "previousClose"):
                val = data.get(key) if hasattr(data, "get") else None
                parsed = _float_or_none(val)
                if parsed and parsed > 0:
                    return parsed
        except Exception:
            continue
    try:
        hist = ticker.history(period="5d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None
    return None


def _score(row: ChainCandidate) -> float:
    oi = row.open_interest or 0
    vol = row.volume or 0
    liq = min(1.0, oi / 500.0) * 0.45 + min(1.0, vol / 100.0) * 0.25
    tight = max(0.0, 1.0 - row.spread_pct / 0.35) * 0.20
    convex = min(1.0, row.otm_pct / 0.015) * 0.10
    return round((liq + tight + convex) * 100, 2)


def _candidate_from_row(
    *,
    snapshot_ts: str,
    underlying: str,
    side: str,
    expiry: str,
    spot: float,
    row,
    settings,
    allow_last_price_proxy: bool,
) -> ChainCandidate | None:
    strike = _float_or_none(row.get("strike"))
    bid = _float_or_none(row.get("bid"))
    ask = _float_or_none(row.get("ask"))
    last_price = _float_or_none(row.get("lastPrice"))
    if strike is None:
        return None
    quote_mode = "bid_ask"
    if bid is not None and ask is not None and bid > 0 and ask > bid:
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / max(mid, 0.01)
        max_spread = (
            settings.long_option_max_spread_pct if settings.long_option_high_reward_mode else 0.50
        )
        if spread_pct > max_spread:
            return None
        entry_limit = round(mid + 0.25 * spread, 2)
    elif allow_last_price_proxy and last_price is not None and last_price > 0:
        # Yahoo often zeros bid/ask/OI off-hours. Keep these rows for free
        # chain/volume research, but tag them as NON-executable proxies.
        bid = 0.0
        ask = 0.0
        mid = last_price
        spread_pct = 999.0
        entry_limit = round(last_price, 2)
        quote_mode = "last_price_proxy_non_executable"
    else:
        return None
    if side == "call":
        otm_pct = (strike - spot) / spot
    else:
        otm_pct = (spot - strike) / spot
    if otm_pct <= 0:
        return None
    if entry_limit < settings.long_option_min_premium:
        return None
    if entry_limit * 100 > settings.max_contract_dollar:
        return None
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    today_et = datetime.now(ET).date()
    dte = (exp_date - today_et).days
    if dte < settings.min_dte or dte > settings.max_dte:
        return None
    c = ChainCandidate(
        snapshot_ts=snapshot_ts,
        underlying=underlying,
        side=side,
        expiry=expiry,
        dte=dte,
        contract_symbol=str(row.get("contractSymbol", "")),
        strike=strike,
        spot=spot,
        otm_pct=otm_pct,
        bid=bid,
        ask=ask,
        mid=round(mid, 4),
        entry_limit=entry_limit,
        spread_pct=round(spread_pct, 6),
        last_price=last_price,
        volume=_int_or_none(row.get("volume")),
        open_interest=_int_or_none(row.get("openInterest")),
        implied_volatility=_float_or_none(row.get("impliedVolatility")),
        score=0.0,
        quote_mode=quote_mode,
    )
    c.score = _score(c)
    return c


def collect(
    symbols: list[str],
    max_expiries: int,
    per_symbol_side: int,
    allow_last_price_proxy: bool,
) -> tuple[list[ChainCandidate], dict]:
    import yfinance as yf

    load_dotenv("/root/.config/flip-options-bot/.env", override=True)
    settings = get_settings()
    snapshot_ts = datetime.now(UTC).isoformat()
    candidates: list[ChainCandidate] = []
    coverage: dict[str, dict] = {}
    for symbol in symbols:
        t = yf.Ticker(symbol)
        try:
            expiries = list(t.options or [])
        except Exception as exc:
            coverage[symbol] = {"error": f"options_fetch_failed:{exc}"}
            continue
        spot = _spot(t)
        coverage[symbol] = {
            "expiries_seen": len(expiries),
            "spot": spot,
            "rows": 0,
            "candidates": 0,
        }
        if not spot:
            coverage[symbol]["error"] = "missing_spot"
            continue
        valid_expiries = []
        today_et = datetime.now(ET).date()
        for expiry in expiries:
            try:
                dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today_et).days
            except ValueError:
                continue
            if settings.min_dte <= dte <= settings.max_dte:
                valid_expiries.append(expiry)
        valid_expiries = valid_expiries[:max_expiries]
        side_buckets: dict[str, list[ChainCandidate]] = {"call": [], "put": []}
        for expiry in valid_expiries:
            try:
                chain = t.option_chain(expiry)
            except Exception:
                continue
            for side, df in (("call", chain.calls), ("put", chain.puts)):
                coverage[symbol]["rows"] += int(len(df))
                for _, row in df.iterrows():
                    candidate = _candidate_from_row(
                        snapshot_ts=snapshot_ts,
                        underlying=symbol,
                        side=side,
                        expiry=expiry,
                        spot=spot,
                        row=row,
                        settings=settings,
                        allow_last_price_proxy=allow_last_price_proxy,
                    )
                    if candidate:
                        side_buckets[side].append(candidate)
        for side in ("call", "put"):
            target = (
                max(settings.long_option_otm_ladder_pct)
                if settings.long_option_high_reward_mode
                else (
                    settings.long_call_target_otm_pct
                    if side == "call"
                    else settings.long_put_target_otm_pct
                )
            )
            side_buckets[side].sort(
                key=lambda c: (
                    c.score,
                    -abs(c.otm_pct - target),
                    -abs(c.dte - settings.target_dte),
                ),
                reverse=True,
            )
            candidates.extend(side_buckets[side][:per_symbol_side])
        coverage[symbol]["candidates"] = sum(1 for c in candidates if c.underlying == symbol)
    candidates.sort(key=lambda c: c.score, reverse=True)
    manifest = {
        "snapshot_ts": snapshot_ts,
        "source": "yfinance current/delayed option chains",
        "symbols": symbols,
        "coverage": coverage,
        "limitations": [
            "Current/delayed chain snapshot only; yfinance is not a historical OPRA quote source.",
            "Use for free forward chain capture, liquidity sanity checks, and provider adapter testing — not final execution backtesting.",
            "Rows tagged last_price_proxy_non_executable have no usable bid/ask and must not be treated as fillable quotes.",
        ],
    }
    return candidates, manifest


def main() -> int:
    load_dotenv("/root/.config/flip-options-bot/.env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=os.environ.get("FOB_WATCHLIST", "SPY,QQQ,IWM,DIA"))
    parser.add_argument("--max-expiries", type=int, default=4)
    parser.add_argument("--per-symbol-side", type=int, default=3)
    parser.add_argument(
        "--no-last-price-proxy",
        action="store_true",
        help="Require nonzero bid/ask; by default yfinance lastPrice is retained as a non-executable research proxy when bid/ask are zero.",
    )
    parser.add_argument("--out-dir", default="runs/yfinance_chain_snapshots")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    candidates, manifest = collect(
        symbols,
        args.max_expiries,
        args.per_symbol_side,
        allow_last_price_proxy=not args.no_last_price_proxy,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "candidates.csv"
    json_path = out_dir / "candidates.json"
    manifest_path = out_dir / "manifest.json"
    with csv_path.open("w", newline="") as f:
        fieldnames = list(ChainCandidate.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))
    json_path.write_text(json.dumps([asdict(c) for c in candidates], indent=2, sort_keys=True))
    manifest["artifacts"] = {
        "csv": str(csv_path),
        "json": str(json_path),
        "manifest": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "top": [asdict(c) for c in candidates[:20]],
                "coverage": manifest["coverage"],
                "artifacts": manifest["artifacts"],
                "limitations": manifest["limitations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
