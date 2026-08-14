#!/usr/bin/env python3
"""Underlying-only replay for flip-options-bot directional long option signals.

This is NOT an option-price backtest. Alpaca historical option bars require OPRA
agreement on this account, so this replay validates signal volume/quality on
underlying minute bars and reports forward directional returns.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from flip_options_bot.config import get_settings
from flip_options_bot.market_time import is_entry_window
from flip_options_bot.strategies.long_call import compute_conviction, make_filters_from_settings
from flip_options_bot.strategies.long_put import (
    compute_conviction as compute_put_conviction,
    make_filters_from_settings as make_put_filters,
)

ET = ZoneInfo("America/New_York")


@dataclass
class SignalReplay:
    decision_ts: str
    symbol: str
    side: str
    conviction: float
    entry_price: float
    exit_ts_60m: str | None
    exit_price_60m: float | None
    dir_ret_15m: float | None
    dir_ret_30m: float | None
    dir_ret_60m: float | None
    dir_ret_120m: float | None
    dir_ret_eod: float | None
    direction_move: float
    vwap_extension: float
    short_momentum: float


def _auth_header() -> str:
    key = os.environ.get("APCA_API_KEY_ID_PAPER", "")
    secret = os.environ.get("APCA_API_SECRET_KEY_PAPER", "")
    if not key or not secret:
        raise SystemExit("missing APCA paper credentials")
    auth = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return f"Authorization: Basic {auth}"


def _curl_json(url: str) -> dict:
    r = subprocess.run(
        ["curl", "-sS", "-H", _auth_header(), url],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed rc={r.returncode}: {r.stderr[:200]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bad json from alpaca: {r.stdout[:300]}") from exc


def fetch_stock_bars(symbols: list[str], start: datetime, end: datetime, data_base: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    token = None
    pages = 0
    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "adjustment": "raw",
            "feed": "iex",
            "limit": "10000",
        }
        if token:
            params["page_token"] = token
        from urllib.parse import urlencode

        url = f"{data_base}/v2/stocks/bars?{urlencode(params)}"
        data = _curl_json(url)
        pages += 1
        for sym, bars in (data.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        next_token = data.get("next_page_token")
        if not next_token or next_token == token:
            break
        token = next_token
        if pages > 100:
            raise RuntimeError("pagination guard tripped")
    for sym in out:
        out[sym].sort(key=lambda b: b["t"])
    return out


def _to_bar(raw: dict) -> dict:
    return {
        "t": raw["t"],
        "o": float(raw["o"]),
        "h": float(raw["h"]),
        "l": float(raw["l"]),
        "c": float(raw["c"]),
        "v": int(raw.get("v") or 0),
    }


def _features(window: list[dict]) -> tuple[float, float, float]:
    closes = [b["c"] for b in window]
    vols = [b["v"] for b in window]
    direction_move = (closes[-1] - closes[0]) / closes[0]
    typical = [(b["h"] + b["l"] + b["c"]) / 3 for b in window]
    vwap = sum(tp * v for tp, v in zip(typical, vols)) / max(sum(vols), 1)
    vwap_extension = abs(closes[-1] - vwap) / vwap
    short_momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 10 else direction_move
    return direction_move, vwap_extension, short_momentum


def _dir_ret(side: str, entry: float, future: float | None) -> float | None:
    if future is None or entry <= 0:
        return None
    if side == "call":
        return future / entry - 1.0
    return entry / future - 1.0


def summarize(rows: list[SignalReplay], start: str, end: str, symbols: list[str], manifest: dict) -> dict:
    def stat(vals: list[float | None]) -> dict:
        clean = [float(v) for v in vals if v is not None and math.isfinite(v)]
        if not clean:
            return {"n": 0}
        wins = [v for v in clean if v > 0]
        return {
            "n": len(clean),
            "win_rate": len(wins) / len(clean),
            "avg": sum(clean) / len(clean),
            "median": sorted(clean)[len(clean) // 2],
            "sum": sum(clean),
        }

    by_side = {}
    for side in ("call", "put"):
        subset = [r for r in rows if r.side == side]
        by_side[side] = {
            "trades": len(subset),
            "ret_15m": stat([r.dir_ret_15m for r in subset]),
            "ret_30m": stat([r.dir_ret_30m for r in subset]),
            "ret_60m": stat([r.dir_ret_60m for r in subset]),
            "ret_120m": stat([r.dir_ret_120m for r in subset]),
            "ret_eod": stat([r.dir_ret_eod for r in subset]),
        }
    by_symbol = {}
    for sym in sorted(set(r.symbol for r in rows)):
        subset = [r for r in rows if r.symbol == sym]
        by_symbol[sym] = {
            "trades": len(subset),
            "ret_60m": stat([r.dir_ret_60m for r in subset]),
            "ret_eod": stat([r.dir_ret_eod for r in subset]),
        }
    return {
        "label": "underlying_directional_replay_not_option_pnl",
        "start": start,
        "end": end,
        "symbols": symbols,
        "symbols_count": len(symbols),
        "trades": len(rows),
        "overall": {
            "ret_15m": stat([r.dir_ret_15m for r in rows]),
            "ret_30m": stat([r.dir_ret_30m for r in rows]),
            "ret_60m": stat([r.dir_ret_60m for r in rows]),
            "ret_120m": stat([r.dir_ret_120m for r in rows]),
            "ret_eod": stat([r.dir_ret_eod for r in rows]),
        },
        "by_side": by_side,
        "by_symbol": by_symbol,
        "manifest": manifest,
        "limitations": [
            "Historical option bars returned 403 OPRA agreement is not signed; this run uses underlying IEX minute bars only.",
            "No historical bid/ask, Greeks, IV, option fills, theta, gamma, or spread slippage are modeled.",
            "A positive directional replay is forward-test plumbing evidence, not live-money or quote-executable edge proof.",
        ],
    }


def run(args: argparse.Namespace) -> dict:
    load_dotenv("/root/.config/flip-options-bot/.env", override=True)
    settings = get_settings()
    symbols = [s.strip().upper() for s in (args.symbols or os.environ.get("FOB_WATCHLIST", "SPY,QQQ,IWM,DIA")).split(",") if s.strip()]
    start = datetime.fromisoformat(args.start).replace(tzinfo=ET).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=ET).astimezone(timezone.utc)
    bars_by_symbol = fetch_stock_bars(symbols, start, end, settings.alpaca_data_base)

    call_filters = make_filters_from_settings(settings)
    put_filters = make_put_filters(settings)
    lookback = max(call_filters.directional_lookback_minutes, put_filters.directional_lookback_minutes)
    candidates_by_ts: dict[str, list[dict]] = defaultdict(list)
    bar_index: dict[tuple[str, str], int] = {}
    bars_norm: dict[str, list[dict]] = {}

    for sym, raws in bars_by_symbol.items():
        bars = [_to_bar(b) for b in raws]
        bars_norm[sym] = bars
        for i, b in enumerate(bars):
            bar_index[(sym, b["t"])] = i
        for i in range(lookback, len(bars)):
            b = bars[i]
            ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            if not is_entry_window(ts):
                continue
            window = bars[i - lookback + 1 : i + 1]
            direction_move, vwap_extension, short_momentum = _features(window)
            # Historical option spread is unavailable; use tight-spread assumption
            # so this tests underlying signal quality, not option chain coverage.
            spread_pct = 0.05
            if settings.long_call_enabled and direction_move >= call_filters.min_direction_move_pct and short_momentum >= call_filters.min_short_momentum_pct:
                conv = compute_conviction(direction_move, vwap_extension, short_momentum, spread_pct, call_filters)
                if conv >= call_filters.min_conviction:
                    candidates_by_ts[b["t"]].append({
                        "symbol": sym,
                        "side": "call",
                        "conviction": conv,
                        "bar_idx": i,
                        "entry_price": b["c"],
                        "features": (direction_move, vwap_extension, short_momentum),
                    })
            if settings.long_put_enabled and direction_move <= -put_filters.min_direction_move_pct and short_momentum <= -put_filters.min_short_momentum_pct:
                conv = compute_put_conviction(direction_move, vwap_extension, short_momentum, spread_pct, put_filters)
                if conv >= put_filters.min_conviction:
                    candidates_by_ts[b["t"]].append({
                        "symbol": sym,
                        "side": "put",
                        "conviction": conv,
                        "bar_idx": i,
                        "entry_price": b["c"],
                        "features": (direction_move, vwap_extension, short_momentum),
                    })

    active_until: dict[str, datetime] = {}
    rows: list[SignalReplay] = []
    for ts_str in sorted(candidates_by_ts):
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        candidates = sorted(candidates_by_ts[ts_str], key=lambda c: c["conviction"], reverse=True)
        open_count = sum(1 for until in active_until.values() if until > ts)
        submitted = 0
        for c in candidates:
            if submitted >= settings.max_submissions_per_scan:
                break
            if open_count >= settings.max_positions:
                break
            sym = c["symbol"]
            if active_until.get(sym, datetime.min.replace(tzinfo=timezone.utc)) > ts:
                continue
            bars = bars_norm[sym]
            i = c["bar_idx"]
            entry_i = i + 1
            if entry_i >= len(bars):
                continue
            entry = bars[entry_i]["c"]
            day_et = datetime.fromisoformat(bars[entry_i]["t"].replace("Z", "+00:00")).astimezone(ET).date()
            same_day_indices = [
                j for j in range(entry_i, len(bars))
                if datetime.fromisoformat(bars[j]["t"].replace("Z", "+00:00")).astimezone(ET).date() == day_et
            ]
            if not same_day_indices:
                continue
            eod_i = same_day_indices[-1]

            def close_after(minutes: int) -> tuple[str | None, float | None, int | None]:
                target = ts + timedelta(minutes=minutes)
                for j in range(entry_i, min(eod_i + 1, len(bars))):
                    jt = datetime.fromisoformat(bars[j]["t"].replace("Z", "+00:00"))
                    if jt >= target:
                        return bars[j]["t"], bars[j]["c"], j
                return bars[eod_i]["t"], bars[eod_i]["c"], eod_i

            ts15, px15, _ = close_after(15)
            ts30, px30, _ = close_after(30)
            ts60, px60, i60 = close_after(args.hold_minutes)
            ts120, px120, _ = close_after(120)
            px_eod = bars[eod_i]["c"]
            direction_move, vwap_extension, short_momentum = c["features"]
            rows.append(SignalReplay(
                decision_ts=ts_str,
                symbol=sym,
                side=c["side"],
                conviction=round(c["conviction"], 6),
                entry_price=entry,
                exit_ts_60m=ts60,
                exit_price_60m=px60,
                dir_ret_15m=_dir_ret(c["side"], entry, px15),
                dir_ret_30m=_dir_ret(c["side"], entry, px30),
                dir_ret_60m=_dir_ret(c["side"], entry, px60),
                dir_ret_120m=_dir_ret(c["side"], entry, px120),
                dir_ret_eod=_dir_ret(c["side"], entry, px_eod),
                direction_move=direction_move,
                vwap_extension=vwap_extension,
                short_momentum=short_momentum,
            ))
            until = datetime.fromisoformat((ts60 or bars[eod_i]["t"]).replace("Z", "+00:00"))
            active_until[sym] = until
            open_count += 1
            submitted += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "underlying_replay_trades.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(SignalReplay.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    manifest = {
        "commit": subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Alpaca stock bars v2 feed=iex",
        "hold_minutes": args.hold_minutes,
        "scan_granularity": "1Min bars; live daemon scans every 30s, replay is bar-limited",
        "settings": {
            "min_dte": settings.min_dte,
            "target_dte": settings.target_dte,
            "max_dte": settings.max_dte,
            "max_positions": settings.max_positions,
            "max_submissions_per_scan": settings.max_submissions_per_scan,
            "long_option_high_reward_mode": settings.long_option_high_reward_mode,
            "long_option_otm_ladder_pct": list(settings.long_option_otm_ladder_pct),
            "long_call_min_conviction": settings.long_call_min_conviction,
            "long_put_min_conviction": settings.long_put_min_conviction,
        },
        "raw_bar_counts": {sym: len(v) for sym, v in bars_by_symbol.items()},
    }
    summary = summarize(rows, args.start, args.end, symbols, manifest)
    summary["artifacts"] = {"trades_csv": str(rows_path), "summary_json": str(out_dir / "summary.json")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="ET date/time, e.g. 2026-07-13T09:30:00")
    parser.add_argument("--end", required=True, help="ET date/time, e.g. 2026-08-13T16:00:00")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--hold-minutes", type=int, default=60)
    parser.add_argument("--out-dir", default="runs/backtests/directional_replay_latest")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
