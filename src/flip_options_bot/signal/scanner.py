"""Scanner — watchlist → candidates → FunnelRow emit.

For each eligible symbol:
1. Pull minute bars (lookback N minutes)
2. Compute direction move, momentum, vwap extension
3. Pull option contracts in expiry window
4. For each contract, compute conviction
5. Filter to top by conviction above floor
6. Emit ONE FunnelRow per scan cycle (even if zero passes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from ..broker import BrokerClient
from ..config import Settings
from ..signal import FunnelRecorder, FunnelRow
from ..strategies.bull_put_credit import (
    BPCSFilters,
    BullPutSpreadSignal,
    compute_bpcs_conviction,
    estimate_credit,
    make_filters_from_settings as make_bpcs_filters,
    passes_bpcs_conviction,
    pick_target_expiry as pick_bpcs_expiry,
    select_strikes,
)
from ..strategies.long_call import (
    LongCallFilters,
    LongCallSignal,
    compute_conviction,
    make_filters_from_settings,
    passes_conviction,
    pick_target_expiry,
    volatility_regime_ok,
)
from ..strategies.long_equity import (
    LongEquityFilters,
    LongEquitySignal,
    compute_conviction as compute_equity_conviction,
    make_filters_from_settings as make_equity_filters,
    passes_conviction as passes_equity_conviction,
)

log = logging.getLogger("flip_options_bot.scanner")


@dataclass
class ScanResult:
    funnel_row: FunnelRow
    candidates: list[LongCallSignal | LongEquitySignal]


@dataclass
class BPCSScanResult:
    funnel_row: FunnelRow
    candidates: list[BullPutSpreadSignal]


class Scanner:
    """One scan cycle. Stateless. No broker state mutation."""

    def __init__(self, settings: Settings, broker: BrokerClient, funnel: FunnelRecorder):
        self.settings = settings
        self.broker = broker
        self.funnel = funnel

    def scan(self, watchlist: list[str], target_dte: int | None = None) -> ScanResult:
        """Run one scan over the watchlist. Emits a FunnelRow."""
        target_dte = target_dte or self.settings.target_dte
        filters = make_filters_from_settings(self.settings)
        row = FunnelRecorder.new_row(watchlist_count=len(watchlist))

        candidates: list[LongCallSignal | LongEquitySignal] = []
        now = datetime.now(timezone.utc)

        for symbol in watchlist:
            try:
                signals: list[LongCallSignal | LongEquitySignal] = list(self._scan_symbol(symbol, filters, now))
                if self.settings.long_equity_enabled and not signals:
                    signals.extend(self._scan_long_equity_symbol(symbol, make_equity_filters(self.settings), now))
            except Exception as e:
                log.warning("scan failed for %s: %s", symbol, e)
                row.chains_failed.append(symbol)
                continue
            row.chains_fetched.append(symbol)
            for s in signals:
                candidates.append(s)
                row.raw_signal_count += 1

        row.sized_count = len(candidates)
        row.submitted_count = 0  # decided by executor, not scanner

        # Conviction distribution for diagnostics
        row.conviction_distribution = [round(c.conviction, 3) for c in candidates]

        if not candidates:
            row.dominant_skip_reason = "no_candidates"
        else:
            row.dominant_skip_reason = "ok"

        # Emit always (even with zero candidates — that's the funnel signal)
        emitted = self.funnel.emit(row)
        if not emitted:
            log.warning("duplicate scan_id %s not re-emitted", row.scan_id)

        return ScanResult(funnel_row=row, candidates=candidates)

    def scan_bpcs(self, watchlist: list[str]) -> BPCSScanResult:
        """BPCS scan over the watchlist. Emits BullPutSpreadSignals.

        Bull put credit spreads profit on NEUTRAL or DOWN moves. We scan
        even when direction_move is negative — that's the whole point.
        Returns 0 candidates if direction is strongly bearish or no
        credit spreads meet conviction threshold.
        """
        filters = make_bpcs_filters(self.settings)
        row = FunnelRecorder.new_row(watchlist_count=len(watchlist))
        candidates: list[BullPutSpreadSignal] = []
        now = datetime.now(timezone.utc)

        for symbol in watchlist:
            try:
                signals = self._scan_bpcs_symbol(symbol, filters, now)
            except Exception as e:
                log.warning("bpcs scan failed for %s: %s", symbol, e)
                row.chains_failed.append(symbol)
                continue
            row.chains_fetched.append(symbol)
            for s in signals:
                candidates.append(s)
                row.raw_signal_count += 1

        row.sized_count = len(candidates)
        row.submitted_count = 0
        row.conviction_distribution = [round(c.conviction, 3) for c in candidates]
        if not candidates:
            row.dominant_skip_reason = "no_candidates"
        else:
            row.dominant_skip_reason = "ok"

        emitted = self.funnel.emit(row)
        if not emitted:
            log.warning("duplicate bpcs scan_id %s not re-emitted", row.scan_id)
        return BPCSScanResult(funnel_row=row, candidates=candidates)

    def _scan_bpcs_symbol(
        self, symbol: str, filters: BPCSFilters, now: datetime
    ) -> list[BullPutSpreadSignal]:
        """Find bull put credit spread candidates on a single symbol."""
        signals: list[BullPutSpreadSignal] = []

        # Pull minute bars to check direction. BPCS wants neutral or
        # slightly bearish — strongly bullish means we should sell
        # upside (not downside) so we skip.
        bars = self.broker.get_stock_bars_minute(symbol, lookback_minutes=60)
        if len(bars) < 30:
            return signals
        closes = [b["c"] for b in bars[-30:]]
        direction_move = (closes[-1] - closes[0]) / closes[0]
        spot = closes[-1]

        # BPCS wants bullish OR neutral. Strongly down → skip.
        # (Conviction check below will also block if too bearish.)
        if direction_move < -0.005:  # -0.5% in 30 min = too bearish
            return signals

        # Pick expiry closest to target_dte
        expiry_gte = (now + timedelta(days=filters.min_dte)).strftime("%Y-%m-%d")
        expiry_lte = (now + timedelta(days=filters.max_dte)).strftime("%Y-%m-%d")
        contracts = self.broker.list_option_contracts(symbol, expiry_gte, expiry_lte, option_type="put")
        if not contracts:
            return signals
        expiry_set = sorted({c["expiry"] for c in contracts})
        expiry = pick_bpcs_expiry(expiry_set, filters)
        if expiry is None:
            return signals
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - now.date()).days

        # Pick strikes by delta proxy. Pass available_strikes so we can snap
        # to the closest available strike when Alpaca paper's chain is
        # incomplete near the spot price.
        all_strikes = sorted({c["strike"] for c in contracts if c["type"] == "put"})
        strike_pair = select_strikes(spot, available_strikes=all_strikes)
        if strike_pair is None:
            return signals
        short_strike, long_strike = strike_pair
        spread_width = short_strike - long_strike
        if spread_width < filters.min_width or spread_width > filters.max_width:
            return signals

        # Find the matching put contracts (need short put + long put same expiry)
        eligible_puts = [
            c for c in contracts
            if c["expiry"] == expiry
            and c["type"] == "put"
            and c["open_interest"] > 0
            and c["strike"] in {short_strike, long_strike}
        ]
        if len(eligible_puts) < 2:
            return signals

        # Pull snapshots for both legs
        short_sym = next((c["symbol"] for c in eligible_puts if c["strike"] == short_strike), None)
        long_sym = next((c["symbol"] for c in eligible_puts if c["strike"] == long_strike), None)
        if not short_sym or not long_sym:
            return signals

        short_snap = self.broker.get_option_snapshot(short_sym, expiry=expiry)
        long_snap = self.broker.get_option_snapshot(long_sym, expiry=expiry)
        if not short_snap or not long_snap:
            return signals
        if not short_snap.get("bid") or not short_snap.get("ask"):
            return signals
        if not long_snap.get("bid") or not long_snap.get("ask"):
            return signals

        # Compute credit
        credit = estimate_credit(
            short_put_bid=float(short_snap["bid"]),
            short_put_ask=float(short_snap["ask"]),
            long_put_bid=float(long_snap["bid"]),
            long_put_ask=float(long_snap["ask"]),
        )
        if credit <= 0:
            return signals

        # Filter: credit must be at least N% of spread width
        # 20% is a sensible floor — anything lower means you're selling
        # too little premium relative to your max loss exposure.
        if credit / spread_width < filters.min_credit_pct_of_width:
            return signals

        # Vol regime: BPCS wants rich vol. Use short-put IV width proxy.
        spread_pct = (float(short_snap["ask"]) - float(short_snap["bid"])) / max((float(short_snap["bid"]) + float(short_snap["ask"])) / 2, 0.01)
        # For BPCS we tolerate wider spreads (we're a seller, not buyer)
        if spread_pct > 0.40:  # 40% cap — stricter than long_call
            return signals

        # IV proxy from short put (rough — assumes ATM-ish short put)
        iv_rank_proxy = 0.30  # placeholder; real calc would use ATM straddle
        # Use spread_pct as a weak IV proxy: wider spread ≈ richer vol
        if spread_pct > 0.15:
            iv_rank_proxy = 0.40
        elif spread_pct < 0.05:
            iv_rank_proxy = 0.10

        conviction = compute_bpcs_conviction(
            spot=spot,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            iv_rank_proxy=iv_rank_proxy,
            direction_move_pct=direction_move,
            filters=filters,
        )
        if not passes_bpcs_conviction(conviction, filters):
            return signals

        max_loss_per_contract = spread_width * 100 - credit * 100
        # Use mid + 25% spread above mid for short put (to get a fill),
        # and mid + 25% spread above mid for long put (to get a fill).
        # These are intentionally aggressive (a tiny bit over mid) so
        # the spread fills within a few minutes in paper.
        short_put_price = round(float(short_snap["bid"]) + 0.50 * (float(short_snap["ask"]) - float(short_snap["bid"])), 2)
        long_put_price = round(float(long_snap["ask"]) + 0.10 * (float(long_snap["ask"]) - float(long_snap["bid"])), 2)
        signals.append(BullPutSpreadSignal(
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            credit_estimate=credit,
            max_loss_per_contract=max_loss_per_contract,
            max_gain_per_contract=credit * 100,
            pop=0.70,  # rough — short put at 30 delta has ~70% POP
            conviction=conviction,
            short_strike_price_estimate=short_put_price,
            long_strike_price_estimate=long_put_price,
            short_put_symbol=short_sym,
            long_put_symbol=long_sym,
            strategy_id="bull_put_credit_spread",
            ts=now.isoformat(),
        ))
        return signals

    # ===== Internals =====

    def _scan_long_equity_symbol(
        self, symbol: str, filters: LongEquityFilters, now: datetime
    ) -> list[LongEquitySignal]:
        """Find bullish share candidates when calls are not the right vehicle."""
        signals: list[LongEquitySignal] = []
        bars = self.broker.get_stock_bars_minute(
            symbol, lookback_minutes=filters.directional_lookback_minutes + 30
        )
        if len(bars) < filters.directional_lookback_minutes:
            return signals
        direction_move, vwap_extension, short_momentum = self._compute_features(bars, filters)
        if direction_move < filters.min_direction_move_pct:
            return signals
        if vwap_extension > filters.max_vwap_extension_pct:
            return signals

        quote = self.broker.get_stock_quote(symbol)
        if not quote or not quote.get("bid") or not quote.get("ask"):
            return signals
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        if bid <= 0 or ask <= 0 or ask < bid:
            return signals
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / max(mid, 0.01)
        if spread_pct > 0.0025:  # shares should be tight; 25 bps is already wide
            return signals

        conviction = compute_equity_conviction(
            direction_move=direction_move,
            vwap_extension=vwap_extension,
            short_momentum=short_momentum,
            filters=filters,
        )
        if not passes_equity_conviction(conviction, filters):
            return signals

        entry_price = round(mid + 0.25 * spread, 2)
        qty = int(filters.max_position_dollar / max(entry_price, 0.01))
        if qty <= 0:
            return signals
        signals.append(LongEquitySignal(
            symbol=symbol,
            qty=qty,
            limit_price=entry_price,
            conviction=conviction,
            stop_price=round(entry_price * (1.0 - filters.stop_loss_pct), 2),
            take_profit_price=round(entry_price * (1.0 + filters.take_profit_pct), 2),
            ts=now.isoformat(),
        ))
        return signals

    def _scan_symbol(
        self, symbol: str, filters: LongCallFilters, now: datetime
    ) -> list[LongCallSignal]:
        """Scan a single symbol. Returns the candidate LongCallSignals."""
        signals: list[LongCallSignal] = []

        # Pull minute bars for the lookback window
        bars = self.broker.get_stock_bars_minute(
            symbol, lookback_minutes=filters.directional_lookback_minutes + 30
        )
        if len(bars) < filters.directional_lookback_minutes:
            # Insufficient bars — structural lesson from flip-alpaca-bot's
            # funnel collapse at the momentum_filter stage.
            return signals

        # Compute features
        direction_move, vwap_extension, short_momentum = self._compute_features(
            bars, filters
        )

        if direction_move < filters.min_direction_move_pct:
            return signals  # Fails the direction filter
        if vwap_extension > filters.max_vwap_extension_pct:
            return signals  # Fails the extension filter

        # Pull option contracts
        expiry_gte = (now + timedelta(days=filters.min_dte)).strftime("%Y-%m-%d")
        expiry_lte = (now + timedelta(days=filters.max_dte)).strftime("%Y-%m-%d")
        contracts = self.broker.list_option_contracts(symbol, expiry_gte, expiry_lte)
        if not contracts:
            return signals

        # Pick expiry closest to target_dte
        expiry_set = sorted({c["expiry"] for c in contracts})
        expiry = pick_target_expiry(expiry_set, filters)
        if expiry is None:
            return signals

        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - now.date()).days

        # Filter contracts to the chosen expiry, calls
        eligible = [
            c for c in contracts
            if c["expiry"] == expiry and c["type"] == "call"
            and c["open_interest"] > 0
        ]
        if not eligible:
            return signals

        # Volatility regime filter — sample spreads across the eligible
        # chain. If median spread > 20%, market makers are pulling quotes
        # (vol crash). Don't buy premium in that regime.
        chain_spreads = []
        for c in eligible[:20]:  # sample up to 20 contracts for the regime check
            snap = self.broker.get_option_snapshot(c["symbol"], expiry=expiry)
            if snap and snap.get("bid", 0) > 0 and snap.get("ask", 0) > 0:
                mid = (snap["bid"] + snap["ask"]) / 2
                spread_pct = (snap["ask"] - snap["bid"]) / max(mid, 0.01)
                chain_spreads.append(spread_pct)
        if not volatility_regime_ok(chain_spreads):
            log.info(
                "%s chain skipped: vol regime wide (median spread=%.2f%%)",
                symbol,
                sorted(chain_spreads)[len(chain_spreads) // 2] * 100 if chain_spreads else 0,
            )
            return signals

        # Pull snapshots for each contract
        spot_quote = self.broker.get_stock_quote(symbol)
        if spot_quote is None:
            return signals
        spot = (spot_quote["bid"] + spot_quote["ask"]) / 2

        for c in eligible:
            snap = self.broker.get_option_snapshot(c["symbol"], expiry=expiry)
            if snap is None or "bid" not in snap or "ask" not in snap:
                continue
            bid = float(snap["bid"])
            ask = float(snap["ask"])
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue  # invalid quote; skip
            mid = (bid + ask) / 2
            spread = ask - bid
            spread_pct = spread / max(mid, 0.01)

            # Skip contracts with absurdly wide spreads (>50%) — never trade
            # a wide-spread option, the slippage alone will eat our gains.
            if spread_pct > 0.50:
                continue

            conviction = compute_conviction(
                direction_move=direction_move,
                vwap_extension=vwap_extension,
                short_momentum=short_momentum,
                spread_pct=spread_pct,
                filters=filters,
            )
            if not passes_conviction(conviction, filters):
                continue

            # Gain-protection entry: pay 25% of the spread above mid, so we
            # sit closer to the bid (the bid IS our floor if we have to exit
            # immediately). For a $1.00 mid / $0.20 spread, that's $1.05.
            # This beats paying mid*1.02 ($1.02) AND pays less than asking
            # full ask ($1.10). The trade fills when the bid lifts to our
            # limit; in paper, it usually fills within seconds.
            entry_price = round(mid + 0.25 * spread, 2)

            signals.append(LongCallSignal(
                symbol=c["symbol"],
                expiry=expiry,
                strike=c["strike"],
                side="buy",
                option_type="call",
                qty=1,
                limit_price=entry_price,
                conviction=conviction,
                dte=dte,
                strategy_id="long_call",
                ts=now.isoformat(),
            ))

        # Keep top N by conviction
        signals.sort(key=lambda s: s.conviction, reverse=True)
        return signals[:3]

    def _compute_features(
        self, bars: list[dict], filters
    ) -> tuple[float, float, float]:
        """Compute direction move, vwap extension, short momentum from minute bars."""
        if len(bars) < filters.directional_lookback_minutes:
            return 0.0, 0.0, 0.0
        # Use the most recent lookback window
        recent = bars[-filters.directional_lookback_minutes:]
        closes = [b["c"] for b in recent]
        volumes = [b["v"] for b in recent]

        first_close = closes[0]
        last_close = closes[-1]
        direction_move = (last_close - first_close) / first_close

        # VWAP over the window
        typical_prices = [(b["h"] + b["l"] + b["c"]) / 3 for b in recent]
        vwap = sum(tp * v for tp, v in zip(typical_prices, volumes)) / max(sum(volumes), 1)
        vwap_extension = abs(last_close - vwap) / vwap

        # Short momentum: last 5 minutes vs first 5 minutes of the window
        if len(closes) >= 10:
            short_momentum = (closes[-5:][-1] - closes[:5][0]) / closes[:5][0]
        else:
            short_momentum = direction_move

        return direction_move, vwap_extension, short_momentum