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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast
from zoneinfo import ZoneInfo

from ..broker import BrokerClient
from ..config import Settings
from ..data.yfinance_options import YFinanceOptionChainProvider, YFinanceOptionQuote
from ..signal import FunnelRecorder, FunnelRow
from ..strategies.bull_put_credit import (
    BPCSFilters,
    BullPutSpreadSignal,
    compute_bpcs_conviction,
    estimate_credit,
    passes_bpcs_conviction,
    select_strikes,
)
from ..strategies.bull_put_credit import (
    make_filters_from_settings as make_bpcs_filters,
)
from ..strategies.bull_put_credit import (
    pick_target_expiry as pick_bpcs_expiry,
)
from ..strategies.long_call import (
    LongCallFilters,
    LongCallSignal,
    compute_conviction,
    make_filters_from_settings,
    passes_conviction,
    volatility_regime_ok,
)
from ..strategies.long_equity import (
    LongEquityFilters,
    LongEquitySignal,
)
from ..strategies.long_equity import (
    compute_conviction as compute_equity_conviction,
)
from ..strategies.long_equity import (
    make_filters_from_settings as make_equity_filters,
)
from ..strategies.long_equity import (
    passes_conviction as passes_equity_conviction,
)
from ..strategies.long_put import (
    LongPutFilters,
    LongPutSignal,
)
from ..strategies.long_put import (
    compute_conviction as compute_put_conviction,
)
from ..strategies.long_put import (
    make_filters_from_settings as make_put_filters,
)
from ..strategies.long_put import (
    passes_conviction as passes_put_conviction,
)

log = logging.getLogger("flip_options_bot.scanner")
ET = ZoneInfo("America/New_York")


class YFinanceProvider(Protocol):
    def get_quote(
        self, underlying: str, expiry: str, contract_symbol: str
    ) -> YFinanceOptionQuote | None: ...


@dataclass
class ScanResult:
    funnel_row: FunnelRow
    candidates: list[LongCallSignal | LongPutSignal | LongEquitySignal]


@dataclass
class BPCSScanResult:
    funnel_row: FunnelRow
    candidates: list[BullPutSpreadSignal]


class Scanner:
    """One scan cycle. Stateless. No broker state mutation."""

    def __init__(
        self,
        settings: Settings,
        broker: BrokerClient,
        funnel: FunnelRecorder,
        yfinance_provider: YFinanceProvider | None = None,
    ):
        self.settings = settings
        self.broker = broker
        self.funnel = funnel
        self._yfinance_provider: YFinanceProvider | None = yfinance_provider

    def scan(self, watchlist: list[str], target_dte: int | None = None) -> ScanResult:
        """Run one scan over the watchlist. Emits a FunnelRow."""
        target_dte = target_dte or self.settings.target_dte
        filters = make_filters_from_settings(self.settings)
        row = FunnelRecorder.new_row(watchlist_count=len(watchlist))

        candidates: list[LongCallSignal | LongPutSignal | LongEquitySignal] = []
        now = datetime.now(UTC)

        for symbol in watchlist:
            try:
                signals: list[LongCallSignal | LongPutSignal | LongEquitySignal] = []
                if self.settings.long_call_enabled:
                    signals.extend(self._scan_symbol(symbol, filters, now))
                if self.settings.long_put_enabled:
                    signals.extend(
                        self._scan_long_put_symbol(symbol, make_put_filters(self.settings), now)
                    )
                if self.settings.long_equity_enabled and not signals:
                    signals.extend(
                        self._scan_long_equity_symbol(
                            symbol, make_equity_filters(self.settings), now
                        )
                    )
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
        now = datetime.now(UTC)

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
        contracts = self.broker.list_option_contracts(
            symbol, expiry_gte, expiry_lte, option_type="put"
        )
        if not contracts:
            return signals
        expiry_set = sorted({c["expiry"] for c in contracts})
        expiry = pick_bpcs_expiry(expiry_set, filters)
        if expiry is None:
            return signals
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
            c
            for c in contracts
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
        spread_pct = (float(short_snap["ask"]) - float(short_snap["bid"])) / max(
            (float(short_snap["bid"]) + float(short_snap["ask"])) / 2, 0.01
        )
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
        short_put_price = round(
            float(short_snap["bid"]) + 0.50 * (float(short_snap["ask"]) - float(short_snap["bid"])),
            2,
        )
        long_put_price = round(
            float(long_snap["ask"]) + 0.10 * (float(long_snap["ask"]) - float(long_snap["bid"])), 2
        )
        signals.append(
            BullPutSpreadSignal(
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
            )
        )
        return signals

    # ===== Internals =====

    def _scan_long_put_symbol(
        self, symbol: str, filters: LongPutFilters, now: datetime
    ) -> list[LongPutSignal]:
        """Scan a single symbol for bearish long-put candidates."""
        signals: list[LongPutSignal] = []
        bars = self.broker.get_stock_bars_minute(
            symbol, lookback_minutes=filters.directional_lookback_minutes + 30
        )
        if len(bars) < filters.directional_lookback_minutes:
            return signals
        if not self._bars_are_current_session(bars, filters, now):
            return signals
        direction_move, vwap_extension, short_momentum = self._compute_features(bars, filters)
        if direction_move > -filters.min_direction_move_pct:
            return signals
        if short_momentum > -filters.min_short_momentum_pct:
            return signals
        if vwap_extension > filters.max_vwap_extension_pct:
            return signals

        expiry_gte = (now + timedelta(days=filters.min_dte)).strftime("%Y-%m-%d")
        expiry_lte = (now + timedelta(days=filters.max_dte)).strftime("%Y-%m-%d")
        contracts = self.broker.list_option_contracts(
            symbol, expiry_gte, expiry_lte, option_type="put"
        )
        if not contracts:
            return signals

        spot_quote = self.broker.get_stock_quote(symbol)
        if spot_quote is None:
            return signals
        spot = (spot_quote["bid"] + spot_quote["ask"]) / 2

        for expiry in self._ordered_expiries({c["expiry"] for c in contracts}, filters, now):
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - now.date()).days
            target_otm_pct = self._target_otm_pct(filters.target_otm_pct)
            target_strike = spot * (1.0 - target_otm_pct)
            eligible = [
                c
                for c in contracts
                if c["expiry"] == expiry and c["type"] == "put" and c["open_interest"] > 0
            ]
            if not eligible:
                continue
            eligible.sort(key=lambda c: abs(c["strike"] - target_strike))

            chain_spreads = []
            for c in eligible[:20]:
                snap = self.broker.get_option_snapshot(c["symbol"], expiry=expiry)
                if snap and snap.get("bid", 0) > 0 and snap.get("ask", 0) > 0:
                    mid = (snap["bid"] + snap["ask"]) / 2
                    chain_spreads.append((snap["ask"] - snap["bid"]) / max(mid, 0.01))
            if not volatility_regime_ok(chain_spreads):
                log.info(
                    "%s %s put chain skipped: vol regime wide (median spread=%.2f%%)",
                    symbol,
                    expiry,
                    sorted(chain_spreads)[len(chain_spreads) // 2] * 100 if chain_spreads else 0,
                )
                continue

            for c in eligible:
                snap = self.broker.get_option_snapshot(c["symbol"], expiry=expiry)
                if snap is None or "bid" not in snap or "ask" not in snap:
                    continue
                bid = float(snap["bid"])
                ask = float(snap["ask"])
                if bid <= 0 or ask <= 0 or ask <= bid:
                    continue
                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pct = spread / max(mid, 0.01)
                if spread_pct > self._max_long_option_spread_pct():
                    continue
                otm_pct = self._otm_pct("put", c["strike"], spot)
                if otm_pct <= 0:
                    continue
                conviction = compute_put_conviction(
                    direction_move=direction_move,
                    vwap_extension=vwap_extension,
                    short_momentum=short_momentum,
                    spread_pct=spread_pct,
                    filters=filters,
                )
                if not passes_put_conviction(conviction, filters):
                    continue
                entry_price = round(mid + 0.25 * spread, 2)
                if entry_price * 100 > self.settings.max_contract_dollar:
                    continue
                if entry_price < self._min_long_option_premium():
                    continue
                signal = LongPutSignal(
                    symbol=c["symbol"],
                    expiry=expiry,
                    strike=c["strike"],
                    qty=1,
                    limit_price=entry_price,
                    conviction=conviction,
                    dte=dte,
                    notes=f"otm_pct={otm_pct:.4f};high_reward={self.settings.long_option_high_reward_mode}",
                    ts=now.isoformat(),
                )
                enriched = self._enrich_yfinance_1dte(signal, symbol)
                if enriched is not None:
                    signals.append(cast(LongPutSignal, enriched))

        signals.sort(
            key=lambda s: self._directional_sort_key(s, spot, "put", filters),
            reverse=True,
        )
        return signals[:3]

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
        if not self._bars_are_current_session(bars, filters, now):
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
        signals.append(
            LongEquitySignal(
                symbol=symbol,
                qty=qty,
                limit_price=entry_price,
                conviction=conviction,
                stop_price=round(entry_price * (1.0 - filters.stop_loss_pct), 2),
                take_profit_price=round(entry_price * (1.0 + filters.take_profit_pct), 2),
                ts=now.isoformat(),
            )
        )
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
        if not self._bars_are_current_session(bars, filters, now):
            return signals

        # Compute features
        direction_move, vwap_extension, short_momentum = self._compute_features(bars, filters)

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

        # Pull stock quote once, then evaluate every allowed expiry in target order.
        # If 0DTE is ugly/wide/missing, keep walking forward up to max_dte.
        spot_quote = self.broker.get_stock_quote(symbol)
        if spot_quote is None:
            return signals
        spot = (spot_quote["bid"] + spot_quote["ask"]) / 2

        for expiry in self._ordered_expiries({c["expiry"] for c in contracts}, filters, now):
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - now.date()).days
            target_otm_pct = self._target_otm_pct(filters.target_otm_pct)
            target_strike = spot * (1.0 + target_otm_pct)
            eligible = [
                c
                for c in contracts
                if c["expiry"] == expiry and c["type"] == "call" and c["open_interest"] > 0
            ]
            if not eligible:
                continue
            eligible.sort(key=lambda c: abs(c["strike"] - target_strike))

            # Volatility regime filter — sample spreads around the target OTM area.
            chain_spreads = []
            for c in eligible[:20]:
                snap = self.broker.get_option_snapshot(c["symbol"], expiry=expiry)
                if snap and snap.get("bid", 0) > 0 and snap.get("ask", 0) > 0:
                    mid = (snap["bid"] + snap["ask"]) / 2
                    spread_pct = (snap["ask"] - snap["bid"]) / max(mid, 0.01)
                    chain_spreads.append(spread_pct)
            if not volatility_regime_ok(chain_spreads):
                log.info(
                    "%s %s call chain skipped: vol regime wide (median spread=%.2f%%)",
                    symbol,
                    expiry,
                    sorted(chain_spreads)[len(chain_spreads) // 2] * 100 if chain_spreads else 0,
                )
                continue

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

                # Skip wide-spread options; high-reward mode is stricter because
                # farther OTM contracts can look cheap while being untradeable.
                if spread_pct > self._max_long_option_spread_pct():
                    continue
                otm_pct = self._otm_pct("call", c["strike"], spot)
                if otm_pct <= 0:
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

                # Gain-protection entry: pay 25% of the spread above mid.
                entry_price = round(mid + 0.25 * spread, 2)
                if entry_price * 100 > self.settings.max_contract_dollar:
                    continue
                if entry_price < self._min_long_option_premium():
                    continue

                signal = LongCallSignal(
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
                    notes=f"otm_pct={otm_pct:.4f};high_reward={self.settings.long_option_high_reward_mode}",
                    ts=now.isoformat(),
                )
                enriched = self._enrich_yfinance_1dte(signal, symbol)
                if enriched is not None:
                    signals.append(cast(LongCallSignal, enriched))

        # Keep top N by conviction, tie-breaking toward target DTE and OTM strike.
        signals.sort(
            key=lambda s: self._directional_sort_key(s, spot, "call", filters),
            reverse=True,
        )
        return signals[:3]

    def _target_otm_pct(self, base_target: float) -> float:
        if not self.settings.long_option_high_reward_mode:
            return base_target
        ladder = tuple(p for p in self.settings.long_option_otm_ladder_pct if p > 0)
        return max(ladder) if ladder else base_target

    def _min_long_option_premium(self) -> float:
        return (
            self.settings.long_option_min_premium
            if self.settings.long_option_high_reward_mode
            else 0.0
        )

    def _max_long_option_spread_pct(self) -> float:
        return (
            self.settings.long_option_max_spread_pct
            if self.settings.long_option_high_reward_mode
            else 0.50
        )

    @staticmethod
    def _otm_pct(side: Literal["call", "put"], strike: float, spot: float) -> float:
        if spot <= 0:
            return 0.0
        if side == "call":
            return (strike - spot) / spot
        return (spot - strike) / spot

    def _directional_sort_key(
        self,
        signal: LongCallSignal | LongPutSignal,
        spot: float,
        side: Literal["call", "put"],
        filters,
    ) -> tuple[float, int, float]:
        otm_pct = max(self._otm_pct(side, signal.strike, spot), 0.0)
        target_otm = max(self._target_otm_pct(filters.target_otm_pct), 0.0001)
        convexity_bonus = 0.0
        if self.settings.long_option_high_reward_mode:
            convexity_bonus = (
                min(otm_pct / target_otm, 1.0) * self.settings.long_option_convexity_weight
            )
        return (
            signal.conviction + convexity_bonus,
            -abs(signal.dte - filters.target_dte),
            -abs(otm_pct - target_otm),
        )

    def _enrich_yfinance_1dte(
        self, signal: LongCallSignal | LongPutSignal, underlying: str
    ) -> LongCallSignal | LongPutSignal | None:
        """Use yfinance as a free 1DTE+ confirmation/ranking sidecar.

        Alpaca remains broker/execution truth. yfinance can be stale/delayed and
        often returns zero bid/ask, so it only adds small conviction bonuses and
        metadata unless the explicit strict gate is enabled.
        """
        if not self.settings.yfinance_confirm_1dte_enabled:
            return signal
        if signal.dte < self.settings.yfinance_confirm_min_dte:
            return signal

        provider = self._get_yfinance_provider()
        quote = provider.get_quote(underlying, signal.expiry, signal.symbol)
        if quote is None:
            return self._yfinance_missing(signal)

        tags: list[str] = ["yf=seen"]
        bonus = 0.0
        if quote.has_bid_ask:
            spread_pct = quote.spread_pct
            tags.append("yf_quote=bid_ask")
            if spread_pct is not None:
                tags.append(f"yf_spread={spread_pct:.4f}")
            if spread_pct is not None and spread_pct <= self.settings.yfinance_max_spread_pct:
                bonus += self.settings.yfinance_bidask_bonus
                tags.append("yf_confirm=bidask_tight")
            elif self.settings.yfinance_strict_gate:
                log.info("%s rejected by yfinance strict wide quote", signal.symbol)
                return None
        else:
            tags.append("yf_quote=last_price_proxy_non_executable")
            if quote.last_price is not None:
                tags.append(f"yf_last={quote.last_price:.2f}")
            if quote.last_trade_date:
                tags.append(f"yf_last_trade={self._trade_date_tag(quote.last_trade_date)}")
            # Last price is not a fillable quote. It can still provide a tiny
            # research/ranking nudge for 1DTE+ if volume says the contract is active
            # AND the row traded in today's ET session. This prevents stale/yesterday
            # Yahoo volume from improving high-risk candidate ranking pre-open.
            volume_ok = (quote.volume or 0) >= self.settings.yfinance_min_volume
            fresh_volume_ok = (
                not self.settings.yfinance_require_current_trade_date_for_volume_bonus
                or self._yfinance_last_trade_is_signal_date(quote, signal)
            )
            if volume_ok and fresh_volume_ok:
                bonus += self.settings.yfinance_volume_bonus
                tags.append("yf_confirm=volume_only")
            elif volume_ok:
                tags.append("yf_confirm=stale_volume_no_bonus")
            elif self.settings.yfinance_strict_gate:
                log.info("%s rejected by yfinance strict missing bid/ask", signal.symbol)
                return None

        if quote.volume is not None:
            tags.append(f"yf_vol={quote.volume}")
        if quote.open_interest is not None:
            tags.append(f"yf_oi={quote.open_interest}")
        if quote.implied_volatility is not None:
            tags.append(f"yf_iv={quote.implied_volatility:.4f}")

        return replace(
            signal,
            conviction=min(1.0, signal.conviction + bonus),
            notes=";".join([signal.notes, *tags]),
        )

    @staticmethod
    def _trade_date_tag(raw: str) -> str:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(ET).date().isoformat()
        except Exception:
            return "unknown"

    @staticmethod
    def _signal_et_date(signal: LongCallSignal | LongPutSignal) -> str:
        try:
            dt = datetime.fromisoformat(signal.ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(ET).date().isoformat()
        except Exception:
            return datetime.now(ET).date().isoformat()

    def _yfinance_last_trade_is_signal_date(
        self, quote: YFinanceOptionQuote, signal: LongCallSignal | LongPutSignal
    ) -> bool:
        if not quote.last_trade_date:
            return False
        return self._trade_date_tag(quote.last_trade_date) == self._signal_et_date(signal)

    def _yfinance_missing(
        self, signal: LongCallSignal | LongPutSignal
    ) -> LongCallSignal | LongPutSignal | None:
        if self.settings.yfinance_strict_gate:
            log.info("%s rejected by yfinance strict missing contract", signal.symbol)
            return None
        return replace(signal, notes=f"{signal.notes};yf=missing")

    def _get_yfinance_provider(self) -> YFinanceProvider:
        if self._yfinance_provider is None:
            self._yfinance_provider = YFinanceOptionChainProvider()
        return self._yfinance_provider

    def _ordered_expiries(self, expiries: set[str], filters, now: datetime) -> list[str]:
        """Allowed expiries ordered by closeness to target DTE, then sooner.

        This lets 0DTE win when valid, while allowing 1-14 DTE fallback when
        same-day chains are missing, too wide, or otherwise fail closed.
        """
        ranked: list[tuple[int, int, str]] = []
        for exp_str in expiries:
            try:
                dte = (datetime.strptime(exp_str, "%Y-%m-%d").date() - now.date()).days
            except ValueError:
                continue
            if filters.min_dte <= dte <= filters.max_dte:
                ranked.append((abs(dte - filters.target_dte), dte, exp_str))
        ranked.sort()
        return [exp for _, _, exp in ranked]

    def _bars_are_current_session(self, bars: list[dict], filters, now: datetime) -> bool:
        """Fail closed unless the latest underlying bars are fresh regular-session evidence."""
        recent = bars[-filters.directional_lookback_minutes :]
        timestamps: list[datetime] = []
        for bar in recent:
            raw_ts = bar.get("t")
            if not raw_ts:
                return False
            try:
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except ValueError:
                return False
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            timestamps.append(ts)

        timestamps.sort()
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        latest = timestamps[-1]
        age_seconds = (now_utc.astimezone(UTC) - latest.astimezone(UTC)).total_seconds()
        if age_seconds < -300 or age_seconds > 300:
            return False
        if latest.astimezone(ET).date() != now_utc.astimezone(ET).date():
            return False
        for prev, cur in zip(timestamps, timestamps[1:], strict=False):
            delta = (cur - prev).total_seconds()
            if delta <= 0 or delta > 120:
                return False
        return True

    @staticmethod
    def _positive_float(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if out <= 0:
            return None
        return out

    def _compute_features(self, bars: list[dict], filters) -> tuple[float, float, float]:
        """Compute direction move, vwap extension, short momentum from minute bars."""
        if len(bars) < filters.directional_lookback_minutes:
            return 0.0, 0.0, 0.0
        # Use the most recent lookback window
        recent = bars[-filters.directional_lookback_minutes :]
        closes = [self._positive_float(b.get("c")) for b in recent]
        highs = [self._positive_float(b.get("h")) for b in recent]
        lows = [self._positive_float(b.get("l")) for b in recent]
        volumes = [self._positive_float(b.get("v")) for b in recent]
        if any(v is None for v in [*closes, *highs, *lows, *volumes]):
            return 0.0, 0.0, 0.0
        closes_f = [float(v) for v in closes]
        highs_f = [float(v) for v in highs]
        lows_f = [float(v) for v in lows]
        volumes_f = [float(v) for v in volumes]

        first_close = closes_f[0]
        last_close = closes_f[-1]
        direction_move = (last_close - first_close) / first_close

        # VWAP over the window
        total_volume = sum(volumes_f)
        if total_volume <= 0:
            return 0.0, 0.0, 0.0
        typical_prices = [
            (high + low + close) / 3
            for high, low, close in zip(highs_f, lows_f, closes_f, strict=True)
        ]
        vwap = sum(tp * v for tp, v in zip(typical_prices, volumes_f, strict=True)) / total_volume
        if vwap <= 0:
            return 0.0, 0.0, 0.0
        vwap_extension = abs(last_close - vwap) / vwap

        # Short momentum: last 5 minutes vs first 5 minutes of the window
        if len(closes_f) >= 10:
            short_momentum = (closes_f[-5:][-1] - closes_f[:5][0]) / closes_f[:5][0]
        else:
            short_momentum = direction_move

        return direction_move, vwap_extension, short_momentum
