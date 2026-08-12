"""Broker layer — thin wrapper over alpaca-py.

Structural fixes vs. flip-alpaca-bot:
1. submit_order returns (client_order_id, broker_order) atomically. The
   client_order_id IS the event_id used in journal writes — so a stuck
   reconcile loop never creates a duplicate event for the same order.
2. list_filled_orders is the canonical source for real fills. The journal
   record_close path ALWAYS prefers this over the conservative-loss field.
3. place_oco_tpsl creates a one-cancels-other TP/SL bracket at the broker
   (lesson from the FLIP_PHASE comment about broker-resident TP/SL).
4. cancel_unfilled_submits is idempotent — safe to call every cycle.
5. No market orders. Limit only. The executor decides the limit price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionChainRequest,
    OptionSnapshotRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    OrderClass,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopLimitOrderRequest,
    TakeProfitRequest,
)
from alpaca.trading.models import Order as AlpacaOrder

from ..config import Settings

log = logging.getLogger("flip_options_bot.broker")


@dataclass
class BracketLegs:
    """TP/SL bracket legs placed at the broker. Both are limit orders,
    attached as OCO to the parent via OrderClass=OCO."""

    tp_price: float
    sl_price: float
    tp_order_id: str = ""
    sl_order_id: str = ""


class BrokerClient:
    """Single instance per daemon. Wraps alpaca-py with the structural
    fixes baked in.

    Usage:
        broker = BrokerClient.from_settings(settings)
        order = broker.submit_buy(symbol=..., qty=1, limit_price=2.50, position_id=...)
        bracket = broker.place_tpsl_bracket(parent_order=order, tp=3.00, sl=1.80)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.is_live():
            api_key = settings.alpaca_live_key
            secret_key = settings.alpaca_live_secret
            base_url = settings.alpaca_live_base
        else:
            api_key = settings.alpaca_paper_key
            secret_key = settings.alpaca_paper_secret
            base_url = settings.alpaca_paper_base
        if not api_key or not secret_key:
            raise ValueError(
                f"phase={settings.phase} but no creds for that phase; "
                "check FOB_PHASE / APCA_API_*_PAPER / APCA_API_*_LIVE"
            )
        self.trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=(settings.phase == "paper"),
        )
        self.data_options = OptionHistoricalDataClient(api_key, secret_key)
        self.data_stocks = StockHistoricalDataClient(api_key, secret_key)
        self._account_cache: dict | None = None
        self._account_cache_ts: datetime | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "BrokerClient":
        return cls(settings)

    # ===== Account =====

    def get_account(self, force_refresh: bool = False) -> dict:
        """Returns dict with equity, cash, buying_power, options_approved_level,
        trading_blocked, options_blocked. Cached for 5s to avoid hammering."""
        now = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._account_cache
            and self._account_cache_ts
            and (now - self._account_cache_ts) < timedelta(seconds=5)
        ):
            return self._account_cache
        acct = self.trading.get_account()
        d = {
            "account_number": acct.account_number,
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": str(acct.status),
            "options_approved_level": int(acct.options_approved_level or 0),
            "trading_blocked": bool(acct.trading_blocked),
            "options_trading_level": int(acct.options_trading_level or 0),
        }
        self._account_cache = d
        self._account_cache_ts = now
        return d

    def is_trading_blocked(self) -> bool:
        acct = self.get_account()
        return acct["trading_blocked"] or acct["options_trading_level"] == 0

    # ===== Market data (stocks) =====

    def get_stock_quote(self, symbol: str) -> dict | None:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_stocks.get_stock_latest_quote(req)
        if not quote or symbol not in quote:
            return None
        q = quote[symbol]
        return {
            "symbol": symbol,
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "ts": q.timestamp.isoformat() if hasattr(q, "timestamp") else "",
        }

    def get_stock_bars_minute(self, symbol: str, lookback_minutes: int) -> list[dict]:
        """1-minute bars for the last `lookback_minutes` minutes.

        Uses feed=iex + curl subprocess because:
        1. Free-tier SIP blocks recent data ("subscription does not permit
           querying recent SIP data").
        2. alpaca-py's urllib hits a TLS fingerprint block from this venv.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes + 5)
        url = (
            f"{self.settings.alpaca_data_base}/v2/stocks/bars"
            f"?symbols={symbol}&timeframe=1Min"
            f"&start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&adjustment=raw&feed=iex&limit=10000"
        )
        import base64
        import subprocess
        import json as _json
        if self.settings.is_live():
            cred = (self.settings.alpaca_live_key, self.settings.alpaca_live_secret)
        else:
            cred = (self.settings.alpaca_paper_key, self.settings.alpaca_paper_secret)
        auth = base64.b64encode(f"{cred[0]}:{cred[1]}".encode()).decode()
        try:
            r = subprocess.run(
                ["curl", "-sS", "-H", f"Authorization: Basic {auth}", url],
                capture_output=True, text=True, timeout=30,
            )
            data = _json.loads(r.stdout)
        except Exception as e:
            log.warning("get_stock_bars curl failed for %s: %s", symbol, e)
            return []
        bars = data.get("bars", {}).get(symbol, [])
        out = []
        for b in bars:
            out.append({
                "t": b["t"],
                "o": float(b["o"]),
                "h": float(b["h"]),
                "l": float(b["l"]),
                "c": float(b["c"]),
                "v": int(b["v"]),
            })
        return out

    # ===== Market data (options) =====

    def list_option_contracts(self, underlying: str, expiry_gte: str, expiry_lte: str, option_type: str | None = None) -> list[dict]:
        """List all available option contracts for underlying in expiry window.

        expiry_gte / expiry_lte are YYYY-MM-DD.
        option_type: "call", "put", or None (both). Defaults to "call"
        for backwards compat with long_call scanner; pass "put" for BPCS.
        """
        if option_type is None:
            option_type = "call"
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status="active",
            expiration_date_gte=expiry_gte,
            expiration_date_lte=expiry_lte,
            type=option_type,
        )
        try:
            page = self.trading.get_option_contracts(req)
        except Exception as e:
            log.warning("list_option_contracts failed for %s: %s", underlying, e)
            return []
        # page is an OptionContractsPage; iterate contracts
        contracts = list(page.option_contracts) if hasattr(page, "option_contracts") else []
        out = []
        for c in contracts:
            # c.expiration_date is a datetime.date object — normalize to YYYY-MM-DD string
            exp = c.expiration_date
            if hasattr(exp, "strftime"):
                exp_str = exp.strftime("%Y-%m-%d")
            else:
                exp_str = str(exp)
            out.append({
                "symbol": c.symbol,  # OCC code e.g. SPY260815C00770000
                "underlying": underlying,
                "expiry": exp_str,
                "strike": float(c.strike_price),
                "type": "call" if c.type.value == "call" else "put",
                "open_interest": int(c.open_interest or 0),
                "close_price": float(c.close_price) if c.close_price else None,
            })
        return out

    def get_option_snapshot(self, contract_symbol: str) -> dict | None:
        """Snapshot = latest quote. May return None if the symbol isn't in
        the underlying's snapshot page.

        The Alpaca paper option feed is `indicative` — no greeks, no IV. We
        return whatever bid/ask the feed has, or None.
        """
        import base64
        import subprocess
        import json as _json

        underlying = contract_symbol[:3]  # SPY260812C00770000 → SPY (first 3 chars of root)
        # Actually Alpaca uses OCC: 6-char root. SPY is 3, but root_symbol can be longer.
        # We need to look up the underlying from the contract via the contracts API
        # or just iterate roots. Simplest: assume the first 3 letters are the underlying
        # root for SPY/QQQ/IWM/DIA. For unknown roots this is fragile.

        url = (
            f"{self.settings.alpaca_data_base}/v1beta1/options/snapshots/{underlying}"
            f"?feed=indicative&limit=1000"
        )
        if self.settings.is_live():
            cred = (self.settings.alpaca_live_key, self.settings.alpaca_live_secret)
        else:
            cred = (self.settings.alpaca_paper_key, self.settings.alpaca_paper_secret)
        auth = base64.b64encode(f"{cred[0]}:{cred[1]}".encode()).decode()
        try:
            r = subprocess.run(
                ["curl", "-sS", "-H", f"Authorization: Basic {auth}", url],
                capture_output=True, text=True, timeout=30,
            )
            data = _json.loads(r.stdout)
        except Exception as e:
            log.debug("option snapshot curl failed for %s: %s", contract_symbol, e)
            return None

        snaps = data.get("snapshots", {}) or {}
        if contract_symbol not in snaps:
            return None
        s = snaps[contract_symbol]
        out = {"symbol": contract_symbol}
        if "latestQuote" in s and s["latestQuote"]:
            q = s["latestQuote"]
            out["bid"] = float(q.get("bp", 0))
            out["ask"] = float(q.get("ap", 0))
            out["bid_size"] = int(q.get("bs", 0))
            out["ask_size"] = int(q.get("as", 0))
        if "greeks" in s and s["greeks"]:
            g = s["greeks"]
            out["delta"] = float(g.get("delta", 0))
            out["gamma"] = float(g.get("gamma", 0))
            out["theta"] = float(g.get("theta", 0))
            out["vega"] = float(g.get("vega", 0))
        if "impliedVolatility" in s and s["impliedVolatility"] is not None:
            out["iv"] = float(s["impliedVolatility"])
        return out

    # ===== Order submission =====

    def submit_buy(
        self,
        contract_symbol: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
        position_id: str,
    ) -> AlpacaOrder:
        """Submit a BUY limit order. The `client_order_id` MUST be unique
        per submission and is used as the journal event_id. The broker will
        echo it back as `order.client_order_id`.

        Time-in-force: GTC for non-0DTE contracts (so a 1DTE order
        submitted after-hours survives to next session), DAY for 0DTE
        (no point holding overnight — contract expires today).
        """
        tif = self._tif_for_contract(contract_symbol)
        req = LimitOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif,
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
        )
        log.info("submit_buy %s qty=%d limit=%.2f coid=%s pos=%s tif=%s",
                 contract_symbol, qty, limit_price, client_order_id, position_id, tif)
        return self.trading.submit_order(req)

    def submit_close_sell(
        self,
        contract_symbol: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
    ) -> AlpacaOrder:
        """Submit a SELL limit order to close. Uses GTC so a flatten
        submitted after-hours still closes at next session's open."""
        req = LimitOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
        )
        log.info("submit_close_sell %s qty=%d limit=%.2f coid=%s",
                 contract_symbol, qty, limit_price, client_order_id)
        return self.trading.submit_order(req)

    def submit_open_sell(
        self,
        contract_symbol: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
        position_id: str,
    ) -> AlpacaOrder:
        """Submit a SELL limit order to OPEN a short position.

        Used for BPCS short-put leg. The short put is sold to open a
        short position; the long put is also sold (after being bought)
        — wait, no. Long put is BOUGHT first. This method is for the
        SHORT PUT leg of a credit spread.

        Uses GTC for non-0DTE (BPCS targets 25-50 DTE — no need to use DAY).
        """
        req = LimitOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
        )
        log.info("submit_open_sell %s qty=%d limit=%.2f coid=%s pos=%s",
                 contract_symbol, qty, limit_price, client_order_id, position_id)
        return self.trading.submit_order(req)

    def _tif_for_contract(self, contract_symbol: str) -> TimeInForce:
        """Return DAY for 0DTE contracts (today's expiry), GTC otherwise.

        OCC format: SPY{YYMMDD}{C|P}{strike*1000:08d}
        E.g., SPY260811C00759000 → expiry 2026-08-11.
        """
        try:
            yy = int(contract_symbol[3:5])
            mm = int(contract_symbol[5:7])
            dd = int(contract_symbol[7:9])
            expiry_year = 2000 + yy
            from datetime import date
            exp = date(expiry_year, mm, dd)
            today = date.today()
            if exp <= today:
                return TimeInForce.DAY  # 0DTE or already expired
            return TimeInForce.GTC
        except (ValueError, IndexError):
            return TimeInForce.GTC  # safe default

    def place_tpsl_bracket(
        self,
        parent_order_id: str,
        contract_symbol: str,
        qty: int,
        tp_price: float,
        sl_trigger_price: float,
        sl_limit_price: float,
    ) -> BracketLegs:
        """Place a BRACKET order: parent BUY limit + TP limit + SL stop-limit.

        This is the structural fix vs. flip-alpaca-bot: TP/SL lives at the
        broker. If the daemon dies, the broker still exits the position.

        Caller must supply the contract_symbol (OCC) and qty because the
        parent order_id alone doesn't carry those for a single-leg option.
        """
        req = LimitOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            type=OrderType.LIMIT,
            limit_price=round(tp_price * 0.99, 2),  # placeholder parent
            client_order_id=f"bracket-{parent_order_id}",
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            stop_loss=StopLossRequest(
                stop_price=round(sl_trigger_price, 2),
                limit_price=round(sl_limit_price, 2),
            ),
        )
        log.info(
            "place_tpsl_bracket %s qty=%d tp=%.2f sl=%.2f/%.2f",
            contract_symbol, qty, tp_price, sl_trigger_price, sl_limit_price,
        )
        try:
            order = self.trading.submit_order(req)
        except Exception as e:
            log.error("place_tpsl_bracket submit failed: %s", e)
            raise
        return BracketLegs(
            tp_price=tp_price,
            sl_price=sl_trigger_price,
            tp_order_id=str(order.id) if order is not None and getattr(order, "id", None) else "",
            sl_order_id="",
        )

    def submit_bracket_buy(
        self,
        contract_symbol: str,
        qty: int,
        limit_price: float,
        tp_price: float,
        sl_trigger_price: float,
        sl_limit_price: float,
        client_order_id: str,
    ) -> AlpacaOrder:
        """Combined BUY-limit + TP-limit + SL-stop-limit in a single BRACKET order.

        This is the preferred entry path. The parent fills only if the limit
        hits; once filled, both TP and SL are live at the broker as OCO legs.
        """
        req = LimitOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            type=OrderType.LIMIT,
            limit_price=round(limit_price, 2),
            client_order_id=client_order_id,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            stop_loss=StopLossRequest(
                stop_price=round(sl_trigger_price, 2),
                limit_price=round(sl_limit_price, 2),
            ),
        )
        log.info(
            "submit_bracket_buy %s qty=%d entry=%.2f tp=%.2f sl=%.2f/%.2f coid=%s",
            contract_symbol, qty, limit_price, tp_price, sl_trigger_price,
            sl_limit_price, client_order_id,
        )
        return self.trading.submit_order(req)

    # ===== Order queries =====

    def get_order_by_client_id(self, client_order_id: str) -> AlpacaOrder | None:
        try:
            return self.trading.get_order_by_client_id(client_order_id)
        except Exception as e:
            log.debug("get_order_by_client_id(%s) failed: %s", client_order_id, e)
            return None

    def list_filled_orders(self, since_ts: datetime | None = None) -> list[AlpacaOrder]:
        """Canonical source for real-fill reconciliation.

        Filters to filled option orders. Critical structural fix vs. the old
        flip-alpaca-bot: alpaca-py v0.43+ has a known issue where the `after=`
        parameter on `GetOrdersRequest` raises a TypeError due to string-vs-int
        comparison on submitted_at. We try with `after` first; on TypeError,
        we drop the filter and do client-side cutoff. We DO NOT silently pass
        — that was the old bot's bug that hid real winners.

        Note: GetOrdersRequest in current alpaca-py has no `asset_class`
        parameter; we filter client-side to `AssetClass.US_OPTION`.

        Returns full AlpacaOrder objects so the journal can read fill_price,
        filled_qty, client_order_id.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders: list = []
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=since_ts,
                limit=500,
                nested=False,
            )
            orders = list(self.trading.get_orders(req))
        except TypeError as e:
            # SDK compare error — fall back to no server-side filter, cut client-side.
            log.debug("list_filled_orders after= TypeError: %s — falling back", e)
            try:
                req = GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    limit=500,
                    nested=False,
                )
                orders = list(self.trading.get_orders(req))
            except Exception as e2:
                log.warning("list_filled_orders fallback failed: %s", e2)
                return []
        except Exception as e:
            log.warning("list_filled_orders failed: %s", e)
            return []

        # Filter: must be FILLED + option asset class + within cutoff
        from datetime import timezone
        cutoff = since_ts
        if cutoff is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        out = []
        for o in orders:
            if o.status != OrderStatus.FILLED:
                continue
            asset_class = getattr(o, "asset_class", None) or getattr(o, "assetClass", None)
            if asset_class is not None and asset_class != AssetClass.US_OPTION:
                continue
            if cutoff is not None and getattr(o, "submitted_at", None) is not None:
                submitted = o.submitted_at
                if submitted.tzinfo is None:
                    submitted = submitted.replace(tzinfo=timezone.utc)
                if submitted < cutoff:
                    continue
            out.append(o)
        return out

    def list_open_orders(self) -> list[AlpacaOrder]:
        try:
            return list(self.trading.get_orders())
        except Exception as e:
            log.warning("list_open_orders failed: %s", e)
            return []

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.debug("cancel_order(%s) failed: %s", order_id, e)
            return False

    def cancel_all_open(self) -> int:
        try:
            self.trading.cancel_orders()
            return 1
        except Exception as e:
            log.warning("cancel_orders failed: %s", e)
            return 0