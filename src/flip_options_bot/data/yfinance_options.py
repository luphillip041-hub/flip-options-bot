"""yfinance option-chain sidecar for free forward snapshots.

This module intentionally does NOT provide historical OPRA data and must not be
used as broker/execution truth. It only confirms/enriches current/delayed
1DTE+ option candidates with Yahoo chain fields when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("flip_options_bot.yfinance_options")


@dataclass(frozen=True)
class YFinanceOptionQuote:
    contract_symbol: str
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None

    @property
    def has_bid_ask(self) -> bool:
        return (
            self.bid is not None
            and self.ask is not None
            and self.bid > 0
            and self.ask > self.bid
        )

    @property
    def mid(self) -> float | None:
        if not self.has_bid_ask:
            return None
        bid = self.bid
        ask = self.ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        if mid is None:
            return None
        bid = self.bid
        ask = self.ask
        if bid is None or ask is None:
            return None
        return (ask - bid) / max(mid, 0.01)


def _float_or_none(value) -> float | None:
    try:
        if value is None or value != value:
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


class YFinanceOptionChainProvider:
    """Fetch current/delayed Yahoo option-chain rows by OCC contract symbol.

    Caches `(underlying, expiry)` chains for the lifetime of the provider so a
    scan can enrich several strikes without hammering Yahoo.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], dict[str, YFinanceOptionQuote]] = {}

    def get_quote(
        self, underlying: str, expiry: str, contract_symbol: str
    ) -> YFinanceOptionQuote | None:
        chain = self._get_chain(underlying.upper(), expiry)
        return chain.get(contract_symbol)

    def _get_chain(self, underlying: str, expiry: str) -> dict[str, YFinanceOptionQuote]:
        key = (underlying, expiry)
        if key in self._cache:
            return self._cache[key]
        try:
            import yfinance as yf

            chain = yf.Ticker(underlying).option_chain(expiry)
        except Exception as exc:
            log.info("yfinance chain fetch failed for %s %s: %s", underlying, expiry, exc)
            self._cache[key] = {}
            return {}

        rows: dict[str, YFinanceOptionQuote] = {}
        for df in (getattr(chain, "calls", None), getattr(chain, "puts", None)):
            if df is None:
                continue
            for _, row in df.iterrows():
                symbol = str(row.get("contractSymbol", ""))
                if not symbol:
                    continue
                rows[symbol] = YFinanceOptionQuote(
                    contract_symbol=symbol,
                    bid=_float_or_none(row.get("bid")),
                    ask=_float_or_none(row.get("ask")),
                    last_price=_float_or_none(row.get("lastPrice")),
                    volume=_int_or_none(row.get("volume")),
                    open_interest=_int_or_none(row.get("openInterest")),
                    implied_volatility=_float_or_none(row.get("impliedVolatility")),
                )
        self._cache[key] = rows
        return rows
