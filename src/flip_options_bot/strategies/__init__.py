"""Strategy registry — pluggable strategies loaded by name.

Pattern from go-trader-prior-art + lean-algorithm-python-prior-art:
- Strategies are auto-discovered by their `STRATEGY_ID` module-level constant.
- Adding a new strategy = dropping a new file in this package. The daemon
  imports `STRATEGIES` from here and iterates the enabled ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config import Settings
from . import long_call, bull_put_credit


@dataclass
class StrategyDescriptor:
    strategy_id: str
    is_enabled: Callable[[Settings], bool]
    make_filters: Callable
    # Signal-evaluation functions, all optional. The scanner calls these
    # in order; any function returning None cuts the candidate from the funnel.
    passes_dte: Callable | None = None
    pick_expiry: Callable | None = None
    compute_conviction: Callable | None = None
    passes_conviction: Callable | None = None


def _is_long_call_enabled(s: Settings) -> bool:
    return s.long_call_enabled


def _is_bpcs_enabled(s: Settings) -> bool:
    return s.bpcs_enabled


STRATEGIES: dict[str, StrategyDescriptor] = {
    "long_call": StrategyDescriptor(
        strategy_id="long_call",
        is_enabled=_is_long_call_enabled,
        make_filters=long_call.make_filters_from_settings,
        passes_dte=long_call.passes_dte_window,
        pick_expiry=long_call.pick_target_expiry,
        compute_conviction=long_call.compute_conviction,
        passes_conviction=long_call.passes_conviction,
    ),
    "bull_put_credit_spread": StrategyDescriptor(
        strategy_id="bull_put_credit_spread",
        is_enabled=_is_bpcs_enabled,
        make_filters=bull_put_credit.make_filters_from_settings,
        passes_dte=bull_put_credit.passes_dte_window,
        pick_expiry=bull_put_credit.pick_target_expiry,
        compute_conviction=bull_put_credit.compute_bpcs_conviction,
        passes_conviction=bull_put_credit.passes_bpcs_conviction,
    ),
}


def enabled_strategies(settings: Settings) -> list[StrategyDescriptor]:
    return [s for s in STRATEGIES.values() if s.is_enabled(settings)]


def get_strategy(strategy_id: str) -> StrategyDescriptor | None:
    return STRATEGIES.get(strategy_id)