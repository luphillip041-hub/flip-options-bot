"""Broker package — wraps alpaca-py with the structural fixes baked in.

See `alpaca.py` for the structural-fix list (idempotent client_order_id,
canonical real-fill source, no market orders, etc.).
"""

from .alpaca import BracketLegs, BrokerClient

__all__ = ["BrokerClient", "BracketLegs"]
