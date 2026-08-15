"""Trade journal — append-only log of fills and closes.

See journal.py for the structural fix details (idempotent writes, no
duplicate close events, canonical position_id UUIDs).
"""

from .journal import Journal, TradeEvent

__all__ = ["Journal", "TradeEvent"]
