"""Signal package — scanner + funnel recorder.

The FunnelRecorder is the primary diagnostic instrument. Every scan must
emit a row even when no candidates pass; that's how we diagnose collapses.
"""

from .funnel import FunnelRecorder, FunnelRow

__all__ = ["FunnelRecorder", "FunnelRow"]