"""Execution package — risk-gated order submission + reconciliation."""

from .executor import Executor, ExecutionResult

__all__ = ["Executor", "ExecutionResult"]