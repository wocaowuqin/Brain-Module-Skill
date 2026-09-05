"""Execution skill boundary for existing safe resource managers."""
from __future__ import annotations
from typing import Any
from bms.base import BMSContext, BaseSkill, _check_layer

class ExecutionSkill(BaseSkill):
    """Delegate commit/rollback to an injected executor; never owns a ledger."""
    def __init__(self, executor: Any = None): self.executor = executor

    @_check_layer("skill")
    def execute(self, context: BMSContext, params: dict) -> dict:
        if self.executor is None: return {"status": "rejected", "data": None, "metadata": {"reason": "executor_not_configured"}}
        operation = str(params.get("operation", "commit"))
        fn = getattr(self.executor, operation, None)
        if not callable(fn): return {"status": "rejected", "data": None, "metadata": {"reason": f"unsupported_operation:{operation}"}}
        try:
            data = fn(params.get("payload", params))
            return {"status": "success", "data": data, "metadata": {"operation": operation}}
        except Exception as exc:
            return {"status": "error", "data": None, "metadata": {"operation": operation, "exception": type(exc).__name__, "reason": str(exc)}}

__all__ = ["ExecutionSkill"]
