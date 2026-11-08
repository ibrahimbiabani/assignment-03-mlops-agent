"""Shared tool plumbing: error types, timing, and structured logging."""

import functools
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from agent.schemas import ToolResult, ToolStatus

logger = logging.getLogger("mlops_agent.tools")

# Set once per incoming user request so every tool log line can be correlated.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


class ToolError(Exception):
    """A tool failed while talking to its backing service."""


class ToolTimeout(ToolError):
    """The backing service did not respond in time."""


class InvalidToolRequest(ToolError):
    """The arguments supplied to the tool are not acceptable."""


class ToolRejected(ToolError):
    """The operation is not permitted (allowlist / permission check)."""


_STATUS_FOR = {
    ToolTimeout: ToolStatus.TIMEOUT,
    InvalidToolRequest: ToolStatus.INVALID_REQUEST,
    ToolRejected: ToolStatus.REJECTED,
}


def tool(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., ToolResult]]:
    """Wrap a function returning a dict so it always returns a ToolResult and logs one line.

    Log lines contain the tool name, status, duration and request id only;
    arguments and outputs are never logged, so secrets cannot leak through them.
    """

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., ToolResult]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            start = time.perf_counter()
            try:
                result = ToolResult(tool=name, status=ToolStatus.SUCCESS, data=fn(*args, **kwargs))
            except ToolError as exc:
                status = _STATUS_FOR.get(type(exc), ToolStatus.ERROR)
                result = ToolResult(tool=name, status=status, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - never let a tool crash the agent
                result = ToolResult(
                    tool=name, status=ToolStatus.ERROR, error=f"{type(exc).__name__}: {exc}"
                )
            result.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                json.dumps(
                    {
                        "event": "tool_call",
                        "request_id": request_id_var.get(),
                        "tool": name,
                        "status": result.status.value,
                        "duration_ms": result.duration_ms,
                    }
                )
            )
            return result

        return wrapper

    return decorator
