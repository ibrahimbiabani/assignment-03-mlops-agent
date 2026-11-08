"""Structured input/output types shared by every agent tool."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    REJECTED = "rejected"


class ToolResult(BaseModel):
    """Every tool returns this, so the agent can tell real evidence from failure.

    `data` is only populated on SUCCESS. On any other status the agent must
    report the error instead of inventing a result.
    """

    tool: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0)

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS
