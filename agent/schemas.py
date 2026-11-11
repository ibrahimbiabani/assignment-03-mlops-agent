"""Structured input/output types shared by the tools, the orchestrator and the API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    REJECTED = "rejected"
    CONFIRMATION_REQUIRED = "confirmation_required"


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


# ---------- agent API ----------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ToolCallRecord(BaseModel):
    """One tool call made while answering a request (the evidence trail)."""

    tool: str
    arguments: dict[str, Any]
    status: ToolStatus
    duration_ms: float
    error: str | None = None
    data: dict[str, Any] | None = None
    action_id: str | None = None  # set when the call became a pending action


class PendingAction(BaseModel):
    """A state-changing operation proposed by the agent, awaiting user confirmation."""

    action_id: str
    tool: str
    arguments: dict[str, Any]
    request_id: str
    created_at: str
    expires_at: str


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    tool_calls: list[ToolCallRecord]
    pending_actions: list[PendingAction]
    unavailable_sources: list[str] = Field(
        description="Tools that failed (error/timeout); their data is NOT in the answer."
    )
    steps: int
    model: str = Field(description="Model(s) that produced this answer, in order of use")
    llm_fallbacks: list[str] = Field(
        default_factory=list, description="Models skipped during this request, with the reason"
    )
    duration_ms: float
    error: str | None = None


class ActionResult(BaseModel):
    action_id: str
    status: str  # executed | rejected | not_found
    result: ToolResult | None = None
    detail: str | None = None
