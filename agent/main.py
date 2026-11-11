"""FastAPI interface of the MLOps agent.

    uvicorn agent.main:app --port 8000

POST /chat                     ask the agent (natural language)
GET  /actions                  pending state-changing actions
POST /actions/{id}/confirm     execute a pending action (the only way anything changes)
POST /actions/{id}/reject      discard a pending action
GET  /tools                    tool catalogue with schemas and permission class
GET  /health

Every response carries an X-Request-ID header (taken from the request if valid) that
also appears in the structured tool logs. If AGENT_API_TOKEN is set, every endpoint
except /health requires "Authorization: Bearer <token>".
"""

import logging
import os
import re
import secrets
import threading
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from agent.gemini_pool import ModelPool
from agent.llm import GeminiLLM, LLMError
from agent.orchestrator import Agent
from agent.schemas import ActionResult, AgentResponse, ChatRequest, PendingAction
from agent.tools.base import new_request_id

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(agent_factory: Callable[[], Agent] | None = None) -> FastAPI:
    # The real service reads .env (git-ignored; existing environment variables win) lazily,
    # not at import time. Tests pass their own factory, so they never load a developer's .env.
    load_env: Callable[[], Any] = load_dotenv if agent_factory is None else (lambda: None)

    def default_agent() -> Agent:
        load_env()
        return Agent(GeminiLLM())

    factory = agent_factory or default_agent
    holder: dict[str, Agent] = {}
    lock = threading.Lock()

    def get_agent() -> Agent:
        with lock:
            if "agent" not in holder:
                try:
                    holder["agent"] = factory()
                except LLMError as exc:
                    raise HTTPException(503, detail=f"agent not configured: {exc}") from exc
            return holder["agent"]

    def check_token(authorization: str | None = Header(default=None)) -> None:
        token = os.environ.get("AGENT_API_TOKEN")
        if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, detail="missing or invalid bearer token")

    app = FastAPI(title="MLOps Agent", version="1.0.0")
    auth = [Depends(check_token)]

    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Any:
        incoming = request.headers.get("X-Request-ID", "")
        rid = incoming if _REQUEST_ID_RE.match(incoming) else new_request_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        load_env()
        body: dict[str, Any] = {
            "status": "ok",
            "llm_configured": bool(os.environ.get("GEMINI_API_KEY")),
        }
        if agent_factory is None:  # real service: show the free-tier chain (no API call made)
            body["llm_models"] = ModelPool().status()
        return body

    @app.get("/tools", dependencies=auth)
    def tools() -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description.split("\n")[0],
                "state_changing": s.state_changing,
                "parameters": s.parameters_schema(),
            }
            for s in get_agent().registry.values()
        ]

    @app.post("/chat", response_model=AgentResponse, dependencies=auth)
    def chat(body: ChatRequest, request: Request) -> AgentResponse:
        return get_agent().run(body.message, request_id=request.state.request_id)

    @app.get("/actions", response_model=list[PendingAction], dependencies=auth)
    def pending_actions() -> list[PendingAction]:
        return get_agent().actions.pending()

    @app.post("/actions/{action_id}/confirm", response_model=ActionResult, dependencies=auth)
    def confirm(action_id: str) -> Any:
        result = get_agent().execute_action(action_id)
        if result.status == "not_found":
            return JSONResponse(status_code=404, content=result.model_dump(mode="json"))
        return result

    @app.post("/actions/{action_id}/reject", response_model=ActionResult, dependencies=auth)
    def reject(action_id: str) -> Any:
        result = get_agent().reject_action(action_id)
        if result.status == "not_found":
            return JSONResponse(status_code=404, content=result.model_dump(mode="json"))
        return result

    return app


app = create_app()
