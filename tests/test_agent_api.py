"""Agent HTTP API tests (scripted LLM, fake tools)."""

import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError, LLMTurn, ToolCall
from agent.main import create_app
from agent.orchestrator import Agent
from agent.registry import spec_from
from agent.tools.base import ConfirmationRequired, tool
from tests.test_agent import ScriptedLLM

executed = []


@tool("api_fake_status")
def api_fake_status() -> dict:
    """Status."""
    return {"healthy": True}


@tool("api_fake_scale")
def api_fake_scale(replicas: int, confirm: bool = False) -> dict:
    """Scale."""
    if not confirm:
        raise ConfirmationRequired("needs confirmation")
    executed.append(replicas)
    return {"replicas": replicas}


REGISTRY = {s.name: s for s in (spec_from(api_fake_status), spec_from(api_fake_scale, True))}


def client_for(turns, monkeypatch, token=None):
    executed.clear()
    if token:
        monkeypatch.setenv("AGENT_API_TOKEN", token)
    else:
        monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    agent = Agent(ScriptedLLM(turns), REGISTRY)
    return TestClient(create_app(lambda: agent))


def test_chat_returns_structured_response_with_request_id(monkeypatch):
    turns = [LLMTurn(tool_calls=[ToolCall("api_fake_status", {}, "1")]), LLMTurn("All healthy.")]
    c = client_for(turns, monkeypatch)

    r = c.post("/chat", json={"message": "is it healthy?"}, headers={"X-Request-ID": "demo-001"})

    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == "demo-001"
    body = r.json()
    assert body["request_id"] == "demo-001"
    assert body["answer"] == "All healthy."
    assert body["tool_calls"][0]["tool"] == "api_fake_status"
    assert body["tool_calls"][0]["data"] == {"healthy": True}


def test_invalid_request_id_header_is_replaced(monkeypatch):
    c = client_for([LLMTurn("hi")], monkeypatch)
    r = c.post("/chat", json={"message": "hi"}, headers={"X-Request-ID": "bad id\n<script>"})
    assert r.headers["X-Request-ID"] != "bad id\n<script>"
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "x" * 4001}])
def test_invalid_chat_requests_rejected(monkeypatch, payload):
    assert client_for([], monkeypatch).post("/chat", json=payload).status_code == 422


def test_confirm_flow_over_http(monkeypatch):
    turns = [
        LLMTurn(tool_calls=[ToolCall("api_fake_scale", {"replicas": 3}, "1")]),
        LLMTurn("Scaling to 3 is pending your confirmation."),
    ]
    c = client_for(turns, monkeypatch)

    body = c.post("/chat", json={"message": "scale to 3"}).json()
    action_id = body["pending_actions"][0]["action_id"]
    assert executed == []
    assert [a["action_id"] for a in c.get("/actions").json()] == [action_id]

    r = c.post(f"/actions/{action_id}/confirm")
    assert r.status_code == 200 and r.json()["status"] == "executed"
    assert executed == [3]
    assert c.post(f"/actions/{action_id}/confirm").status_code == 404  # single use
    assert c.get("/actions").json() == []


def test_reject_flow(monkeypatch):
    turns = [LLMTurn(tool_calls=[ToolCall("api_fake_scale", {"replicas": 2}, "1")]), LLMTurn("ok")]
    c = client_for(turns, monkeypatch)
    action_id = c.post("/chat", json={"message": "scale"}).json()["pending_actions"][0]["action_id"]

    assert c.post(f"/actions/{action_id}/reject").json()["status"] == "rejected"
    assert c.post(f"/actions/{action_id}/confirm").status_code == 404
    assert executed == []


def test_bearer_token_required_when_configured(monkeypatch):
    c = client_for([LLMTurn("hi")], monkeypatch, token="t0ken")
    assert c.get("/health").status_code == 200  # health stays open
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert c.get("/actions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = c.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer t0ken"})
    assert ok.status_code == 200


def test_tools_endpoint_lists_permission_class(monkeypatch):
    tools = {t["name"]: t for t in client_for([], monkeypatch).get("/tools").json()}
    assert tools["api_fake_scale"]["state_changing"] is True
    assert tools["api_fake_status"]["state_changing"] is False
    assert "confirm" not in tools["api_fake_scale"]["parameters"]["properties"]


def test_unconfigured_llm_returns_503(monkeypatch):
    def factory():
        raise LLMError("GEMINI_API_KEY is not set")

    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    c = TestClient(create_app(factory))
    r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["detail"]
