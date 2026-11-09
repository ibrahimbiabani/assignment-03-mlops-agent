"""Airflow tool tests against an in-process fake Airflow 3 API over real HTTP."""

import json
import logging
import socket

import pytest

from agent.schemas import ToolStatus
from agent.tools import airflow_tools
from tests.fake_airflow import FAILED_RUN, PASSWORD, TOKEN, USERNAME, FakeAirflow


@pytest.fixture
def airflow(monkeypatch):
    fake = FakeAirflow()
    monkeypatch.setenv("AIRFLOW_BASE_URL", fake.url)
    monkeypatch.setenv("AIRFLOW_USERNAME", USERNAME)
    monkeypatch.setenv("AIRFLOW_PASSWORD", PASSWORD)
    monkeypatch.setenv("AIRFLOW_TIMEOUT_S", "2")
    monkeypatch.delenv("AIRFLOW_ALLOWED_TRIGGER_DAGS", raising=False)
    yield fake
    fake.close()


@pytest.fixture
def shared_client(airflow, monkeypatch):
    """Reuse one client across calls, as a long-running agent process would."""
    client = airflow_tools.get_client()
    monkeypatch.setattr(airflow_tools, "get_client", lambda: client)
    return client


# ---------- successful tool execution ----------


def test_list_dags_authenticates_with_bearer_token(airflow):
    result = airflow_tools.list_dags()

    assert result.ok
    dags = {d["dag_id"]: d for d in result.data["dags"]}
    assert dags["training_pipeline"]["tags"] == ["ml"]
    assert dags["data_backup"]["is_paused"] is True
    assert airflow.calls("GET")[-1]["auth"] == f"Bearer {TOKEN}"


def test_token_is_reused_and_refreshed_when_expired(airflow, shared_client):
    airflow_tools.list_dags()
    airflow_tools.list_dags()
    assert len([c for c in airflow.calls("POST") if c["path"] == "/auth/token"]) == 1

    shared_client._token = "expired-token"  # server answers 401 -> client logs in again

    assert airflow_tools.list_dags().ok
    assert len([c for c in airflow.calls("POST") if c["path"] == "/auth/token"]) == 2


def test_get_dag_runs_newest_first_with_state_filter(airflow):
    result = airflow_tools.get_dag_runs("training_pipeline", state="failed")

    assert result.ok
    assert [r["dag_run_id"] for r in result.data["runs"]] == [FAILED_RUN]
    query = airflow.calls("GET")[-1]["query"]
    assert query == {"limit": "10", "order_by": "-run_after", "state": "failed"}


def test_get_dag_run_reports_task_states(airflow):
    result = airflow_tools.get_dag_run("training_pipeline", FAILED_RUN)

    assert result.ok
    assert result.data["run"]["state"] == "failed"
    assert result.data["task_state_counts"] == {
        "success": 2,
        "failed": 1,
        "upstream_failed": 2,
    }


def test_get_failed_tasks_finds_latest_failed_run_and_root_cause(airflow):
    result = airflow_tools.get_failed_tasks("training_pipeline")

    assert result.ok
    assert result.data["dag_run_id"] == FAILED_RUN
    assert result.data["root_cause_tasks"] == ["train_model"]
    states = {t["task_id"]: t["state"] for t in result.data["failed_tasks"]}
    assert states == {
        "train_model": "failed",
        "evaluate_model": "upstream_failed",
        "quality_gate": "upstream_failed",
    }


def test_get_task_logs_uses_latest_try_extracts_errors_and_redacts_secrets(airflow):
    result = airflow_tools.get_task_logs("training_pipeline", FAILED_RUN, "train_model")

    assert result.ok
    assert result.data["try_number"] == 3
    assert result.data["error_lines"] == ["[error] Task failed with exception"]
    # The real failure reason comes from Airflow 3's structured error_detail.
    exc = result.data["exceptions"][0]
    assert exc["type"] == "ValueError"
    assert exc["message"].startswith("Input X contains NaN.")
    assert exc["location"] == "/opt/airflow/project/training/train.py:171 in train_model"
    dumped = json.dumps(result.data)
    assert "hunter2" not in dumped and "abc123" not in dumped
    assert "password=***" in " ".join(result.data["tail"])


def test_get_task_logs_accepts_plain_string_content(airflow):
    result = airflow_tools.get_task_logs(
        "training_pipeline", FAILED_RUN, "validate_dataset", try_number=1
    )
    assert result.data["error_lines"] == ["ERROR something went wrong"]
    assert result.data["exceptions"] == []


def test_health(airflow):
    result = airflow_tools.health()
    assert result.data["scheduler"] == {
        "status": "healthy",
        "latest_heartbeat": "2026-11-09T10:05:00Z",
    }
    assert result.data["dag_processor"] is None


# ---------- trigger safety ----------


def test_trigger_without_confirmation_is_rejected_and_sends_nothing(airflow):
    result = airflow_tools.trigger_dag("training_pipeline")

    assert result.status == ToolStatus.REJECTED
    assert "confirmation" in result.error
    assert airflow.calls() == []


def test_trigger_of_unapproved_dag_is_rejected_even_when_confirmed(airflow):
    result = airflow_tools.trigger_dag("data_backup", confirm=True)

    assert result.status == ToolStatus.REJECTED
    assert "approved" in result.error
    assert airflow.calls() == []


def test_confirmed_trigger_of_approved_dag(airflow):
    result = airflow_tools.trigger_dag("training_pipeline", conf={"epochs": 5}, confirm=True)

    assert result.ok
    assert result.data["run"]["state"] == "queued"
    assert "warning" not in result.data
    body = airflow.calls("POST")[-1]["body"]
    assert body["logical_date"] is None  # required-but-nullable in Airflow 3
    assert body["conf"] == {"epochs": 5}


def test_trigger_of_paused_dag_warns(airflow, monkeypatch):
    monkeypatch.setenv("AIRFLOW_ALLOWED_TRIGGER_DAGS", "training_pipeline,data_backup")
    result = airflow_tools.trigger_dag("data_backup", confirm=True)
    assert result.ok
    assert "paused" in result.data["warning"]


# ---------- invalid requests ----------


@pytest.mark.parametrize(
    "call",
    [
        lambda: airflow_tools.get_dag_runs("../../admin"),
        lambda: airflow_tools.get_dag_runs("training_pipeline", state="exploded"),
        lambda: airflow_tools.get_dag_runs("training_pipeline", limit=0),
        lambda: airflow_tools.get_dag_run("training_pipeline", "run id with spaces"),
        lambda: airflow_tools.get_task_logs("training_pipeline", FAILED_RUN, "a/b"),
        lambda: airflow_tools.trigger_dag("training_pipeline", conf=["x"], confirm=True),
    ],
)
def test_invalid_arguments_rejected_before_any_http_call(airflow, call):
    result = call()
    assert result.status == ToolStatus.INVALID_REQUEST
    assert airflow.calls() == []


def test_unknown_dag_is_invalid_request(airflow):
    result = airflow_tools.get_dag_runs("no_such_dag")
    assert result.status == ToolStatus.INVALID_REQUEST
    assert result.data is None


# ---------- service failures: no invented results ----------


def test_bad_credentials_fail_without_leaking_password(airflow, monkeypatch, caplog):
    monkeypatch.setenv("AIRFLOW_PASSWORD", "wrong-password")
    caplog.set_level(logging.INFO, logger="mlops_agent.tools")

    result = airflow_tools.list_dags()

    assert result.status == ToolStatus.ERROR
    assert result.data is None
    assert "authentication failed" in result.error
    assert "wrong-password" not in result.error + caplog.text


def test_timeout_returns_timeout_status(airflow, monkeypatch):
    monkeypatch.setenv("AIRFLOW_TIMEOUT_S", "0.2")
    airflow.state["delay"] = 1.0

    result = airflow_tools.get_dag_runs("training_pipeline")

    assert result.status == ToolStatus.TIMEOUT
    assert result.data is None


def test_unreachable_airflow_returns_error_without_data(monkeypatch):
    with socket.socket() as s:  # find a port with nothing listening
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("AIRFLOW_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("AIRFLOW_USERNAME", USERNAME)

    result = airflow_tools.get_failed_tasks("training_pipeline")

    assert result.status == ToolStatus.ERROR
    assert result.data is None
    assert "unreachable" in result.error
