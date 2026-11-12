"""Apache Airflow workflow tools (Task 3), via the Airflow 3 REST API (/api/v2).

Authentication: a JWT is obtained from POST /auth/token with AIRFLOW_USERNAME /
AIRFLOW_PASSWORD and sent as a Bearer token. Credentials and tokens are never
logged or returned.

Only `airflow_trigger_dag` changes state. It requires confirm=True and a DAG id
listed in AIRFLOW_ALLOWED_TRIGGER_DAGS (default: training_pipeline).
"""

import json
import os
import re
from collections import Counter
from typing import Any
from urllib.parse import quote

import requests

from agent.tools.base import (
    ConfirmationRequired,
    InvalidToolRequest,
    ToolError,
    ToolRejected,
    ToolTimeout,
    request_id_var,
    tool,
)

DAG_RUN_STATES = frozenset({"queued", "running", "success", "failed"})
FAILED_TASK_STATES = frozenset({"failed", "upstream_failed", "up_for_retry"})
MAX_CONF_BYTES = 10_000

_DAG_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,250}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,250}$")
_ERROR_LINE_RE = re.compile(r"error|exception|traceback|failed", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization)\b(\s*[=:]\s*)(\S+)"
)


class AirflowClient:
    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self.timeout = timeout
        self._token: str | None = None
        self._session = requests.Session()

    def _send(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            return self._session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise ToolTimeout(f"Airflow did not respond within {self.timeout}s") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ToolError(f"Airflow API is unreachable at {self.base_url}") from exc

    def _authenticate(self) -> None:
        resp = self._send(
            "POST",
            f"{self.base_url}/auth/token",
            json={"username": self._username, "password": self._password},
        )
        if resp.status_code == 404:
            raise ToolError(
                f"{self.base_url} is not an Airflow 3 API server (POST /auth/token returned 404); "
                "check AIRFLOW_BASE_URL and that Airflow's port is actually published"
            )
        if resp.status_code not in (200, 201):
            raise ToolError(f"Airflow authentication failed (HTTP {resp.status_code})")
        self._token = resp.json().get("access_token")
        if not self._token:
            raise ToolError("Airflow authentication response contained no access_token")

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Call /api/v2{path}; re-authenticate once if the token has expired."""
        url = f"{self.base_url}/api/v2{path}"
        for attempt in range(2):
            headers = {"Accept": "application/json"}
            if self._username:
                if self._token is None:
                    self._authenticate()
                headers["Authorization"] = f"Bearer {self._token}"
            resp = self._send(method, url, headers=headers, **kwargs)
            if resp.status_code == 401 and self._username and attempt == 0:
                self._token = None
                continue
            break
        return _handle(resp, path)


def _handle(resp: requests.Response, path: str) -> Any:
    if resp.status_code < 400:
        return resp.json() if resp.content else {}
    try:
        detail = str(resp.json().get("detail", ""))[:300]
    except ValueError:
        detail = resp.text[:300]
    code = resp.status_code
    if code == 401:
        raise ToolError("Airflow rejected the credentials (HTTP 401)")
    if code == 403:
        raise ToolRejected(f"Airflow denied permission for {path} (HTTP 403)")
    if code == 404:
        raise InvalidToolRequest(f"not found in Airflow: {path}")
    if code in (400, 409, 422):
        raise InvalidToolRequest(f"Airflow rejected the request (HTTP {code}): {detail}")
    raise ToolError(f"Airflow API error (HTTP {code}): {detail}")


def get_client() -> AirflowClient:
    return AirflowClient(
        base_url=os.environ.get("AIRFLOW_BASE_URL", "http://localhost:8080"),
        username=os.environ.get("AIRFLOW_USERNAME"),
        password=os.environ.get("AIRFLOW_PASSWORD"),
        timeout=float(os.environ.get("AIRFLOW_TIMEOUT_S", "10")),
    )


def allowed_trigger_dags() -> frozenset[str]:
    raw = os.environ.get("AIRFLOW_ALLOWED_TRIGGER_DAGS", "training_pipeline")
    return frozenset(d.strip() for d in raw.split(",") if d.strip())


# ---------- validation / helpers ----------


def _ident(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _DAG_ID_RE.match(value):
        raise InvalidToolRequest(f"invalid {kind}: {value!r}")
    return value


def _dag(dag_id: str) -> str:
    return _ident(dag_id, "dag_id")


def _run(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise InvalidToolRequest(f"invalid dag_run_id: {run_id!r}")
    return quote(run_id, safe="")


def _limit(n: int) -> int:
    if not isinstance(n, int) or not 1 <= n <= 100:
        raise InvalidToolRequest(f"limit must be an integer between 1 and 100, got {n!r}")
    return n


def _redact(line: str) -> str:
    return _SECRET_RE.sub(r"\1\2***", line)


def _run_summary(r: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dag_run_id",
        "state",
        "run_type",
        "triggered_by",
        "logical_date",
        "run_after",
        "start_date",
        "end_date",
        "duration",
        "conf",
        "note",
    )
    return {k: r.get(k) for k in keys}


def _ti_summary(ti: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "state",
        "try_number",
        "max_tries",
        "start_date",
        "end_date",
        "duration",
        "operator_name",
        "state_reason",
    )
    return {k: ti.get(k) for k in keys}


def _task_instances(client: AirflowClient, dag_id: str, run_path: str) -> list[dict[str, Any]]:
    data = client.request("GET", f"/dags/{dag_id}/dagRuns/{run_path}/taskInstances")
    return data.get("task_instances", [])


# ---------- tools ----------


@tool("airflow_health")
def health() -> dict[str, Any]:
    """Status of the Airflow metadatabase, scheduler, triggerer and DAG processor."""
    h = get_client().request("GET", "/monitor/health")

    def status(component: str) -> dict[str, Any] | None:
        c = h.get(component)
        if c is None:
            return None
        hb = next((v for k, v in c.items() if k.startswith("latest_") and "heartbeat" in k), None)
        return {"status": c.get("status"), "latest_heartbeat": hb}

    return {c: status(c) for c in ("metadatabase", "scheduler", "triggerer", "dag_processor")}


@tool("airflow_list_dags")
def list_dags(limit: int = 50) -> dict[str, Any]:
    """List DAGs with pause state, schedule, tags and import-error flag."""
    data = get_client().request("GET", "/dags", params={"limit": _limit(limit)})
    return {
        "total_entries": data.get("total_entries"),
        "dags": [
            {
                "dag_id": d.get("dag_id"),
                "is_paused": d.get("is_paused"),
                "is_stale": d.get("is_stale"),
                "has_import_errors": d.get("has_import_errors"),
                "schedule": d.get("timetable_summary"),
                "next_run_after": d.get("next_dagrun_run_after"),
                "owners": d.get("owners"),
                "tags": [t.get("name") for t in d.get("tags") or []],
            }
            for d in data.get("dags", [])
        ],
    }


@tool("airflow_get_dag_runs")
def get_dag_runs(dag_id: str, limit: int = 10, state: str | None = None) -> dict[str, Any]:
    """Most recent runs of a DAG, newest first, optionally filtered by state."""
    params: dict[str, Any] = {"limit": _limit(limit), "order_by": "-run_after"}
    if state is not None:
        if state not in DAG_RUN_STATES:
            raise InvalidToolRequest(f"state must be one of {sorted(DAG_RUN_STATES)}")
        params["state"] = state
    data = get_client().request("GET", f"/dags/{_dag(dag_id)}/dagRuns", params=params)
    runs = [_run_summary(r) for r in data.get("dag_runs", [])]
    return {"dag_id": dag_id, "runs": runs, "count": len(runs)}


@tool("airflow_get_dag_run")
def get_dag_run(dag_id: str, dag_run_id: str) -> dict[str, Any]:
    """Status of one DAG run with the state of each of its tasks."""
    client = get_client()
    run_path = _run(dag_run_id)
    run = client.request("GET", f"/dags/{_dag(dag_id)}/dagRuns/{run_path}")
    tis = _task_instances(client, dag_id, run_path)
    return {
        "dag_id": dag_id,
        "run": _run_summary(run),
        "task_state_counts": dict(Counter(str(ti.get("state")) for ti in tis)),
        "tasks": [_ti_summary(ti) for ti in tis],
    }


@tool("airflow_get_failed_tasks")
def get_failed_tasks(dag_id: str, dag_run_id: str | None = None) -> dict[str, Any]:
    """Failed tasks of a run (default: the most recent failed run).

    `root_cause_tasks` are those that failed themselves; tasks in
    `upstream_failed` only failed because something before them did.
    """
    client = get_client()
    _dag(dag_id)
    if dag_run_id is None:
        latest = client.request(
            "GET",
            f"/dags/{dag_id}/dagRuns",
            params={"state": "failed", "order_by": "-run_after", "limit": 1},
        ).get("dag_runs", [])
        if not latest:
            return {
                "dag_id": dag_id,
                "dag_run_id": None,
                "failed_tasks": [],
                "note": "no failed runs",
            }
        dag_run_id = latest[0]["dag_run_id"]
    failed = [
        _ti_summary(ti)
        for ti in _task_instances(client, dag_id, _run(dag_run_id))
        if ti.get("state") in FAILED_TASK_STATES
    ]
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "failed_tasks": failed,
        "root_cause_tasks": [t["task_id"] for t in failed if t["state"] != "upstream_failed"],
    }


@tool("airflow_get_task_logs")
def get_task_logs(
    dag_id: str,
    dag_run_id: str,
    task_id: str,
    try_number: int | None = None,
    max_lines: int = 200,
) -> dict[str, Any]:
    """Tail of a task attempt's log, its error lines and raised exceptions (secrets redacted).

    Airflow 3 returns structured messages ({event, level, error_detail, ...});
    `exceptions` is taken from `error_detail`, which holds the actual failure reason.
    """
    client = get_client()
    task = _ident(task_id, "task_id")
    base = f"/dags/{_dag(dag_id)}/dagRuns/{_run(dag_run_id)}/taskInstances/{task}"
    if not isinstance(max_lines, int) or not 1 <= max_lines <= 2000:
        raise InvalidToolRequest("max_lines must be an integer between 1 and 2000")
    if try_number is None:
        try_number = client.request("GET", base).get("try_number") or 1
    elif not isinstance(try_number, int) or try_number < 0:
        raise InvalidToolRequest(f"invalid try_number: {try_number!r}")
    data = client.request("GET", f"{base}/logs/{try_number}", params={"full_content": "true"})

    content = data.get("content", [])
    if isinstance(content, str):
        content = content.splitlines()
    lines, error_lines, exceptions = [], [], []
    for c in content:
        msg = c if isinstance(c, dict) else {"event": str(c)}
        level = str(msg.get("level") or "").lower()
        line = _redact(f"[{level}] {msg.get('event', '')}" if level else str(msg.get("event", "")))
        line = line[:1000]
        lines.append(line)
        if level in ("error", "critical") or _ERROR_LINE_RE.search(line):
            error_lines.append(line)
        for exc in msg.get("error_detail") or []:
            frames = exc.get("frames") or []
            last = frames[-1] if frames else {}
            exceptions.append(
                {
                    "type": exc.get("exc_type"),
                    "message": _redact(str(exc.get("exc_value", "")))[:2000],
                    "location": (
                        f"{last.get('filename')}:{last.get('lineno')} in {last.get('name')}"
                        if last
                        else None
                    ),
                }
            )
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "task_id": task_id,
        "try_number": try_number,
        "total_lines": len(lines),
        "exceptions": exceptions[-10:],
        "error_lines": error_lines[-50:],
        "tail": lines[-max_lines:],
    }


@tool("airflow_trigger_dag")
def trigger_dag(
    dag_id: str, conf: dict[str, Any] | None = None, confirm: bool = False
) -> dict[str, Any]:
    """Trigger an approved DAG. Requires confirm=True from the user."""
    _dag(dag_id)
    if dag_id not in allowed_trigger_dags():
        raise ToolRejected(f"DAG '{dag_id}' is not on the approved trigger list")
    conf = conf or {}
    if not isinstance(conf, dict) or len(json.dumps(conf)) > MAX_CONF_BYTES:
        raise InvalidToolRequest(f"conf must be a JSON object under {MAX_CONF_BYTES} bytes")
    # Last check, so a dry run (confirm=False) validates everything else first.
    if confirm is not True:
        raise ConfirmationRequired(f"triggering '{dag_id}' requires explicit user confirmation")

    client = get_client()
    dag = client.request("GET", f"/dags/{dag_id}")
    run = client.request(
        "POST",
        f"/dags/{dag_id}/dagRuns",
        json={
            "logical_date": None,
            "conf": conf,
            "note": f"Triggered by MLOps agent (request {request_id_var.get()})",
        },
    )
    result = {"dag_id": dag_id, "run": _run_summary(run)}
    if dag.get("is_paused"):
        result["warning"] = "DAG is paused; the run will stay queued until it is unpaused."
    return result
