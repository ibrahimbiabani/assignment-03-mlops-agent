"""A minimal in-process fake of the Airflow 3 REST API, served over real HTTP.

Response shapes follow the Airflow 3 OpenAPI spec (/api/v2). Only the endpoints
used by agent.tools.airflow_tools are implemented.
"""

import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

USERNAME, PASSWORD, TOKEN = "admin", "s3cret-pw", "jwt-token-123"
FAILED_RUN = "manual__2026-11-09T10:00:00+00:00"
OK_RUN = "scheduled__2026-11-08T00:00:00+00:00"


def _ti(task_id: str, state: str, try_number: int = 1, max_tries: int = 2) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "state": state,
        "try_number": try_number,
        "max_tries": max_tries,
        "start_date": "2026-11-09T10:00:05Z",
        "end_date": "2026-11-09T10:00:09Z",
        "duration": 4.0,
        "operator_name": "PythonOperator",
        "state_reason": None,
        "map_index": -1,
    }


def default_state() -> dict[str, Any]:
    dag = {
        "dag_id": "training_pipeline",
        "is_paused": False,
        "is_stale": False,
        "has_import_errors": False,
        "timetable_summary": None,
        "next_dagrun_run_after": None,
        "owners": ["mlops"],
        "tags": [{"name": "ml", "dag_id": "training_pipeline"}],
    }
    run = {
        "dag_id": "training_pipeline",
        "run_type": "manual",
        "triggered_by": "rest_api",
        "logical_date": None,
        "run_after": "2026-11-09T10:00:00Z",
        "conf": {},
        "note": None,
    }
    return {
        "dags": {
            "training_pipeline": dag,
            "data_backup": {**dag, "dag_id": "data_backup", "is_paused": True, "tags": []},
        },
        "runs": {  # newest first
            "training_pipeline": [
                {**run, "dag_run_id": FAILED_RUN, "state": "failed", "duration": 9.0},
                {**run, "dag_run_id": OK_RUN, "state": "success", "run_type": "scheduled"},
            ]
        },
        "tis": {
            FAILED_RUN: [
                _ti("validate_dataset", "success"),
                _ti("preprocess_data", "success"),
                _ti("train_model", "failed", try_number=3),
                _ti("evaluate_model", "upstream_failed", try_number=0),
                _ti("quality_gate", "upstream_failed", try_number=0),
            ],
            OK_RUN: [_ti("validate_dataset", "success"), _ti("train_model", "success")],
        },
        "logs": {
            (FAILED_RUN, "train_model", 3): [
                {"event": "Loading features from data/processed/train.csv"},
                {"event": "Connecting to MLflow with password=hunter2"},
                {"event": "ERROR - ValueError: Input X contains NaN."},
                {"event": "Task failed with exception"},
            ]
        },
        "valid_tokens": {TOKEN},
        "delay": 0.0,
        "requests": [],
    }


class FakeAirflow:
    def __init__(self) -> None:
        self.state = default_state()
        handler = type("Handler", (_Handler,), {"fake": self})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def calls(self, method: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.state["requests"] if method in (None, r["method"])]


class _Handler(BaseHTTPRequestHandler):
    fake: FakeAirflow

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass

    def _reply(self, code: int, body: Any) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _dispatch(self, method: str) -> None:
        s = self.fake.state
        url = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        s["requests"].append(
            {
                "method": method,
                "path": url.path,
                "query": query,
                "body": body,
                "auth": self.headers.get("Authorization"),
            }
        )
        if s["delay"]:
            time.sleep(s["delay"])

        if url.path == "/auth/token" and method == "POST":
            if body == {"username": USERNAME, "password": PASSWORD}:
                return self._reply(201, {"access_token": TOKEN})
            return self._reply(401, {"detail": "Invalid credentials"})

        parts = [unquote(p) for p in url.path.split("/") if p]
        if parts[:2] != ["api", "v2"]:
            return self._reply(404, {"detail": "Not Found"})
        parts = parts[2:]
        if parts == ["monitor", "health"]:
            return self._reply(
                200,
                {
                    "metadatabase": {"status": "healthy"},
                    "scheduler": {
                        "status": "healthy",
                        "latest_scheduler_heartbeat": "2026-11-09T10:05:00Z",
                    },
                    "triggerer": {"status": None, "latest_triggerer_heartbeat": None},
                    "dag_processor": None,
                },
            )
        auth = self.headers.get("Authorization", "")
        if auth.removeprefix("Bearer ") not in s["valid_tokens"]:
            return self._reply(401, {"detail": "Not authenticated"})
        return self._route(method, parts, query, body)

    def _route(self, method: str, parts: list[str], query: dict, body: Any) -> None:
        s = self.fake.state
        if parts == ["dags"]:
            dags = list(s["dags"].values())
            return self._reply(200, {"dags": dags, "total_entries": len(dags)})
        if len(parts) < 2 or parts[0] != "dags" or parts[1] not in s["dags"]:
            return self._reply(404, {"detail": "DAG not found"})
        dag_id = parts[1]
        if len(parts) == 2:
            return self._reply(200, s["dags"][dag_id])
        runs = s["runs"].setdefault(dag_id, [])
        if parts[2:] == ["dagRuns"] and method == "GET":
            found = [r for r in runs if query.get("state") in (None, r["state"])]
            found = found[: int(query.get("limit", 50))]
            return self._reply(200, {"dag_runs": found, "total_entries": len(found)})
        if parts[2:] == ["dagRuns"] and method == "POST":
            if not isinstance(body, dict) or "logical_date" not in body:
                return self._reply(422, {"detail": "logical_date: Field required"})
            run = {
                "dag_run_id": "manual__2026-11-09T12:00:00+00:00",
                "dag_id": dag_id,
                "state": "queued",
                "run_type": "manual",
                "triggered_by": "rest_api",
                "logical_date": body["logical_date"],
                "conf": body.get("conf"),
                "note": body.get("note"),
            }
            runs.insert(0, run)
            return self._reply(200, run)
        run = next((r for r in runs if len(parts) > 3 and r["dag_run_id"] == parts[3]), None)
        if run is None:
            return self._reply(404, {"detail": "DagRun not found"})
        rest = parts[4:]
        tis = s["tis"].get(run["dag_run_id"], [])
        if not rest:
            return self._reply(200, run)
        if rest == ["taskInstances"]:
            return self._reply(200, {"task_instances": copy.deepcopy(tis)})
        ti = next((t for t in tis if len(rest) > 1 and t["task_id"] == rest[1]), None)
        if ti is None:
            return self._reply(404, {"detail": "TaskInstance not found"})
        if len(rest) == 2:
            return self._reply(200, ti)
        if len(rest) == 4 and rest[2] == "logs":
            content = s["logs"].get((run["dag_run_id"], ti["task_id"], int(rest[3])), [])
            return self._reply(200, {"content": content, "continuation_token": None})
        return self._reply(404, {"detail": "Not Found"})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")
