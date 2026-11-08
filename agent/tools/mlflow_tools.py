"""MLflow experiment-intelligence tools (Task 2).

Uses MlflowClient only; nothing is scraped from the MLflow UI. All tools are read-only.

Run tags written by the training pipeline and read here:
  data.version, data.git_commit, data.dvc_path, data.dvc_md5  -> dataset provenance (DVC)
  quality_gate                                                -> "passed" / "failed"
"""

import functools
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import requests
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from agent.schemas import ToolResult
from agent.tools.base import InvalidToolRequest, ToolError, ToolTimeout, tool

PROVENANCE_TAGS = ("data.version", "data.git_commit", "data.dvc_path", "data.dvc_md5")
QUALITY_GATE_TAG = "quality_gate"
RUN_STATUSES = frozenset({"RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"})
SMALL_MARGIN = 0.01

_METRIC_RE = re.compile(r"^[\w.\-/ ]{1,250}$")
_MODEL_RE = re.compile(r"^[\w.\- ]{1,200}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")
_NOT_FOUND_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def get_client() -> MlflowClient:
    # Fail fast instead of MLflow's default of 7 retries with long backoff.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))


@contextmanager
def _mlflow_errors() -> Iterator[None]:
    """Translate MLflow/HTTP exceptions into the agent's tool error types."""
    try:
        yield
    except ToolError:
        raise
    except requests.exceptions.Timeout as exc:
        raise ToolTimeout("MLflow tracking server did not respond in time") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ToolError("MLflow tracking server is unreachable") from exc
    except MlflowException as exc:
        msg = str(exc)[:500]
        if exc.error_code in _NOT_FOUND_CODES:
            raise InvalidToolRequest(msg) from exc
        if "timed out" in msg.lower():
            raise ToolTimeout(f"MLflow request timed out: {msg}") from exc
        raise ToolError(f"MLflow request failed: {msg}") from exc


def mlflow_tool(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., ToolResult]]:
    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., ToolResult]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> dict[str, Any]:
            with _mlflow_errors():
                return fn(*args, **kwargs)

        return tool(name)(inner)

    return decorator


# ---------- validation ----------


def _check_metric(metric: str) -> str:
    if not isinstance(metric, str) or not _METRIC_RE.match(metric):
        raise InvalidToolRequest(f"invalid metric name: {metric!r}")
    return metric


def _check_mode(mode: str) -> str:
    if mode not in ("max", "min"):
        raise InvalidToolRequest(f"mode must be 'max' or 'min', got {mode!r}")
    return mode


def _check_limit(n: int, upper: int = 200) -> int:
    if not isinstance(n, int) or not 1 <= n <= upper:
        raise InvalidToolRequest(f"limit must be an integer between 1 and {upper}, got {n!r}")
    return n


# ---------- helpers ----------


def _ts(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat() if ms else None


def _summary(run: Any, include_params: bool = True) -> dict[str, Any]:
    tags = run.data.tags
    out = {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": _ts(run.info.start_time),
        "end_time": _ts(run.info.end_time),
        "metrics": dict(run.data.metrics),
        "provenance": {k: tags[k] for k in PROVENANCE_TAGS if k in tags},
        "quality_gate": tags.get(QUALITY_GATE_TAG),
    }
    if include_params:
        out["params"] = dict(run.data.params)
    return out


def _experiment_ids(client: MlflowClient, names: list[str] | None) -> list[str]:
    if not names:
        return [e.experiment_id for e in client.search_experiments()]
    ids = []
    for name in names:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise InvalidToolRequest(f"experiment not found: {name!r}")
        ids.append(exp.experiment_id)
    return ids


def _recent_finished(client: MlflowClient, exp_ids: list[str], n: int) -> list[Any]:
    if not exp_ids:
        return []
    return client.search_runs(
        exp_ids,
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=n,
    )


def _rank(runs: list[Any], metric: str, mode: str) -> list[tuple[Any, float]]:
    scored = [(r, r.data.metrics[metric]) for r in runs if metric in r.data.metrics]
    return sorted(scored, key=lambda rv: rv[1], reverse=(mode == "max"))


def _caveats(metric: str, ranked: list[tuple[Any, float]]) -> list[str]:
    """Evidence-based reasons the top-scoring run may not be ready for production."""
    best_run, best_val = ranked[0]
    notes = [
        f"Selected on '{metric}' alone; the top score does not by itself make a run "
        "production-ready."
    ]
    gate = best_run.data.tags.get(QUALITY_GATE_TAG)
    if gate is None:
        notes.append("No quality-gate result is recorded for this run.")
    elif gate != "passed":
        notes.append(f"Quality gate recorded as '{gate}'.")
    if not any(k in best_run.data.tags for k in PROVENANCE_TAGS):
        notes.append("No dataset provenance tags recorded; the data version cannot be confirmed.")
    if len(ranked) == 1:
        notes.append("Only one run logged this metric, so there was nothing to compare against.")
    elif abs(best_val - ranked[1][1]) < SMALL_MARGIN:
        notes.append(f"Margin over the runner-up is only {abs(best_val - ranked[1][1]):.4f}.")
    return notes


def _brief(run: Any, value: float) -> dict[str, Any]:
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "value": value,
        "quality_gate": run.data.tags.get(QUALITY_GATE_TAG),
    }


# ---------- tools ----------


@mlflow_tool("mlflow_list_experiments")
def list_experiments(max_results: int = 50) -> dict[str, Any]:
    """List active experiments."""
    exps = get_client().search_experiments(max_results=_check_limit(max_results))
    return {
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
                "last_update_time": _ts(e.last_update_time),
            }
            for e in exps
        ]
    }


@mlflow_tool("mlflow_list_runs")
def list_runs(
    experiment_names: list[str] | None = None, max_results: int = 10, status: str | None = None
) -> dict[str, Any]:
    """List the most recent runs (any status unless `status` is given), newest first."""
    _check_limit(max_results)
    if status is not None and status not in RUN_STATUSES:
        raise InvalidToolRequest(f"status must be one of {sorted(RUN_STATUSES)}")
    client = get_client()
    ids = _experiment_ids(client, experiment_names)
    runs = (
        client.search_runs(
            ids,
            filter_string=f"attributes.status = '{status}'" if status else "",
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )
        if ids
        else []
    )
    return {"runs": [_summary(r) for r in runs], "count": len(runs)}


@mlflow_tool("mlflow_get_run")
def get_run(run_id: str) -> dict[str, Any]:
    """Metrics, parameters, user tags and top-level artifacts of one run."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise InvalidToolRequest(f"invalid run id: {run_id!r}")
    client = get_client()
    run = client.get_run(run_id)
    out = _summary(run)
    out["tags"] = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    out["artifacts"] = [
        {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
        for a in client.list_artifacts(run_id)
    ]
    return out


@mlflow_tool("mlflow_find_best_run")
def find_best_run(
    metric: str,
    mode: str = "max",
    experiment_names: list[str] | None = None,
    max_runs: int = 20,
) -> dict[str, Any]:
    """Best of the `max_runs` most recent FINISHED runs by `metric`, with caveats."""
    _check_metric(metric)
    _check_mode(mode)
    _check_limit(max_runs)
    client = get_client()
    runs = _recent_finished(client, _experiment_ids(client, experiment_names), max_runs)
    ranked = _rank(runs, metric, mode)
    result: dict[str, Any] = {
        "metric": metric,
        "mode": mode,
        "candidates_considered": len(runs),
        "runs_with_metric": len(ranked),
    }
    if not ranked:
        result["best"] = None
        result["reason"] = f"None of the {len(runs)} most recent finished runs logged '{metric}'."
        return result
    best_run, best_val = ranked[0]
    result["best"] = {**_summary(best_run), "value": best_val}
    result["ranking"] = [_brief(r, v) for r, v in ranked[:5]]
    result["caveats"] = _caveats(metric, ranked)
    return result


@mlflow_tool("mlflow_compare_runs")
def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Side-by-side metrics and parameters for 2-10 runs."""
    if not isinstance(run_ids, list) or not 2 <= len(run_ids) <= 10:
        raise InvalidToolRequest("provide between 2 and 10 run ids")
    for rid in run_ids:
        if not isinstance(rid, str) or not _RUN_ID_RE.match(rid):
            raise InvalidToolRequest(f"invalid run id: {rid!r}")
    client = get_client()
    runs = [client.get_run(rid) for rid in run_ids]
    metrics = sorted({m for r in runs for m in r.data.metrics})
    return {
        "runs": [_summary(r) for r in runs],
        "metric_table": {m: {r.info.run_id: r.data.metrics.get(m) for r in runs} for m in metrics},
    }


@mlflow_tool("mlflow_compare_experiments")
def compare_experiments(
    experiment_names: list[str], metric: str, mode: str = "max", max_runs: int = 20
) -> dict[str, Any]:
    """Best recent run of each experiment by `metric`."""
    if not isinstance(experiment_names, list) or len(experiment_names) < 2:
        raise InvalidToolRequest("provide at least two experiment names")
    _check_metric(metric)
    _check_mode(mode)
    _check_limit(max_runs)
    client = get_client()
    rows, leaders = [], []
    for name, exp_id in zip(
        experiment_names, _experiment_ids(client, experiment_names), strict=True
    ):
        runs = _recent_finished(client, [exp_id], max_runs)
        ranked = _rank(runs, metric, mode)
        rows.append(
            {
                "experiment": name,
                "finished_runs": len(runs),
                "runs_with_metric": len(ranked),
                "best": _brief(*ranked[0]) if ranked else None,
            }
        )
        if ranked:
            leaders.append((name, ranked[0][1]))
    leaders.sort(key=lambda nv: nv[1], reverse=(mode == "max"))
    return {
        "metric": metric,
        "mode": mode,
        "experiments": rows,
        "leading_experiment": leaders[0][0] if leaders else None,
    }


@mlflow_tool("mlflow_get_model_versions")
def get_model_versions(model_name: str | None = None, max_results: int = 20) -> dict[str, Any]:
    """Registered models, or the versions (with aliases and source run) of one model."""
    _check_limit(max_results)
    client = get_client()
    if model_name is None:
        models = client.search_registered_models(max_results=max_results)
        return {
            "registered_models": [
                {"name": m.name, "aliases": dict(m.aliases), "description": m.description}
                for m in models
            ]
        }
    if not isinstance(model_name, str) or not _MODEL_RE.match(model_name):
        raise InvalidToolRequest(f"invalid model name: {model_name!r}")
    model = client.get_registered_model(model_name)
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in model.aliases.items():
        aliases_by_version.setdefault(str(version), []).append(alias)
    versions = client.search_model_versions(
        f"name='{model_name}'", max_results=max_results, order_by=["version_number DESC"]
    )
    return {
        "model_name": model_name,
        "versions": [
            {
                "version": str(v.version),
                "run_id": v.run_id,
                "status": v.status,
                "aliases": sorted(aliases_by_version.get(str(v.version), [])),
                "tags": dict(v.tags),
                "created": _ts(v.creation_timestamp),
            }
            for v in versions
        ],
    }
