"""MLflow tool tests.

Most tests run against a real, temporary SQLite-backed MLflow store so the
client calls (filters, ordering, registry) are exercised for real. Service
failures are simulated with a fake client.
"""

from types import SimpleNamespace

import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from agent.schemas import ToolStatus
from agent.tools import mlflow_tools

T0 = 1_793_000_000_000  # fixed epoch ms so run ordering is deterministic


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow")
    client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    art = (root / "artifacts").as_uri()
    base = client.create_experiment("iris-baseline", artifact_location=art)
    tuned = client.create_experiment("iris-tuned", artifact_location=art)
    ids = {}

    def run(key, exp, minute, metrics, tags=None, status="FINISHED"):
        r = client.create_run(exp, start_time=T0 + minute * 60_000, run_name=key, tags=tags or {})
        for k, v in metrics.items():
            client.log_metric(r.info.run_id, k, v)
        client.log_param(r.info.run_id, "n_estimators", 100 + minute)
        client.set_terminated(r.info.run_id, status=status, end_time=T0 + minute * 60_000 + 1)
        ids[key] = r.info.run_id

    provenance_v1 = {"data.version": "data-v1.0", "data.git_commit": "1111111"}
    provenance_v2 = {"data.version": "data-v2.1", "data.git_commit": "a4c92e1"}
    run("old", base, 0, {"f1": 0.80, "log_loss": 0.50}, {"quality_gate": "passed", **provenance_v1})
    run(
        "best", base, 1, {"f1": 0.91, "log_loss": 0.30}, {"quality_gate": "passed", **provenance_v2}
    )
    run("failed", base, 2, {"f1": 0.99}, status="FAILED")
    run("no_f1", base, 3, {"accuracy": 0.90})
    run("tuned", tuned, 4, {"f1": 0.905, "log_loss": 0.28}, {"quality_gate": "failed"})
    client.log_text(ids["best"], "model card", "model/README.txt")

    client.create_registered_model("iris-classifier")
    for key in ("old", "best"):
        client.create_model_version(
            "iris-classifier", source=f"runs:/{ids[key]}/model", run_id=ids[key]
        )
    client.set_registered_model_alias("iris-classifier", "champion", "2")
    return client, ids


@pytest.fixture
def mlflow(store, monkeypatch):
    client, ids = store
    monkeypatch.setattr(mlflow_tools, "get_client", lambda: client)
    return ids


# ---------- successful tool execution ----------


def test_list_experiments(mlflow):
    result = mlflow_tools.list_experiments()
    assert result.ok
    names = {e["name"] for e in result.data["experiments"]}
    assert {"iris-baseline", "iris-tuned"} <= names


def test_list_runs_newest_first_and_status_filter(mlflow):
    result = mlflow_tools.list_runs(["iris-baseline"])
    assert [r["run_name"] for r in result.data["runs"]] == ["no_f1", "failed", "best", "old"]

    failed = mlflow_tools.list_runs(["iris-baseline"], status="FAILED")
    assert [r["run_name"] for r in failed.data["runs"]] == ["failed"]


def test_get_run_returns_metrics_params_tags_and_artifacts(mlflow):
    result = mlflow_tools.get_run(mlflow["best"])
    assert result.ok
    assert result.data["metrics"]["f1"] == pytest.approx(0.91)
    assert result.data["params"]["n_estimators"] == "101"
    assert result.data["provenance"] == {"data.version": "data-v2.1", "data.git_commit": "a4c92e1"}
    assert not any(k.startswith("mlflow.") for k in result.data["tags"])
    assert {"path": "model", "is_dir": True, "file_size": None} in result.data["artifacts"]


def test_find_best_run_ignores_failed_runs_and_runs_without_metric(mlflow):
    result = mlflow_tools.find_best_run("f1", experiment_names=["iris-baseline"])

    assert result.ok
    best = result.data["best"]
    assert best["run_name"] == "best"  # not the FAILED run with f1=0.99
    assert best["provenance"]["data.version"] == "data-v2.1"
    assert result.data["candidates_considered"] == 3
    assert result.data["runs_with_metric"] == 2
    assert [r["run_name"] for r in result.data["ranking"]] == ["best", "old"]
    # Even a clean winner carries the "not automatically production-ready" caveat.
    assert len(result.data["caveats"]) == 1
    assert "production-ready" in result.data["caveats"][0]


def test_find_best_run_flags_small_margin(mlflow):
    result = mlflow_tools.find_best_run("f1")
    assert result.data["best"]["run_name"] == "best"
    assert result.data["ranking"][1]["run_name"] == "tuned"
    assert any("Margin over the runner-up" in c for c in result.data["caveats"])


def test_find_best_run_min_mode_reports_failed_gate_and_missing_provenance(mlflow):
    result = mlflow_tools.find_best_run("log_loss", mode="min")
    assert result.data["best"]["run_name"] == "tuned"
    caveats = " ".join(result.data["caveats"])
    assert "Quality gate recorded as 'failed'" in caveats
    assert "No dataset provenance" in caveats


def test_find_best_run_with_unlogged_metric_returns_no_candidate(mlflow):
    result = mlflow_tools.find_best_run("auc")
    assert result.ok
    assert result.data["best"] is None
    assert "auc" in result.data["reason"]


def test_compare_runs_builds_metric_table(mlflow):
    result = mlflow_tools.compare_runs([mlflow["old"], mlflow["best"]])
    table = result.data["metric_table"]
    assert table["f1"] == {mlflow["old"]: pytest.approx(0.80), mlflow["best"]: pytest.approx(0.91)}


def test_compare_experiments(mlflow):
    result = mlflow_tools.compare_experiments(["iris-baseline", "iris-tuned"], "f1")
    assert result.data["leading_experiment"] == "iris-baseline"
    tuned = result.data["experiments"][1]
    assert tuned["best"]["quality_gate"] == "failed"


def test_model_versions_include_aliases_and_source_run(mlflow):
    result = mlflow_tools.get_model_versions("iris-classifier")
    v2, v1 = result.data["versions"]
    assert (v2["version"], v2["aliases"], v2["run_id"]) == ("2", ["champion"], mlflow["best"])
    assert v1["aliases"] == []

    models = mlflow_tools.get_model_versions()
    assert models.data["registered_models"][0]["aliases"] == {"champion": 2}


# ---------- invalid requests ----------


@pytest.mark.parametrize(
    "call",
    [
        lambda: mlflow_tools.list_runs(["no-such-experiment"]),
        lambda: mlflow_tools.get_run("0" * 32),
        lambda: mlflow_tools.get_model_versions("no-such-model"),
        lambda: mlflow_tools.get_model_versions("x' OR name LIKE '%"),
        lambda: mlflow_tools.find_best_run("f1` DESC, metrics.`x"),
        lambda: mlflow_tools.find_best_run("f1", mode="highest"),
        lambda: mlflow_tools.find_best_run("f1", max_runs=0),
        lambda: mlflow_tools.compare_runs(["only-one"]),
        lambda: mlflow_tools.compare_experiments(["iris-baseline"], "f1"),
        lambda: mlflow_tools.list_runs(status="DONE"),
    ],
)
def test_invalid_requests_are_rejected(mlflow, call):
    result = call()
    assert result.status == ToolStatus.INVALID_REQUEST
    assert result.data is None


# ---------- service failures: no invented results ----------


def _failing_client(exc):
    def boom(*args, **kwargs):
        raise exc

    return SimpleNamespace(search_experiments=boom, get_experiment_by_name=boom, search_runs=boom)


def test_server_error_returns_error_without_data(monkeypatch):
    exc = MlflowException("API request to http://mlflow:5000 failed: Max retries exceeded")
    monkeypatch.setattr(mlflow_tools, "get_client", lambda: _failing_client(exc))

    result = mlflow_tools.find_best_run("f1")

    assert result.status == ToolStatus.ERROR
    assert result.data is None
    assert "MLflow request failed" in result.error


def test_http_timeout_returns_timeout_status(monkeypatch):
    exc = requests.exceptions.ReadTimeout("read timed out")
    monkeypatch.setattr(mlflow_tools, "get_client", lambda: _failing_client(exc))

    result = mlflow_tools.list_experiments()

    assert result.status == ToolStatus.TIMEOUT
    assert result.data is None


def test_unreachable_tracking_server_with_real_client(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")

    result = mlflow_tools.list_experiments()

    assert result.status in {ToolStatus.ERROR, ToolStatus.TIMEOUT}
    assert result.data is None
