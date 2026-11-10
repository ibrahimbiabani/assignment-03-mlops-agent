"""Prediction API tests with an in-memory model (no MLflow)."""

import threading
import time

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from api.main import create_app

INFO = {"name": "breast-cancer-classifier", "alias": "candidate", "version": "7", "run_id": "r1"}


@pytest.fixture(scope="module")
def trained():
    df = load_breast_cancer(as_frame=True).frame
    df.columns = [c.replace(" ", "_") for c in df.columns]
    x, y = df.drop(columns=["target"]), df["target"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    return model, x


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def ready_client(trained):
    model, _ = trained
    with TestClient(create_app(loader=lambda: (model, INFO), retry_s=0.01)) as c:
        assert _wait(lambda: c.get("/health/ready").status_code == 200)
        yield c


def test_not_ready_until_model_loaded(trained):
    release = threading.Event()

    def slow_loader():
        release.wait(5)
        return trained[0], INFO

    with TestClient(create_app(loader=slow_loader, retry_s=0.01)) as c:
        assert c.get("/health/live").status_code == 200
        assert c.get("/health/ready").status_code == 503
        assert c.post("/predict", json={"instances": [{"x": 1.0}]}).status_code == 503
        release.set()
        assert _wait(lambda: c.get("/health/ready").status_code == 200)


def test_failed_loads_are_retried_and_reported(trained):
    attempts = {"n": 0}

    def flaky_loader():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("MLflow unreachable")
        return trained[0], INFO

    app = create_app(loader=flaky_loader, retry_s=0.01)
    state = app.state.model_state
    with TestClient(app) as c:
        assert _wait(lambda: c.get("/health/ready").status_code == 200)
    assert state.attempts == 3
    assert state.last_error is None  # cleared after the successful load


def test_readiness_exposes_last_error_while_failing(trained):
    def broken_loader():
        raise ConnectionError("MLflow unreachable at http://host.docker.internal:5000")

    app = create_app(loader=broken_loader, retry_s=0.01)
    with TestClient(app) as c:
        assert _wait(lambda: c.get("/health/ready").json().get("last_error") is not None)
        body = c.get("/health/ready").json()
        assert body["status"] == "loading"
        assert "MLflow unreachable" in body["last_error"]
        assert c.get("/health/live").status_code == 200  # never restarted for this
    # Shutdown stops the retry loop instead of leaking a thread.
    attempts = app.state.model_state.attempts
    time.sleep(0.1)
    assert app.state.model_state.attempts <= attempts + 1


def test_predict(ready_client, trained):
    _, x = trained
    rows = x.head(3).to_dict("records")

    r = ready_client.post("/predict", json={"instances": rows})

    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "7"
    assert len(body["predictions"]) == 3 and set(body["predictions"]) <= {0, 1}
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])


def test_predict_reports_missing_and_unknown_features(ready_client, trained):
    row = trained[1].iloc[0].to_dict()
    row.pop("mean_radius")
    row["colour"] = 1.0

    r = ready_client.post("/predict", json={"instances": [row]})

    assert r.status_code == 422
    assert r.json()["detail"] == {
        "instance": 0,
        "missing_features": ["mean_radius"],
        "unknown_features": ["colour"],
    }


@pytest.mark.parametrize(
    "payload",
    [{"instances": []}, {"instances": [{"a": 1.0}] * 101}, {"instances": [{"a": "x"}]}, {}],
)
def test_predict_rejects_malformed_requests(ready_client, payload):
    assert ready_client.post("/predict", json=payload).status_code == 422


def test_version(ready_client):
    body = ready_client.get("/version").json()
    assert body["model"] == INFO
    assert body["model_loaded"] is True
