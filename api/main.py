"""Prediction API: FastAPI inference service for the breast-cancer classifier.

The model is loaded from the MLflow Model Registry (MODEL_NAME@MODEL_ALIAS) in a
background thread, retrying until it succeeds. Until then:
  /health/live  -> 200  (process is up; Kubernetes must not restart it)
  /health/ready -> 503  (no model yet; Kubernetes keeps the Pod out of the Service)
So an unreachable MLflow shows up as an unready Pod with a clear error, not a crash loop.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_VERSION = os.environ.get("APP_VERSION", "dev")
MODEL_NAME = os.environ.get("MODEL_NAME", "breast-cancer-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "candidate")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("prediction_api")

Loader = Callable[[], tuple[Any, dict[str, Any]]]


def load_from_registry() -> tuple[Any, dict[str, Any]]:
    """Resolve MODEL_NAME@MODEL_ALIAS in MLflow and load that exact version."""
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    return model, {
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
    }


class ModelState:
    def __init__(self) -> None:
        self.model: Any = None
        self.info: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.attempts = 0
        self.stop = threading.Event()  # set on shutdown so the loader thread exits

    @property
    def features(self) -> list[str]:
        return list(getattr(self.model, "feature_names_in_", []))


def _load_until_ready(state: ModelState, loader: Loader, retry_s: float) -> None:
    while state.model is None and not state.stop.is_set():
        state.attempts += 1
        try:
            model, info = loader()
            state.info, state.model, state.last_error = info, model, None
            log.info("model loaded: %s (attempt %d)", info, state.attempts)
        except Exception as exc:  # noqa: BLE001 - keep retrying; readiness reports the error
            state.last_error = f"{type(exc).__name__}: {exc}"[:500]
            log.error("model load attempt %d failed: %s", state.attempts, state.last_error)
            state.stop.wait(retry_s)


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[int]
    probabilities: list[float]


def create_app(loader: Loader = load_from_registry, retry_s: float | None = None) -> FastAPI:
    state = ModelState()
    retry = retry_s if retry_s is not None else float(os.environ.get("MODEL_LOAD_RETRY_S", "10"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=_load_until_ready, args=(state, loader, retry), daemon=True).start()
        yield
        state.stop.set()

    app = FastAPI(title="Prediction API", version=APP_VERSION, lifespan=lifespan)
    app.state.model_state = state

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        if state.model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "loading",
                    "attempts": state.attempts,
                    "last_error": state.last_error,
                },
            )
        return JSONResponse({"status": "ready", "model": state.info})

    @app.get("/version")
    def version() -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "model": state.info,
            "model_loaded": state.model is not None,
            "last_error": state.last_error,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if state.model is None:
            raise HTTPException(503, detail=f"model not loaded: {state.last_error or 'loading'}")
        expected = state.features
        for i, row in enumerate(req.instances):
            missing = sorted(set(expected) - row.keys())
            unknown = sorted(row.keys() - set(expected))
            if missing or unknown:
                raise HTTPException(
                    422,
                    detail={
                        "instance": i,
                        "missing_features": missing,
                        "unknown_features": unknown,
                    },
                )
        start = time.perf_counter()
        frame = pd.DataFrame(req.instances)[expected]
        preds = state.model.predict(frame)
        proba = state.model.predict_proba(frame)[:, 1]
        log.info(
            "predicted %d instance(s) in %.1f ms",
            len(frame),
            (time.perf_counter() - start) * 1000,
        )
        return PredictResponse(
            model_name=state.info["name"],
            model_version=state.info["version"],
            predictions=[int(p) for p in preds],
            probabilities=[round(float(p), 6) for p in proba],
        )

    return app


app = create_app()
