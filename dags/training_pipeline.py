"""Training pipeline.

validate_dataset -> preprocess_data -> train_model -> evaluate_model
    -> quality_gate -> register_model

* validate_dataset and quality_gate fail immediately (no retries): bad data or a weak
  model will not fix itself, and failing blocks every downstream task.
* All other tasks retry twice with exponential backoff for transient errors
  (for example the MLflow server being briefly unavailable).
* register_model is the only promotion step. It runs only after the quality gate
  passes, and re-checks the gate tag recorded on the MLflow run.
"""

import logging
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

PROJECT_DIR = Path(os.environ.get("MLOPS_PROJECT_DIR", "/opt/airflow/project"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=15),
}


def _params() -> dict[str, Any]:
    return dict(get_current_context()["params"])


def _model_params(p: dict[str, Any]) -> dict[str, Any]:
    keys = ("model_type", "C", "n_estimators", "max_depth", "seed")
    return {k: p[k] for k in keys if p.get(k) is not None}


def _work_dir() -> Path:
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", get_current_context()["run_id"])
    path = PROJECT_DIR / "artifacts" / "airflow" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


@dag(
    dag_id="training_pipeline",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ml", "training"],
    doc_md=__doc__,
    params={
        "data_path": Param("data/raw/breast_cancer.csv", type="string"),
        "model_type": Param("logistic_regression", enum=["logistic_regression", "random_forest"]),
        "C": Param(1.0, type="number", exclusiveMinimum=0),
        "n_estimators": Param(200, type="integer", minimum=1),
        "max_depth": Param(None, type=["null", "integer"]),
        "seed": Param(42, type="integer"),
        "min_f1": Param(0.90, type="number", minimum=0, maximum=1),
        "min_roc_auc": Param(0.95, type="number", minimum=0, maximum=1),
    },
)
def training_pipeline() -> None:
    @task(retries=0)
    def validate_dataset() -> dict[str, Any]:
        from training import train

        data = (PROJECT_DIR / _params()["data_path"]).resolve()
        if not data.is_relative_to(PROJECT_DIR.resolve()):
            raise AirflowFailException(f"data_path must be inside the project: {data}")
        try:
            stats = train.validate_dataset(data)
        except train.DataValidationError as exc:
            raise AirflowFailException(f"dataset validation failed: {exc}") from exc
        provenance = train.dataset_provenance(data, PROJECT_DIR)
        log.info("dataset provenance: %s", provenance)
        return {"data_path": str(data), "stats": stats, "provenance": provenance}

    @task
    def preprocess_data(validated: dict[str, Any]) -> dict[str, str]:
        from training import train

        return train.preprocess(
            validated["data_path"], _work_dir() / "processed", seed=_params()["seed"]
        )

    @task
    def train_model(split: dict[str, str]) -> str:
        from training import train

        return train.train_model(
            split["train"], _work_dir() / "model.joblib", _model_params(_params())
        )

    @task
    def evaluate_model(
        model_path: str, split: dict[str, str], validated: dict[str, Any]
    ) -> dict[str, Any]:
        from training import train

        ctx = get_current_context()
        return train.evaluate_and_log(
            model_path,
            split["test"],
            _model_params(_params()),
            validated["provenance"],
            {"trigger": "airflow", "airflow.dag_run_id": ctx["run_id"]},
        )

    @task(retries=0)
    def quality_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
        from training import train

        p = _params()
        thresholds = {"f1": p["min_f1"], "roc_auc": p["min_roc_auc"]}
        try:
            return train.apply_quality_gate(evaluation["run_id"], thresholds)
        except train.QualityGateFailed as exc:
            raise AirflowFailException(str(exc)) from exc

    @task
    def register_model(gate: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        from training import train

        return train.register_model(gate["run_id"], evaluation["model_uri"])

    validated = validate_dataset()
    split = preprocess_data(validated)
    model_path = train_model(split)
    evaluation = evaluate_model(model_path, split, validated)
    gate = quality_gate(evaluation)
    register_model(gate, evaluation)


training_pipeline()
