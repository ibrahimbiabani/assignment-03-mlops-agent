"""Training pipeline stages.

validate -> preprocess -> train -> evaluate/log to MLflow -> quality gate -> register.

Each stage is a plain function so it can run from the Airflow DAG
(dags/training_pipeline.py), from the command line, or from tests.

Run locally:  python -m training.train --data data/raw/breast_cancer.csv
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("training")

TARGET = "target"
MIN_ROWS = 100
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "breast-cancer-classifier")
REGISTERED_MODEL = os.environ.get("REGISTERED_MODEL_NAME", "breast-cancer-classifier")
DEFAULT_THRESHOLDS = {"f1": 0.90, "roc_auc": 0.95}
QUALITY_GATE_TAG = "quality_gate"  # read by agent/tools/mlflow_tools.py


class DataValidationError(ValueError):
    """The dataset must not be used for training."""


class QualityGateFailed(RuntimeError):
    """The trained model does not meet the promotion thresholds."""


# ---------- provenance ----------


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=15, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def dataset_provenance(data_path: Path, repo: Path) -> dict[str, str]:
    """Provenance tags for MLflow: DVC hash, Git commit and version tag of the dataset.

    Tag names match agent/tools/mlflow_tools.PROVENANCE_TAGS.
    """
    dvc_file = data_path.with_name(data_path.name + ".dvc")
    rel = data_path.resolve().relative_to(repo.resolve()).as_posix()
    tags = {"data.dvc_path": rel, "data.git_commit": "unknown", "data.version": "unknown"}
    if dvc_file.is_file():
        outs = (yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}).get("outs", [])
        if outs and outs[0].get("md5"):
            tags["data.dvc_md5"] = outs[0]["md5"]
        commit = _git(repo, "log", "-1", "--format=%H", "--", f"{rel}.dvc")
        if commit:
            tags["data.git_commit"] = commit
            version_tags = [
                t for t in (_git(repo, "tag", "--points-at", commit) or "").split() if t
            ]
            tags["data.version"] = version_tags[0] if version_tags else f"untagged-{commit[:7]}"
    return tags


# ---------- stages ----------


def validate_dataset(data_path: str | Path) -> dict[str, Any]:
    """Fail if the data is missing, malformed, or differs from its DVC-tracked version."""
    path = Path(data_path)
    if not path.is_file():
        raise DataValidationError(f"dataset not found: {path}")
    df = pd.read_csv(path)

    problems = []
    if TARGET not in df.columns:
        problems.append(f"missing target column '{TARGET}'")
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows; at least {MIN_ROWS} required")
    nulls = df.isna().sum()
    if nulls.any():
        problems.append(f"null values in columns: {sorted(nulls[nulls > 0].index.tolist())}")
    if TARGET in df.columns:
        labels = set(df[TARGET].dropna().unique().tolist())
        if not labels <= {0, 1} or len(labels) < 2:
            problems.append(f"target must contain both classes 0 and 1, found {sorted(labels)}")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")

    dvc_file = path.with_name(path.name + ".dvc")
    actual_md5 = file_md5(path)
    if dvc_file.is_file():
        outs = (yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}).get("outs", [])
        recorded = outs[0].get("md5") if outs else None
        if recorded and recorded != actual_md5:
            problems.append(
                f"data differs from its DVC-tracked version (recorded md5 {recorded}, "
                f"actual {actual_md5}); run 'dvc checkout' or 'dvc add' and commit"
            )

    if problems:
        raise DataValidationError("; ".join(problems))
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "class_balance": df[TARGET].value_counts(normalize=True).round(3).to_dict(),
        "md5": actual_md5,
        "duplicate_rows": int(df.duplicated().sum()),
    }
    log.info("dataset valid: %s", json.dumps(stats, default=str))
    return stats


def preprocess(
    data_path: str | Path, out_dir: str | Path, test_size: float = 0.2, seed: int = 42
) -> dict[str, str]:
    """Drop duplicates and write a stratified train/test split."""
    df = pd.read_csv(data_path).drop_duplicates()
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"train": str(out / "train.csv"), "test": str(out / "test.csv")}
    # LF on every OS so DVC hashes of the split match between Windows and Linux.
    train.to_csv(paths["train"], index=False, lineterminator="\n")
    test.to_csv(paths["test"], index=False, lineterminator="\n")
    log.info("split %d rows into train=%d test=%d", len(df), len(train), len(test))
    return paths


def build_model(params: dict[str, Any]) -> Pipeline:
    kind = params.get("model_type", "logistic_regression")
    if kind == "logistic_regression":
        clf = LogisticRegression(C=float(params.get("C", 1.0)), max_iter=1000)
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            random_state=int(params.get("seed", 42)),
        )
    else:
        raise ValueError(f"unknown model_type: {kind!r}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def train_model(train_path: str | Path, model_path: str | Path, params: dict[str, Any]) -> str:
    df = pd.read_csv(train_path)
    model = build_model(params).fit(df.drop(columns=[TARGET]), df[TARGET])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("trained %s on %d rows", params.get("model_type", "logistic_regression"), len(df))
    return str(model_path)


def compute_metrics(model: Any, test: pd.DataFrame) -> dict[str, float]:
    """Held-out metrics; shared by the MLflow path and the DVC `evaluate` stage."""
    x, y = test.drop(columns=[TARGET]), test[TARGET]
    pred = model.predict(x)
    return {
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, model.predict_proba(x)[:, 1])),
    }


def evaluate_and_log(
    model_path: str | Path,
    test_path: str | Path,
    params: dict[str, Any],
    provenance: dict[str, str],
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score on the held-out split and log params, metrics, provenance and model to MLflow."""
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    x = df.drop(columns=[TARGET])
    metrics = compute_metrics(model, df)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(tags={**provenance, **(extra_tags or {})}) as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(model, name="model", input_example=x.head(3))
    log.info("logged run %s: %s", run.info.run_id, json.dumps(metrics))
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "metrics": metrics}


def apply_quality_gate(run_id: str, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare the run's logged metrics to thresholds and record the outcome on the run."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    client = MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    failures = {
        m: {"required": t, "actual": metrics.get(m)}
        for m, t in thresholds.items()
        if metrics.get(m) is None or metrics[m] < t
    }
    status = "failed" if failures else "passed"
    client.set_tag(run_id, QUALITY_GATE_TAG, status)
    client.set_tag(run_id, "quality_gate.thresholds", json.dumps(thresholds))
    if failures:
        client.set_tag(run_id, "quality_gate.failures", json.dumps(failures))
        raise QualityGateFailed(f"run {run_id} failed quality gate: {json.dumps(failures)}")
    log.info("run %s passed quality gate %s", run_id, thresholds)
    return {"run_id": run_id, "quality_gate": status, "thresholds": thresholds}


def register_model(run_id: str, model_uri: str, alias: str = "candidate") -> dict[str, Any]:
    """Register a gate-approved model and point `alias` at it. Never called if the gate fails."""
    client = MlflowClient()
    if client.get_run(run_id).data.tags.get(QUALITY_GATE_TAG) != "passed":
        raise QualityGateFailed(f"refusing to register run {run_id}: quality gate not passed")
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL, tags={"source_run_id": run_id})
    client.set_registered_model_alias(REGISTERED_MODEL, alias, mv.version)
    log.info("registered %s version %s as @%s", REGISTERED_MODEL, mv.version, alias)
    return {"model_name": REGISTERED_MODEL, "version": str(mv.version), "alias": alias}


# ---------- local entry point ----------


def run_all(data: Path, work: Path, params: dict[str, Any], repo: Path) -> dict[str, Any]:
    validate_dataset(data)
    split = preprocess(data, work / "processed", seed=int(params.get("seed", 42)))
    model_path = train_model(split["train"], work / "model.joblib", params)
    result = evaluate_and_log(
        model_path, split["test"], params, dataset_provenance(data, repo), {"trigger": "cli"}
    )
    gate = apply_quality_gate(result["run_id"])
    return {**result, **gate, **register_model(result["run_id"], result["model_uri"])}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/raw/breast_cancer.csv"))
    p.add_argument("--work-dir", type=Path, default=Path("artifacts/local"))
    p.add_argument("--model-type", default="logistic_regression")
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()
    params = {"model_type": args.model_type, "C": args.C, "seed": 42}
    print(json.dumps(run_all(args.data, args.work_dir, params, Path.cwd()), indent=2))


if __name__ == "__main__":
    main()
