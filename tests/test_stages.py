"""DVC pipeline stages (python -m training.stages ...) on a temporary copy of the data."""

import json

import pytest
import yaml
from sklearn.datasets import load_breast_cancer

from training import stages, train


@pytest.fixture
def workdir(tmp_path):
    df = load_breast_cancer(as_frame=True).frame
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.to_csv(tmp_path / "raw.csv", index=False)
    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "preprocess": {"test_size": 0.25, "seed": 1},
                "train": {"model_type": "logistic_regression", "C": 0.5},
            }
        )
    )
    return tmp_path


def run(workdir, *argv):
    stages.main(["--params", str(workdir / "params.yaml"), *map(str, argv)])


def test_stages_produce_split_model_and_metrics(workdir):
    run(workdir, "preprocess", "--data", workdir / "raw.csv", "--out", workdir / "processed")
    run(
        workdir,
        "train",
        "--train",
        workdir / "processed/train.csv",
        "--model",
        workdir / "m.joblib",
    )
    run(
        workdir,
        "evaluate",
        "--model",
        workdir / "m.joblib",
        "--test",
        workdir / "processed/test.csv",
        "--metrics",
        workdir / "metrics.json",
    )

    test_rows = sum(1 for _ in open(workdir / "processed/test.csv")) - 1
    assert test_rows == 143  # params.yaml test_size=0.25 of 569 rows
    metrics = json.loads((workdir / "metrics.json").read_text())
    assert set(metrics) == {"f1", "accuracy", "precision", "recall", "roc_auc"}
    assert metrics["f1"] > 0.9


def test_preprocess_refuses_invalid_data(workdir):
    (workdir / "raw.csv").write_text("a,b\n1,2\n")
    with pytest.raises(train.DataValidationError):
        run(workdir, "preprocess", "--data", workdir / "raw.csv", "--out", workdir / "out")
