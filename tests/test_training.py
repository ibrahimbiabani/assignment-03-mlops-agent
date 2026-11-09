"""Training pipeline tests against a real temporary MLflow store and a real Git repo."""

import subprocess

import mlflow
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer

from agent.tools import mlflow_tools
from training import train


@pytest.fixture(scope="module")
def tracking(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow-training")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    exp = MlflowClient().create_experiment(train.EXPERIMENT, artifact_location=root.as_uri())
    yield uri, exp
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def dataset(tmp_path):
    df = load_breast_cancer(as_frame=True).frame
    df.columns = [c.replace(" ", "_") for c in df.columns]
    path = tmp_path / "data" / "raw" / "breast_cancer.csv"
    path.parent.mkdir(parents=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def _write_dvc_file(data_path, md5):
    data_path.with_name(data_path.name + ".dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": md5, "size": 1, "hash": "md5", "path": data_path.name}]})
    )


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


# ---------- validation ----------


def test_valid_dataset_passes(dataset):
    stats = train.validate_dataset(dataset)
    assert stats["rows"] == 569
    assert set(stats["class_balance"]) == {0, 1}


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda df: df.assign(mean_radius=df["mean_radius"].mask(df.index < 5)), "null values"),
        (lambda df: df.drop(columns=["target"]), "missing target"),
        (lambda df: df.assign(target=1), "both classes"),
        (lambda df: df.head(50), "at least 100"),
        (lambda df: df.assign(mean_radius="large"), "non-numeric"),
    ],
)
def test_invalid_dataset_is_rejected(dataset, corrupt, message):
    corrupt(pd.read_csv(dataset)).to_csv(dataset, index=False)
    with pytest.raises(train.DataValidationError, match=message):
        train.validate_dataset(dataset)


def test_data_that_differs_from_dvc_record_is_rejected(dataset):
    _write_dvc_file(dataset, "0" * 32)
    with pytest.raises(train.DataValidationError, match="differs from its DVC-tracked version"):
        train.validate_dataset(dataset)

    _write_dvc_file(dataset, train.file_md5(dataset))
    assert train.validate_dataset(dataset)["md5"] == train.file_md5(dataset)


# ---------- provenance ----------


def test_provenance_from_dvc_file_and_git_tag(dataset):
    repo = dataset.parents[2]
    _write_dvc_file(dataset, train.file_md5(dataset))
    _git(repo, "init", "-q")
    _git(repo, "add", "data/raw/breast_cancer.csv.dvc")
    _git(repo, "commit", "-q", "-m", "track dataset")
    _git(repo, "tag", "data-v1.0")

    tags = train.dataset_provenance(dataset, repo)

    assert tags["data.version"] == "data-v1.0"
    assert tags["data.dvc_path"] == "data/raw/breast_cancer.csv"
    assert tags["data.dvc_md5"] == train.file_md5(dataset)
    assert len(tags["data.git_commit"]) == 40


def test_provenance_without_git_history_is_marked_unknown(dataset):
    tags = train.dataset_provenance(dataset, dataset.parents[2])
    assert tags["data.git_commit"] == "unknown"
    assert tags["data.version"] == "unknown"
    assert "data.dvc_md5" not in tags


# ---------- end to end ----------


def _train_and_log(dataset, tmp_path, params, provenance):
    split = train.preprocess(dataset, tmp_path / "processed")
    model = train.train_model(split["train"], tmp_path / "model.joblib", params)
    return train.evaluate_and_log(model, split["test"], params, provenance)


def test_pipeline_logs_provenance_passes_gate_and_registers(
    tracking, dataset, tmp_path, monkeypatch
):
    provenance = {
        "data.version": "data-v1.0",
        "data.git_commit": "a" * 40,
        "data.dvc_path": "data/raw/breast_cancer.csv",
        "data.dvc_md5": train.file_md5(dataset),
    }
    result = _train_and_log(dataset, tmp_path, {"model_type": "logistic_regression"}, provenance)

    assert result["metrics"]["f1"] > 0.9
    assert train.apply_quality_gate(result["run_id"])["quality_gate"] == "passed"
    registered = train.register_model(result["run_id"], result["model_uri"])
    assert registered["alias"] == "candidate"

    # The agent's MLflow tool sees the run, its dataset provenance and the gate result.
    monkeypatch.setattr(mlflow_tools, "get_client", MlflowClient)
    best = mlflow_tools.find_best_run("f1", experiment_names=[train.EXPERIMENT]).data["best"]
    assert best["run_id"] == result["run_id"]
    assert best["provenance"]["data.version"] == "data-v1.0"
    assert best["quality_gate"] == "passed"
    versions = mlflow_tools.get_model_versions(train.REGISTERED_MODEL).data["versions"]
    assert versions[0]["aliases"] == ["candidate"]


def test_failed_quality_gate_blocks_registration(tracking, dataset, tmp_path):
    result = _train_and_log(dataset, tmp_path, {"model_type": "logistic_regression"}, {})

    with pytest.raises(train.QualityGateFailed, match="f1"):
        train.apply_quality_gate(result["run_id"], {"f1": 0.999})

    tags = MlflowClient().get_run(result["run_id"]).data.tags
    assert tags["quality_gate"] == "failed"
    assert "f1" in tags["quality_gate.failures"]
    with pytest.raises(train.QualityGateFailed, match="refusing to register"):
        train.register_model(result["run_id"], result["model_uri"])
