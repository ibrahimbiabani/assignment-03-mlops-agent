"""Structure of the training DAG. Needs Airflow, so it is skipped in CI and on Windows.

Run inside the Airflow container:
    docker compose exec airflow bash -c \
        "cd /opt/airflow/project && python -m pytest -q -p no:cacheprovider tests/test_dag.py"
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.dag_processing.dagbag import DagBag  # noqa: E402

STAGES = [
    "validate_dataset",
    "preprocess_data",
    "train_model",
    "evaluate_model",
    "quality_gate",
    "register_model",
]


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(Path(__file__).parents[1] / "dags"))
    assert bag.import_errors == {}
    return bag.get_dag("training_pipeline")


def test_dag_has_required_stages_in_order(dag):
    assert set(dag.task_ids) == set(STAGES)
    for upstream, downstream in zip(STAGES, STAGES[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_registration_only_reachable_through_quality_gate(dag):
    assert dag.get_task("register_model").upstream_task_ids == {"quality_gate", "evaluate_model"}
    assert "quality_gate" in dag.get_task("evaluate_model").downstream_task_ids


def test_retry_policy(dag):
    # Bad data / a weak model will not fix itself: fail fast.
    assert dag.get_task("validate_dataset").retries == 0
    assert dag.get_task("quality_gate").retries == 0
    for task_id in ("preprocess_data", "train_model", "evaluate_model", "register_model"):
        assert dag.get_task(task_id).retries == 2


def test_dag_is_manual_only(dag):
    assert dag.schedule is None or str(dag.timetable.summary) in ("None", "")
    assert dag.max_active_runs == 1
