"""Retraining policy: retraining is recommended only when evidence violates a rule."""

from datetime import UTC, datetime, timedelta

import pytest

from agent.schemas import ToolStatus
from agent.tools import policy_tools


def _days_ago(n):
    return (datetime.now(UTC) - timedelta(days=n)).isoformat()


GOOD = {
    "model_trained_at": _days_ago(2),
    "f1": 0.94,
    "roc_auc": 0.979,
    "quality_gate": "passed",
    "model_data_version": "data-v1.0",
    "latest_data_version": "data-v1.0",
}


def test_policy_is_loaded_from_config():
    data = policy_tools.get_retraining_policy().data
    assert data["rules"]["max_model_age_days"] == 30
    assert data["model_name"] == "breast-cancer-classifier"


def test_compliant_model_needs_no_retraining():
    data = policy_tools.check_retraining(**GOOD).data
    assert data["retraining_recommended"] is False
    assert data["violated"] == [] and data["unknown"] == []
    assert len(data["satisfied"]) == 5


@pytest.mark.parametrize(
    "change, rule",
    [
        ({"model_trained_at": _days_ago(45)}, "max_model_age_days"),
        ({"f1": 0.85}, "min_f1"),
        ({"quality_gate": "failed"}, "require_quality_gate_passed"),
        ({"latest_data_version": "data-v2.0"}, "require_latest_data_version"),
    ],
)
def test_each_violation_triggers_a_recommendation(change, rule):
    data = policy_tools.check_retraining(**{**GOOD, **change}).data
    assert data["retraining_recommended"] is True
    assert [v["rule"] for v in data["violated"]] == [rule]


def test_missing_evidence_is_unknown_not_assumed():
    data = policy_tools.check_retraining(f1=0.94, model_data_version="unknown").data
    assert data["retraining_recommended"] is False
    unknown = {u["rule"] for u in data["unknown"]}
    assert {"max_model_age_days", "min_roc_auc", "require_quality_gate_passed"} <= unknown
    assert "require_latest_data_version" in unknown  # 'unknown' tag is not treated as a version


def test_bad_timestamp_is_invalid_request():
    result = policy_tools.check_retraining(model_trained_at="last tuesday")
    assert result.status == ToolStatus.INVALID_REQUEST


def test_missing_policy_file_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("RETRAINING_POLICY_PATH", str(tmp_path / "nope.yaml"))
    result = policy_tools.get_retraining_policy()
    assert result.status == ToolStatus.ERROR
    assert result.data is None
