"""Retraining policy tools.

The policy lives in config/retraining_policy.yaml (path overridable with
RETRAINING_POLICY_PATH). The check is deterministic: the LLM supplies evidence it
retrieved from MLflow/DVC, and this tool decides which rules are violated, satisfied,
or cannot be evaluated because evidence is missing.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agent.tools.base import InvalidToolRequest, ToolError, tool

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "retraining_policy.yaml"


def load_policy() -> dict[str, Any]:
    path = Path(os.environ.get("RETRAINING_POLICY_PATH", DEFAULT_POLICY_PATH))
    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ToolError(f"retraining policy not readable: {path}") from exc
    if not isinstance(policy.get("rules"), dict):
        raise ToolError(f"retraining policy has no 'rules' section: {path}")
    return policy


def _parse_time(value: str) -> datetime:
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise InvalidToolRequest(f"model_trained_at is not an ISO-8601 time: {value!r}") from exc
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


@tool("policy_get_retraining_policy")
def get_retraining_policy() -> dict[str, Any]:
    """The explicit retraining policy (freshness, metric floors, gate, data-version rules)."""
    return {**load_policy(), "now": datetime.now(UTC).isoformat()}


@tool("policy_check_retraining")
def check_retraining(
    model_trained_at: str | None = None,
    f1: float | None = None,
    roc_auc: float | None = None,
    quality_gate: str | None = None,
    model_data_version: str | None = None,
    latest_data_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate recorded evidence against the retraining policy.

    Pass values taken from tool results (e.g. the served model's MLflow run start_time,
    metrics, quality_gate tag and data.version tag, and the newest DVC dataset tag).
    Leave a value out when it is unknown; that rule is then reported as 'unknown'.
    Retraining is recommended only if at least one rule is violated.
    """
    rules = load_policy()["rules"]
    violated, satisfied, unknown = [], [], []

    def judge(rule: str, value: Any, ok: bool | None, detail: str) -> None:
        entry = {"rule": rule, "evidence": value, "detail": detail}
        (unknown if ok is None else satisfied if ok else violated).append(entry)

    if "max_model_age_days" in rules:
        limit = rules["max_model_age_days"]
        if model_trained_at is None:
            judge("max_model_age_days", None, None, "model training time not provided")
        else:
            age = (datetime.now(UTC) - _parse_time(model_trained_at)).total_seconds() / 86400
            judge("max_model_age_days", round(age, 1), age <= limit, f"age must be <= {limit} days")
    for metric, value in (("f1", f1), ("roc_auc", roc_auc)):
        key = f"min_{metric}"
        if key in rules:
            floor = rules[key]
            ok = None if value is None else value >= floor
            judge(key, value, ok, f"{metric} must be >= {floor}")
    if rules.get("require_quality_gate_passed"):
        ok = None if quality_gate is None else quality_gate == "passed"
        judge("require_quality_gate_passed", quality_gate, ok, "quality_gate tag must be 'passed'")
    if rules.get("require_latest_data_version"):
        known = None not in (model_data_version, latest_data_version) and "unknown" not in (
            model_data_version,
            latest_data_version,
        )
        ok = model_data_version == latest_data_version if known else None
        judge(
            "require_latest_data_version",
            {"model": model_data_version, "latest": latest_data_version},
            ok,
            "model must be trained on the newest dataset version",
        )
    return {
        "retraining_recommended": bool(violated),
        "violated": violated,
        "satisfied": satisfied,
        "unknown": unknown,
        "note": "Recommendation only. Retraining is never started without user confirmation.",
    }
