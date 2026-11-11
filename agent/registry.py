"""Tool registry: every tool the agent can call, its input schema and its permission class.

Input schemas are generated from each tool's Python signature, so the LLM sees exactly
the parameters the function accepts. `confirm` is never exposed: state-changing tools
can only be confirmed by the user through the orchestrator's pending-action flow.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from agent.schemas import ToolResult
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools, policy_tools

# Descriptions for parameters shared across tools (shown to the LLM).
PARAM_DOCS = {
    "path": "Dataset path relative to the repo, e.g. data/raw/breast_cancer.csv",
    "rev": "Git revision: commit SHA, tag such as data-v1.0, or HEAD~1",
    "rev_a": "Older Git revision (commit SHA, data-v* tag, HEAD~1, ...)",
    "rev_b": "Newer Git revision (commit SHA, data-v* tag, HEAD, ...)",
    "metric": "MLflow metric name, e.g. f1, roc_auc, accuracy",
    "mode": "'max' if higher is better, 'min' if lower is better",
    "experiment_names": "MLflow experiment names; omit for all experiments",
    "run_id": "MLflow run id (32 hex characters)",
    "run_ids": "2-10 MLflow run ids",
    "model_name": "Registered model name; omit to list all registered models",
    "max_results": "Maximum number of items to return",
    "max_runs": "How many of the most recent finished runs to consider",
    "status": "MLflow run status filter: FINISHED, FAILED, RUNNING, KILLED or SCHEDULED",
    "dag_id": "Airflow DAG id, e.g. training_pipeline",
    "dag_run_id": "Airflow DAG run id, e.g. manual__2026-11-09T08:00:16.073707+00:00",
    "task_id": "Airflow task id, e.g. quality_gate",
    "try_number": "Task attempt number; omit for the latest attempt",
    "max_lines": "Maximum log lines to return",
    "conf": 'Run configuration for the DAG (its params), e.g. {"min_f1": 0.92}',
    "limit": "Maximum number of items to return",
    "state": "DAG run state filter: queued, running, success or failed",
    "namespace": "Kubernetes namespace (only approved namespaces are allowed)",
    "pod_name": "Kubernetes Pod name",
    "container": "Container name; omit for the Pod's only container",
    "previous": "True to read the previous (crashed) container's log",
    "tail_lines": "Number of log lines from the end",
    "deployment": "Kubernetes Deployment name, e.g. prediction-api",
    "service": "Kubernetes Service name; defaults to the deployment name",
    "replicas": "Desired replica count (1-5)",
    "model_trained_at": "ISO-8601 start time of the served model's MLflow run",
    "f1": "Served model's F1 from MLflow",
    "roc_auc": "Served model's ROC-AUC from MLflow",
    "quality_gate": "Served model's quality_gate tag from MLflow (passed/failed)",
    "model_data_version": "data.version tag of the served model's MLflow run",
    "latest_data_version": "Newest dataset version tag from DVC/Git, e.g. data-v2.0",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., ToolResult]
    description: str
    args_model: type[BaseModel]
    state_changing: bool

    def parameters_schema(self) -> dict[str, Any]:
        return _clean_schema(self.args_model.model_json_schema())


def _clean_schema(node: Any) -> Any:
    """Drop pydantic titles and collapse Optional[X] (anyOf X|null) to X for the LLM."""
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if any_of and len(any_of) == 2 and {"type": "null"} in any_of:
            inner = next(s for s in any_of if s != {"type": "null"})
            node = {**{k: v for k, v in node.items() if k != "anyOf"}, **inner}
        return {k: _clean_schema(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def spec_from(func: Callable[..., ToolResult], state_changing: bool = False) -> ToolSpec:
    name = func.tool_name  # type: ignore[attr-defined]
    fields: dict[str, Any] = {}
    for p in inspect.signature(func).parameters.values():
        if p.name == "confirm":
            continue
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (p.annotation, Field(default, description=PARAM_DOCS.get(p.name)))
    args_model = create_model(
        f"{name}_args",
        __config__=ConfigDict(extra="forbid"),  # unknown arguments (incl. confirm) are refused
        **fields,
    )
    doc = inspect.getdoc(func) or name
    if state_changing:
        doc += (
            "\n\nSTATE-CHANGING: calling this only creates a proposal. It is executed "
            "only if the user confirms it; tell the user it is pending confirmation."
        )
    return ToolSpec(name, func, doc, args_model, state_changing)


def default_registry() -> dict[str, ToolSpec]:
    read_only = [
        dvc_tools.list_tracked_datasets,
        dvc_tools.get_dataset_version,
        dvc_tools.check_data_changes,
        dvc_tools.compare_dataset_versions,
        dvc_tools.get_reproduction_steps,
        mlflow_tools.list_experiments,
        mlflow_tools.list_runs,
        mlflow_tools.get_run,
        mlflow_tools.find_best_run,
        mlflow_tools.compare_runs,
        mlflow_tools.compare_experiments,
        mlflow_tools.get_model_versions,
        airflow_tools.health,
        airflow_tools.list_dags,
        airflow_tools.get_dag_runs,
        airflow_tools.get_dag_run,
        airflow_tools.get_failed_tasks,
        airflow_tools.get_task_logs,
        kubernetes_tools.list_workloads,
        kubernetes_tools.get_pod_status,
        kubernetes_tools.get_pod_logs,
        kubernetes_tools.list_services,
        kubernetes_tools.find_unhealthy,
        kubernetes_tools.get_deployed_version,
        policy_tools.get_retraining_policy,
        policy_tools.check_retraining,
    ]
    state_changing = [
        airflow_tools.trigger_dag,
        kubernetes_tools.restart_deployment,
        kubernetes_tools.scale_deployment,
    ]
    specs = [spec_from(f) for f in read_only] + [spec_from(f, True) for f in state_changing]
    return {s.name: s for s in specs}
