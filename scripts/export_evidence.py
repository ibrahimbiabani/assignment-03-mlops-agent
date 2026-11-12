"""Export read-only tool output as evidence for the technical report.

    python scripts/export_evidence.py [--out docs/evidence] [--only dvc,mlflow,airflow,kubernetes]

Calls the same tools the agent uses, with no LLM and no state changes, and writes one
JSON file per technology. Every entry keeps the tool's status, so an unavailable service
appears as an error rather than as missing evidence.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.registry import default_registry

DATASET = "data/raw/breast_cancer.csv"
DAG_ID = "training_pipeline"


def call(registry: dict, name: str, **kwargs: Any) -> dict[str, Any]:
    result = registry[name].func(**kwargs)
    return {"tool": name, "arguments": kwargs, **result.model_dump(mode="json", exclude={"tool"})}


def dvc(r: dict) -> list[dict]:
    return [
        call(r, "dvc_list_tracked_datasets"),
        call(r, "dvc_get_dataset_version", path=DATASET),
        call(r, "dvc_check_data_changes"),
        call(r, "dvc_compare_dataset_versions", rev_a="data-v1.0", rev_b="data-v2.0"),
        call(r, "dvc_get_reproduction_steps", path=DATASET, rev="data-v1.0"),
    ]


def mlflow(r: dict) -> list[dict]:
    out = [
        call(r, "mlflow_list_experiments"),
        call(r, "mlflow_list_runs", max_results=5),
        best := call(r, "mlflow_find_best_run", metric="f1", max_runs=5),
    ]
    run_id = ((best.get("data") or {}).get("best") or {}).get("run_id")
    if run_id:
        out.append(call(r, "mlflow_get_run", run_id=run_id))
    out.append(call(r, "mlflow_get_model_versions"))
    return out


def airflow(r: dict) -> list[dict]:
    out = [
        call(r, "airflow_health"),
        call(r, "airflow_list_dags"),
        runs := call(r, "airflow_get_dag_runs", dag_id=DAG_ID, limit=5),
    ]
    for run in ((runs.get("data") or {}).get("runs") or [])[:1]:
        out.append(call(r, "airflow_get_dag_run", dag_id=DAG_ID, dag_run_id=run["dag_run_id"]))
        out.append(call(r, "airflow_get_failed_tasks", dag_id=DAG_ID, dag_run_id=run["dag_run_id"]))
    return out


def kubernetes(r: dict) -> list[dict]:
    out = [
        call(r, "k8s_list_workloads"),
        call(r, "k8s_list_services"),
        call(r, "k8s_find_unhealthy"),
        call(r, "k8s_get_deployed_version", deployment="prediction-api"),
    ]
    pods = ((out[0].get("data") or {}).get("pods")) or []
    if pods:
        pod = pods[0]["name"]
        out.append(call(r, "k8s_get_pod_status", pod_name=pod))
        out.append(call(r, "k8s_get_pod_logs", pod_name=pod, tail_lines=20))
    return out


SECTIONS = {"dvc": dvc, "mlflow": mlflow, "airflow": airflow, "kubernetes": kubernetes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("docs/evidence"))
    p.add_argument("--only", default=",".join(SECTIONS))
    args = p.parse_args()
    load_dotenv()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    registry = default_registry()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        entries = SECTIONS[name](registry)
        doc = {"exported_at": datetime.now(UTC).isoformat(timespec="seconds"), "calls": entries}
        path = args.out / f"tools_{name}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        statuses = ", ".join(f"{e['tool']}={e['status']}" for e in entries)
        print(f"{path}: {statuses}")


if __name__ == "__main__":
    main()
