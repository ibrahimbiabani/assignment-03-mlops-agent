# MLOps AI Agent

A conversational agent that inspects and operates an ML system through DVC, MLflow,
Apache Airflow and Kubernetes. Operational answers are built only from real tool responses.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,dvc]"
```

## Running tests

```bash
ruff check .
pytest
```

External services are never contacted by the tests; subprocess and API calls are mocked.

## Tools

Every tool returns a `ToolResult` (`agent/schemas.py`) with `status` of `success`, `error`,
`timeout`, `invalid_request` or `rejected`. `data` is only set on success, so the agent cannot
mistake a failure for a result. Each call writes one JSON log line with the tool name, status,
duration and request id; arguments and outputs are never logged.

### DVC (`agent/tools/dvc_tools.py`)

| Tool | Purpose | Backing command |
|---|---|---|
| `dvc_list_tracked_datasets` | List datasets tracked by `.dvc` files with their hashes | reads `*.dvc` |
| `dvc_get_dataset_version` | Current hash, Git commit and tags for a dataset | `git log`, `git tag` |
| `dvc_check_data_changes` | Detect workspace data that differs from DVC records | `dvc status --json` |
| `dvc_compare_dataset_versions` | Diff tracked data between two revisions | `dvc diff --json` |
| `dvc_get_reproduction_steps` | Steps to restore a dataset at a past revision | `git rev-parse`, `git show` |

Permissions: read-only. Only the commands above can run (allowlist in `APPROVED_COMMANDS`),
without a shell. Revisions and paths are validated before use; paths must stay inside the repo.
Set `MLOPS_REPO_PATH` to point the tools at a repository other than the current directory.

### MLflow (`agent/tools/mlflow_tools.py`)

| Tool | Purpose |
|---|---|
| `mlflow_list_experiments` | List active experiments |
| `mlflow_list_runs` | Recent runs (newest first), optionally filtered by status |
| `mlflow_get_run` | Metrics, params, user tags, provenance and artifacts of one run |
| `mlflow_find_best_run` | Best recent **finished** run by a metric (`max`/`min`), with ranking and caveats |
| `mlflow_compare_runs` | Side-by-side metric table for 2-10 runs |
| `mlflow_compare_experiments` | Best run per experiment and the leading experiment |
| `mlflow_get_model_versions` | Registered models, or a model's versions with aliases and source run |

Permissions: read-only, via `MlflowClient` (no UI scraping). Configure with `MLFLOW_TRACKING_URI`.
Requests fail fast (2 retries, 10 s timeout) unless `MLFLOW_HTTP_REQUEST_MAX_RETRIES` /
`MLFLOW_HTTP_REQUEST_TIMEOUT` are set.

`mlflow_find_best_run` never treats the top score as production-ready. It returns caveats drawn
from the recorded evidence: a missing or failed quality gate, missing dataset provenance, a
small margin over the runner-up, or only one comparable run.

**Run tag convention** (written by the training pipeline, read by the agent):

| Tag | Meaning |
|---|---|
| `data.version` | DVC dataset version label, e.g. a Git tag such as `data-v2.1` |
| `data.git_commit` | Git commit of the dataset's `.dvc` file |
| `data.dvc_path` | Path of the tracked dataset |
| `data.dvc_md5` | DVC content hash of the dataset |
| `quality_gate` | `passed` or `failed` |

These tags are what link an MLflow run back to its DVC dataset version.

### Airflow (`agent/tools/airflow_tools.py`)

| Tool | Purpose | Endpoint (Airflow 3, `/api/v2`) |
|---|---|---|
| `airflow_health` | Metadatabase, scheduler, triggerer, DAG processor status | `GET /monitor/health` |
| `airflow_list_dags` | DAGs with pause state, schedule, tags, import errors | `GET /dags` |
| `airflow_get_dag_runs` | Recent runs of a DAG, newest first, optional state filter | `GET /dags/{id}/dagRuns` |
| `airflow_get_dag_run` | One run's status plus the state of every task | `GET .../dagRuns/{run}` + `/taskInstances` |
| `airflow_get_failed_tasks` | Failed tasks of a run (default: latest failed run) and root-cause tasks | `GET .../taskInstances` |
| `airflow_get_task_logs` | Log tail and error lines of a task attempt, secrets redacted | `GET .../logs/{try}` |
| `airflow_trigger_dag` | Trigger an approved DAG (**state-changing**) | `POST /dags/{id}/dagRuns` |

Configuration (environment variables, never committed):

| Variable | Default | Purpose |
|---|---|---|
| `AIRFLOW_BASE_URL` | `http://localhost:8080` | Airflow API server |
| `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | unset | Exchanged for a JWT at `POST /auth/token` |
| `AIRFLOW_TIMEOUT_S` | `10` | Per-request timeout |
| `AIRFLOW_ALLOWED_TRIGGER_DAGS` | `training_pipeline` | Comma-separated DAGs the agent may trigger |

Permissions: all tools are read-only except `airflow_trigger_dag`, which is rejected unless the
DAG is on the approved list **and** the call carries `confirm=True`. The token is cached and
refreshed once on HTTP 401. Credentials and tokens are never logged or returned, and
`password=`/`token=`-style values in task logs are replaced with `***`.

`airflow_get_failed_tasks` separates tasks that failed themselves (`root_cause_tasks`) from
tasks that were only `upstream_failed`.

## Development log

| Step | Change |
|---|---|
| 1 | Project scaffolding following the assignment's recommended layout |
| 2 | Shared `ToolResult` schema, `@tool` wrapper with structured logging, DVC tools (Task 1) and 17 tests |
| 3 | Local `.venv` with editable install (`.[dev,dvc]`); ruff lint/format and pytest all passing |
| 4 | MLflow tools (Task 2) and 23 tests against a real temporary SQLite MLflow store plus mocked server failures (40 tests total) |
| 5 | Airflow 3 REST API tools (Task 3, agent side) and 21 tests against an in-process fake Airflow API over real HTTP (61 tests total) |
