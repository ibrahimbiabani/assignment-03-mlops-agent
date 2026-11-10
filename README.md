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

External services are never contacted by the tests: DVC subprocess calls are mocked, MLflow
uses a temporary SQLite store, and Airflow is an in-process fake API. `tests/test_dag.py` needs
Airflow, so it is skipped locally and in CI; run it inside the Airflow container:

```bash
docker compose exec airflow python -m pip install -q pytest
docker compose exec airflow bash -c "cd /opt/airflow/project && python -m pytest -q -p no:cacheprovider tests/test_dag.py"
```

## Local stack (MLflow + Airflow)

Requires Docker Desktop.

```bash
cp .env.example .env              # set AIRFLOW_ADMIN_PASSWORD and AIRFLOW_PASSWORD (same value)
docker compose up -d --build
docker compose exec airflow airflow dags unpause training_pipeline
```

| Service | URL | Notes |
|---|---|---|
| MLflow | http://localhost:5000 | SQLite backend and artifacts in the `mlflow-data` volume |
| Airflow | http://localhost:8080 | `airflow standalone` (Airflow 3.4.0), user `admin`, password from `.env` |

The Airflow image (`docker/airflow/Dockerfile`) adds `mlflow-skinny` under Airflow's official
constraints. The project is mounted at `/opt/airflow/project` so the DAG can import `training/`,
read the DVC-tracked data and its Git history, and write per-run files to `artifacts/`.

## Dataset (DVC)

`data/raw/breast_cancer.csv` (Wisconsin breast-cancer data from scikit-learn) is tracked by DVC;
Git stores only `data/raw/breast_cancer.csv.dvc`. The default remote `localremote` is the folder
`../dvc-remote` next to the repository.

```bash
dvc pull                                         # fetch the data for the checked-out version
python -m training.make_dataset --rows 400       # how version 1 was created
dvc add data/raw/breast_cancer.csv && dvc push   # after changing the data
git tag data-v1.0                                # name dataset versions with data-* tags
```

## Training pipeline (`dags/training_pipeline.py`)

`validate_dataset -> preprocess_data -> train_model -> evaluate_model -> quality_gate -> register_model`

| Task | What it does | Retries |
|---|---|---|
| `validate_dataset` | Schema, nulls, both classes, minimum rows, and the file's md5 must match its `.dvc` record | 0 |
| `preprocess_data` | Drop duplicates, stratified 80/20 split | 2 |
| `train_model` | scikit-learn pipeline (`logistic_regression` or `random_forest`) | 2 |
| `evaluate_model` | F1, accuracy, precision, recall, ROC-AUC on the held-out split; logs params, metrics, model and dataset provenance tags to MLflow | 2 |
| `quality_gate` | Fails unless `f1 >= min_f1` (0.90) and `roc_auc >= min_roc_auc` (0.95); records `quality_gate` on the MLflow run | 0 |
| `register_model` | Registers the model as `breast-cancer-classifier` and moves the `@candidate` alias to it | 2 |

Validation and the quality gate fail without retrying (bad data or a weak model will not fix
itself), and every later task is then skipped as `upstream_failed`. `register_model` is the only
promotion step and re-checks the gate tag before registering. Thresholds and model settings are
DAG params, e.g. trigger with conf `{"model_type": "random_forest", "min_f1": 0.95}`.

The same stages run without Airflow: `python -m training.train --data data/raw/breast_cancer.csv`.

## Prediction API on Kubernetes

`api/main.py` is a FastAPI inference service. At startup it loads
`breast-cancer-classifier@candidate` from the MLflow registry in the background and retries
until it succeeds.

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Liveness: 200 while the process runs (model loading never causes a restart) |
| `GET /health/ready` | Readiness: 200 once the model is loaded, otherwise 503 with the last load error |
| `GET /version` | App version plus the served model name, version, alias and MLflow run |
| `POST /predict` | `{"instances": [{feature: value, ...}]}` (1-100 rows) -> predictions and probabilities |

Deploy to Docker Desktop's Kubernetes (Settings -> Kubernetes -> Kubeadm). MLflow must be
running in docker compose; Pods reach it at `host.docker.internal:5000`.

```bash
docker build -t prediction-api:1.0.0 .
kubectl config use-context docker-desktop
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl -n mlops-app rollout status deployment/prediction-api
```

| Manifest | Contents |
|---|---|
| `namespace.yaml` | `mlops-app`, the only namespace the agent may touch |
| `configmap.yaml` | MLflow URI, model name/alias, retry settings (nothing secret) |
| `deployment.yaml` | 2 replicas, readiness + liveness probes, CPU/memory requests and limits, non-root, read-only root filesystem, rolling updates with `maxUnavailable: 0` |
| `service.yaml` | ClusterIP Service, port 80 -> container port 8000 |
| `agent-rbac.yaml` | `mlops-agent` ServiceAccount + Role: read the namespace, patch Deployments (restart) and their scale; nothing else |

Give the agent the restricted identity (the token stays in your kubeconfig, not the repo; re-run
the first command when it expires):

```bash
kubectl config set-credentials mlops-agent --token="$(kubectl -n mlops-app create token mlops-agent --duration=168h)"
kubectl config set-context mlops-agent --cluster=docker-desktop --user=mlops-agent --namespace=mlops-app
```

Then set `K8S_CONTEXT=mlops-agent` in `.env`.

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

### Kubernetes (`agent/tools/kubernetes_tools.py`)

| Tool | Purpose |
|---|---|
| `k8s_list_workloads` | Deployments (desired/ready replicas, image, version, revision) and Pods |
| `k8s_get_pod_status` | Phase, conditions, container states (e.g. `CrashLoopBackOff`, last `OOMKilled`), probe settings, the Pod's events |
| `k8s_get_pod_logs` | Log tail of a Pod, or of its previous crashed container; error lines; secrets redacted |
| `k8s_list_services` | Services with type, ports, selector, and ready/not-ready endpoint counts |
| `k8s_find_unhealthy` | Unhealthy/restarting Pods with reasons, under-replicated Deployments, newest Warning events |
| `k8s_get_deployed_version` | Manifest version/image/revision plus what the running app reports at `/version` (via the API server's service proxy) |
| `k8s_restart_deployment` | Rolling restart (**state-changing**) |
| `k8s_scale_deployment` | Scale to 1-5 replicas (**state-changing**) |

| Variable | Default | Purpose |
|---|---|---|
| `K8S_CONTEXT` | `docker-desktop` | kubeconfig context; use the restricted `mlops-agent` |
| `K8S_ALLOWED_NAMESPACES` | `mlops-app` | Namespaces the tools may access |
| `K8S_TIMEOUT_S` | `10` | Per-request timeout |

Permissions are enforced twice. The tools reject any namespace outside the allowed list, and
both state-changing tools also require `confirm=True`; both checks happen before any API call.
In the cluster, the `mlops-agent` Role allows only reads in `mlops-app` plus patching
Deployments and their scale, so it cannot delete, create, read Secrets or reach other
namespaces even if the code were bypassed.

## Development log

| Step | Change |
|---|---|
| 1 | Project scaffolding following the assignment's recommended layout |
| 2 | Shared `ToolResult` schema, `@tool` wrapper with structured logging, DVC tools (Task 1) and 17 tests |
| 3 | Local `.venv` with editable install (`.[dev,dvc]`); ruff lint/format and pytest all passing |
| 4 | MLflow tools (Task 2) and 23 tests against a real temporary SQLite MLflow store plus mocked server failures (40 tests total) |
| 5 | Airflow 3 REST API tools (Task 3, agent side) and 21 tests against an in-process fake Airflow API over real HTTP (61 tests total) |
| 6 | GitHub Actions CI: ruff lint, format check and pytest on Python 3.11 and 3.14 for pushes to `main` and pull requests |
| 7 | DVC-tracked dataset v1 (400 rows) with local remote; `training/train.py` stages; Airflow 3.4 + MLflow 3.18 Docker stack; `training_pipeline` DAG. Verified live: a run triggered via the agent tool passed all 6 tasks (F1 0.94, model v1 `@candidate`); a run with `min_f1=0.999` failed at `quality_gate`, blocked `register_model`, and was diagnosed by the agent tools from Airflow 3 structured `error_detail` logs |
| 8 | Task 4: FastAPI prediction service (`api/`), its image, Kubernetes manifests (namespace, ConfigMap, 2-replica Deployment with probes and limits, ClusterIP Service, agent RBAC) and 8 Kubernetes tools with namespace + confirmation guards. Deployed on Docker Desktop Kubernetes 1.36: Pods load model v1 from MLflow; `/version` read through the API proxy; confirmed scale/restart worked; an injected bad image tag (`ErrImagePull`) was diagnosed by the tools via the restricted `mlops-agent` identity while the Service kept serving. 42 new tests (115 total) |
