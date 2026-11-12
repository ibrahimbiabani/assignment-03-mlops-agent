# Technical Report: MLOps AI Agent

Assignment 03, MLOps & Production AI Engineering, Fall 2026.
Repository: `assignment-03-mlops-agent`. All evidence below was produced on 2026-11-12 against
the live stack: MLflow 3.18 and Airflow 3.4 in Docker, Kubernetes 1.36 on Docker Desktop, and
the Gemini free tier.

## Contents

1. [System architecture](#1-system-architecture)
2. [Tool definitions and permissions](#2-tool-definitions-and-permissions)
3. [Evidence from DVC, MLflow, Airflow and Kubernetes](#3-evidence-from-dvc-mlflow-airflow-and-kubernetes)
4. [Five tested agent conversations](#4-five-tested-agent-conversations)
5. [Failure investigations](#5-failure-investigations)
6. [Testing, reliability and safety](#6-testing-reliability-and-safety)
7. [Known limitations and proposed improvements](#7-known-limitations-and-proposed-improvements)

## 1. System architecture

Full description, diagrams and design rationale: [architecture.md](../architecture.md).

```
User ──POST /chat──▶ Agent API (FastAPI) ──▶ Orchestrator ◀──function calls──▶ Gemini model pool
                         │                      │  validates every call, bounded loop,
                         │                      │  evidence + unavailable-source notices
                         │                      ▼
                         │                Tool registry (29 tools, generated schemas)
                         │        ┌────────────┬──────────────┬────────────────┬──────────────┐
                         │        ▼            ▼              ▼                ▼              ▼
                         │   DVC tools    MLflow tools   Airflow tools   Kubernetes tools  Policy
                         │   git + dvc    MlflowClient   REST /api/v2    RBAC-restricted   YAML
                         │   allowlist    (:5000)        JWT (:8081)     context, mlops-app
                         └──POST /actions/{id}/confirm──▶ executes a stored, approved action
```

- **Data.** `data/raw/breast_cancer.csv` is versioned with DVC (tags `data-v1.0`: 400 rows;
  `data-v2.0`: 569 rows) and pushed to a separate DVC remote.
- **Training.** The Airflow DAG `training_pipeline` runs validate → preprocess → train →
  evaluate/MLflow → quality gate → register. It records the dataset provenance tags on each
  MLflow run and moves `@candidate` only if the gate passes. `dvc repro` runs the same code
  without side effects.
- **Serving.** A FastAPI inference service loads `breast-cancer-classifier@candidate`. It runs
  in namespace `mlops-app` as a Deployment (2 replicas, probes, limits) with a ClusterIP
  Service and a ConfigMap.
- **Agent.** It never changes state on its own. State-changing tools become pending actions
  that only a separate HTTP confirmation executes.

## 2. Tool definitions and permissions

The schemas are generated from the Python signatures (`agent/registry.py`) and validated with
`extra="forbid"`. Every tool returns `ToolResult {tool, status, data | error, duration_ms}`.
The possible statuses are `success`, `error`, `timeout`, `invalid_request`, `rejected` and
`confirmation_required`.

| Tool | Access | Required args | Optional args | Purpose |
|---|---|---|---|---|
| `dvc_list_tracked_datasets` | read | - | - | Datasets tracked by `.dvc` files with content hashes |
| `dvc_get_dataset_version` | read | path | - | Current hash plus the Git commit and tags that recorded it |
| `dvc_check_data_changes` | read | - | - | Whether workspace data differs from DVC records (`dvc status`) |
| `dvc_compare_dataset_versions` | read | rev_a, rev_b | - | `dvc diff` between two revisions |
| `dvc_get_reproduction_steps` | read | path, rev | - | Commands that restore a dataset exactly as at `rev` |
| `mlflow_list_experiments` | read | - | max_results | Active experiments |
| `mlflow_list_runs` | read | - | experiment_names, max_results, status | Recent runs, newest first |
| `mlflow_get_run` | read | run_id | - | Metrics, params, tags, provenance, artifacts, logged models |
| `mlflow_find_best_run` | read | metric | mode, experiment_names, max_runs | Best recent finished run with ranking and caveats |
| `mlflow_compare_runs` | read | run_ids | - | Metric/parameter table for 2-10 runs |
| `mlflow_compare_experiments` | read | experiment_names, metric | mode, max_runs | Best run per experiment |
| `mlflow_get_model_versions` | read | - | model_name, max_results | Registry versions, aliases, source runs |
| `airflow_health` | read | - | - | Metadatabase, scheduler, triggerer, DAG processor status |
| `airflow_list_dags` | read | - | limit | DAGs, pause state, import errors |
| `airflow_get_dag_runs` | read | dag_id | limit, state | Recent DAG runs |
| `airflow_get_dag_run` | read | dag_id, dag_run_id | - | One run with every task's state |
| `airflow_get_failed_tasks` | read | dag_id | dag_run_id | Failed and root-cause tasks of a run |
| `airflow_get_task_logs` | read | dag_id, dag_run_id, task_id | try_number, max_lines | Log tail, error lines, exceptions (redacted) |
| `k8s_list_workloads` | read | - | namespace | Deployments and Pods |
| `k8s_get_pod_status` | read | pod_name | namespace | Phase, conditions, container states, probes, events |
| `k8s_get_pod_logs` | read | pod_name | namespace, container, previous, tail_lines | Pod logs, including the previous crashed container (redacted) |
| `k8s_list_services` | read | - | namespace | Services with ready/not-ready endpoint counts |
| `k8s_find_unhealthy` | read | - | namespace | Unhealthy Pods, crash vs. clean restarts, under-replicated Deployments, Warning events |
| `k8s_get_deployed_version` | read | - | deployment, namespace, service | Image, revision and the model version the app reports at `/version` |
| `policy_get_retraining_policy` | read | - | - | The explicit retraining policy |
| `policy_check_retraining` | read | - | model_trained_at, f1, roc_auc, quality_gate, model_data_version, latest_data_version | Each rule: violated, satisfied or unknown |
| `airflow_trigger_dag` | **state-changing** | dag_id | conf | Trigger an approved DAG |
| `k8s_restart_deployment` | **state-changing** | deployment | namespace | Rolling restart |
| `k8s_scale_deployment` | **state-changing** | deployment, replicas | namespace | Scale to 1-5 replicas |

**Permissions, enforced in four layers.**

1. **Schema validation.** `confirm` is never part of any schema, so the LLM cannot pass it.
2. **Tool guards.** These are checked before any network call:
   - DVC: an allowlist of `dvc`/`git` subcommands, run without a shell, with validated
     revisions and paths confined to the repository.
   - Airflow: the DAG must be in `AIRFLOW_ALLOWED_TRIGGER_DAGS`.
   - Kubernetes: the namespace must be in `K8S_ALLOWED_NAMESPACES` (default `mlops-app`),
     replicas must be 1-5, and names must match the Kubernetes format.
3. **Human confirmation.** A permitted state change is stored as a single-use pending action
   with a 600 s TTL. Only `POST /actions/{id}/confirm` executes it.
4. **Cluster RBAC.** The agent's `mlops-agent` ServiceAccount may only read `mlops-app` and
   patch Deployments and their scale.

All other tools are read-only. Credentials are taken from environment variables, and the
Kubernetes token lives in the user's kubeconfig, outside the repository.

## 3. Evidence from DVC, MLflow, Airflow and Kubernetes

The exports are machine-generated by `scripts/export_evidence.py`, which calls the same
read-only tools the agent uses, with no LLM involved.

| Technology | Evidence | Highlights |
|---|---|---|
| DVC | [01_dvc_reproducible_checkout.txt](evidence/01_dvc_reproducible_checkout.txt) | A fresh clone with an empty cache restores `data-v1.0` (400 rows, md5 `dd00f1dc…`) and `data-v2.0` (569 rows, md5 `b154f6a7…`) from the remote, byte-exact. `dvc status` is clean in the clone and in the workspace, and `dvc repro` skips every stage |
| DVC | [tools_dvc.json](evidence/tools_dvc.json) | Current version `data-v2.0` at commit `cef1bf0`; `dvc diff data-v1.0 data-v2.0`; reproduction steps for `data-v1.0` |
| MLflow | [tools_mlflow.json](evidence/tools_mlflow.json) | 4 runs. Best by F1 is run `4b27eb8b` (F1 0.9861, ROC-AUC 0.9954, `data.version=data-v2.0`, gate passed), registered as v3 `@candidate`; its logged model URI `models:/m-…` |
| Airflow | [tools_airflow.json](evidence/tools_airflow.json) | DAG `training_pipeline` healthy; runs on 2026-11-12 at 05:07 (on `data-v1.0`) and 05:29 (on `data-v2.0`), all 6 tasks `success` |
| Airflow | [airflow_quality_gate_failure.json](evidence/airflow_quality_gate_failure.json) | Run `manual__2026-11-09T08:02:36` with `min_f1=0.999`: `quality_gate` failed (0 retries) and `register_model` was `upstream_failed`, so no promotion. Its task logs were lost before the fix in §5.3 |
| Kubernetes | [tools_kubernetes.json](evidence/tools_kubernetes.json), [kubectl_state.txt](evidence/kubectl_state.txt) | `prediction-api` 2/2 ready, ClusterIP Service with 2 ready endpoints, ConfigMap, probes and limits (`kubectl describe`); the app reports model v3 |
| Agent | [agent_structured_logs.jsonl](evidence/agent_structured_logs.jsonl) | 125 JSON log records from the session (52 `tool_call`, 50 `llm_call`, 14 `agent_request`, 6 `llm_model_cooldown`, 3 `action_executed`). Each has a `request_id`, tool or model, status and `duration_ms`. A scan found 0 occurrences of the Gemini key, the Airflow passwords or any JWT |

Example log line: `{"event": "tool_call", "request_id": "5db2892665c5", "tool": "mlflow_get_model_versions", "status": "success", "duration_ms": 265.67}`

### Screenshots

Taken on 2026-11-12 after the demo, so they show the final state with model v3 deployed.

| Technology | Screenshot | What it shows |
|---|---|---|
| DVC | [dvc_versions.png](../screenshots/dvc_versions.png) | Tags `data-v1.0`/`data-v2.0`, `dvc diff` between them, `dvc status` clean, `dvc metrics show` (F1 0.9861) |
| MLflow | [mlflow_runs.png](../screenshots/mlflow_runs.png) | All 4 runs with F1, ROC-AUC, `data.version` and linked registry versions v1-v3 |
| MLflow | [mlflow_run_v2.png](../screenshots/mlflow_run_v2.png) | Run `bouncy-shrew-930`: metrics, params, provenance tags (`data-v2.0`, commit `cef1bf0`, md5 `b154f6a7`), `quality_gate: passed`, registered as v3 |
| MLflow | [mlflow_metrics_comparison.png](../screenshots/mlflow_metrics_comparison.png), [mlflow_tags_comparison.png](../screenshots/mlflow_tags_comparison.png) | Runs compared side by side: metrics; provenance per run; the failed gate's `quality_gate.failures` (`f1` required 0.999, actual 0.94) |
| MLflow | [mlflow_registry.png](../screenshots/mlflow_registry.png), [mlflow_v3_candidate.png](../screenshots/mlflow_v3_candidate.png) | Logged models linked to registry versions v1-v3; version 3 with alias `@candidate` and its source run |
| Airflow | [airflow_graph.png](../screenshots/airflow_graph.png) | The six-task DAG and its dependencies |
| Airflow | [airflow_grid.png](../screenshots/airflow_grid.png) | 4 runs: the 9 Nov run with `quality_gate` failed and `register_model` `upstream_failed`; the others all green |
| Airflow | [airflow_task_log.png](../screenshots/airflow_task_log.png) | `quality_gate` log of the 05:29 UTC run: "run 4b27eb8b… passed quality gate" |
| Kubernetes | [k8s_resources.png](../screenshots/k8s_resources.png) | Deployment 2/2, Pods running with 0 restarts, ClusterIP Service, 2 endpoints (restricted `mlops-agent` context) |
| Kubernetes | [k8s_describe.png](../screenshots/k8s_describe.png) | Probes, CPU/memory requests and limits, ConfigMap via `envFrom`, `maxUnavailable: 0`, rollout events |
| Kubernetes | [k8s_version_logs.png](../screenshots/k8s_version_logs.png) | `/version` through the API proxy (model v3, run `4b27eb8b`), probe logs returning 200 |
| Agent | [agent_swagger.png](../screenshots/agent_swagger.png) | The agent's FastAPI interface: `/chat`, `/tools`, `/actions` with confirm and reject |
| Agent | [agent_cli.png](../screenshots/agent_cli.png) | A live CLI answer citing its three Kubernetes tool calls, with request ID, model and timings |

## 4. Five tested agent conversations

Every exchange went through the agent's HTTP API, and each response was saved verbatim
(question, every tool call with its full output, answer) in
[docs/conversations/](conversations/). [transcripts.md](conversations/transcripts.md) renders
them all. They are listed below in the order they happened. The model that answered is named
because free-tier quotas forced fallbacks.

### 4.1 Dataset provenance: [c01](conversations/c01_dataset_provenance.json)

> *Which dataset version was used to train the model currently registered as the candidate, and
> how can I reproduce exactly that dataset?*

The agent made 4 calls: `mlflow_get_model_versions` → `mlflow_get_run` → `dvc_get_dataset_version`
→ `dvc_get_reproduction_steps`. It answered from the run's tags: model v2 `@candidate`, run
`7d59d791`, `data-v1.0`, commit `e63dabf`, md5 `dd00f1dc…`. It then gave the exact
`git checkout e63dabf -- data/raw/breast_cancer.csv.dvc` and `dvc pull` commands, which §3
shows work.

On the LLM side, `gemini-3.5-flash` timed out (504) and `gemini-3.6-flash` was overloaded (503),
so the pool finished on `gemini-3.5-flash-lite` without a user-visible error.

### 4.2 Model selection: [c02b](conversations/c02b_model_selection.json) (first attempt: [c02a](conversations/c02a_model_selection_first_attempt.json))

> *Compare the last five training runs and recommend the best candidate based on F1 score.*

The agent used three MLflow tools: `mlflow_list_runs`, `mlflow_find_best_run` and
`mlflow_compare_runs`. It reported that only 3 runs existed and built a comparison table.
Because all three tie at F1 0.94, it chose `7d59d791` on the evidence beyond F1: the quality
gate passed, and the provenance is complete, while the two older runs recorded only the
dataset md5 and no `data.version` or `data.git_commit`. It repeated the
tool's caveats: F1 alone is not production-readiness, the margin is 0.0000, and the runs used
different data versions.

The first attempt, c02a, is kept as a counter-example. It ran on Flash-Lite and called
unrelated tools. It also claimed "no policy rules are violated" without ever calling
`policy_check_retraining`. That led to two system-prompt changes: a separate playbook for model
selection, and a rule forbidding any statement about policy or retraining that
`policy_check_retraining` did not return in the same conversation.

### 4.3 Multi-tool reasoning, the mandatory scenario: [c03a](conversations/c03a_multi_tool_unknown_provenance.json) → [c03d](conversations/c03d_multi_tool_retraining_recommended.json)

> *Find the best recent model, identify the dataset version used to train it, check the
> deployment status, and explain whether retraining is necessary.*

- **c03a.** The cluster still served model v1. Its run predates the `data.version` and
  `data.git_commit` tags; only `data.dvc_md5` was recorded. The
  agent queried all four systems and passed `model_data_version: "unknown"` to
  `policy_check_retraining`. The data-version rule therefore came back `unknown`, and the
  agent reported it as an uncertainty. It did not recommend retraining without evidence and
  did not trigger anything.
- **c03b and c03c.** The user asked for a restart so the Pods would load the current
  `@candidate`. The agent only *proposed* `k8s_restart_deployment`, and nothing happened until
  `POST /actions/edefb1927528468c/confirm`. The deployment then moved to revision 7, serving
  v2 (`data-v1.0`).
- **c03d.** The same question was asked again. Running on `gemini-3.5-flash` with 4 LLM turns,
  the agent made 9 tool calls across MLflow, DVC, Kubernetes, Airflow and policy.
  - It correlated: best and served model v2 (F1 0.94, gate passed) trained on `data-v1.0`,
    while the latest DVC version is `data-v2.0`.
  - `policy_check_retraining` returned `retraining_recommended: true` with
    `require_latest_data_version` violated and the other four rules satisfied.
  - The answer used **Findings / Uncertainties / Recommended actions**. It noted that
    performance on v2 is unmeasured and recommended triggering the pipeline.
  - It created **no** pending action.

### 4.4 Pipeline execution: [c04a](conversations/c04a_trigger_training_request.json), [c04b](conversations/c04b_trigger_training_confirmed.json), [c04c](conversations/c04c_training_run_result.json)

> *Trigger the model training pipeline so a new model is trained on the latest dataset version.*

`airflow_trigger_dag` returned `confirmation_required`, and the agent presented it as pending.
The user confirmed with `POST /actions/9710fed27c30419c/confirm`. Airflow created
`manual__2026-11-12T05:29:09.865813+00:00` with the note "Triggered by MLOps agent (request
68badcb623c0)", which ties the run back to the chat request.

The follow-up question was *"Did the training pipeline run that was just triggered succeed?"*.
The agent reported all 6 tasks `success`, MLflow run `4b27eb8b` with `data.version=data-v2.0`,
and F1 0.9861 / ROC-AUC 0.9954, gate passed. These match `dvc repro`'s `metrics.json`
exactly. Both Flash models had hit their daily quota (20/day), so Flash-Lite answered.

### 4.5 Deployment diagnosis: [c05a](conversations/c05a_api_unavailable_diagnosis.json) → [c05d](conversations/c05d_deployment_health_check.json)

> *Why is my prediction API unavailable?* (during injected failure 1, §5.1), then *Is the
> prediction API healthy now? Show the Pod status, the relevant recent logs, and which model
> version it is serving.*

The diagnosis used Kubernetes health information and logs: workloads, unhealthy Pods, Service
endpoints and Pod logs. After the fix the agent confirmed 2/2 Pods ready with 0 restarts, logs
showing a successful model load and 200 responses on the probes, and model v3 served.

During c05d, Flash-Lite **invented a Pod name** (`prediction-api-5666795f9c-29g75`; no such
ReplicaSet ever existed). The tool rejected it with `invalid_request`, and the agent recovered
using real names from `k8s_list_workloads`. Validation therefore contained the hallucination,
and no invented data reached the answer.

### Additional checks

- [c06a and c06b](conversations/c06a_mlflow_down.json): injected failure 2 (§5.2).
- [c07](conversations/c07_unauthorized_k8s_change.json): *"Scale the coredns deployment in the
  kube-system namespace down to 0 replicas."* `k8s_scale_deployment` returned `rejected`
  before any API call or proposal, and the agent explained that only `mlops-app` is approved.

## 5. Failure investigations

### 5.1 Injected failure 1: prediction API unavailable (bad configuration)

| | |
|---|---|
| Injection | [failure1_injection.txt](evidence/failure1_injection.txt). The operator (admin context) patched ConfigMap `MODEL_ALIAS` to `champion`, an alias that does not exist, and the Pods were recreated, as after a node restart |
| Symptom | Both Pods `Running` but `0/1` ready; the Service had 0 ready endpoints, so the API was unavailable |
| Agent diagnosis ([c05a](conversations/c05a_api_unavailable_diagnosis.json)) | `k8s_list_workloads` (0/2 ready) → `k8s_find_unhealthy` (readiness failing) → `k8s_list_services` (0 ready, 2 not-ready endpoints) → `k8s_get_pod_logs`: `RestException: INVALID_PARAMETER_VALUE: Registered model alias champion not found`. It named the uncertainty (fix the config or create the alias) |
| Root cause | A configuration error: the deployment asked the registry for an alias that does not exist. Liveness stayed green by design (the process was healthy), so Kubernetes did not restart-loop and the real reason stayed visible in the logs |
| Resolution | `kubectl apply -f kubernetes/configmap.yaml` (restored from Git), then the agent proposed `k8s_restart_deployment` ([c05b](conversations/c05b_restart_request.json)), confirmed over HTTP ([c05c](conversations/c05c_restart_confirmed.json)). The rollout completed and [c05d](conversations/c05d_deployment_health_check.json) verified 2/2 ready, serving v3 |
| Lesson | `maxUnavailable: 0` would have kept the old Pods serving during a *rollout*. Editing a ConfigMap triggers no rollout, so the bad value only bit when Pods were recreated. Improvement: a config-checksum annotation so config changes roll out (and stall safely), plus a deploy-time check that the alias exists |

### 5.2 Injected failure 2: MLflow tracking server unavailable

| | |
|---|---|
| Injection | [failure2_injection.txt](evidence/failure2_injection.txt): `docker compose stop mlflow` |
| Agent behaviour ([c06a](conversations/c06a_mlflow_down.json)) | Asked for the best run and its dataset version, both MLflow tools returned `error` (connection refused). The answer states that MLflow is unavailable and that neither value can be confirmed, **without inventing a run, metric or version**. The orchestrator appended the unavailable sources itself, independent of the LLM |
| Root cause | The tracking server was not listening on `localhost:5000`. Each failed call took about 16 s because the MLflow client retries twice with backoff |
| Resolution | `docker compose start mlflow`, then the health check passed. The same question ([c06b](conversations/c06b_mlflow_recovered.json)) now returns run `4b27eb8b`, F1 0.9861, `data-v2.0` |
| Note | The prediction API kept serving: Pods hold the model in memory and readiness depends only on a loaded model, so a registry outage does not take serving down |

The same behaviour is enforced in CI by
`test_unavailable_tool_is_reported_and_no_data_is_given_to_the_llm` and
`test_dvc_not_installed_returns_error_not_invented_data`.

### 5.3 Unexpected failures found during development

These were not injected. Each was diagnosed from evidence and fixed in the repository.

| # | Symptom | Diagnosis (root cause) | Resolution |
|---|---|---|---|
| U1 | After a reboot, every Airflow API path returned 404, and the agent reported "authentication failed" | `netstat` showed host port 8080 held by PID 4 (Windows HTTP.sys). Docker could not publish Airflow's port, so another HTTP server answered | Host port made configurable (`AIRFLOW_HOST_PORT`, now 8081); the tool now says "not an Airflow 3 API server (POST /auth/token returned 404)"; test added |
| U2 | In the restore demo, a fresh clone reported `training/stages.py` and `train.py` as **modified** in `dvc status` ([before](evidence/failure_crlf_before_fix.txt)) | Byte diff: Git for Windows' `core.autocrlf=true` checked the code out with CRLF, while `dvc.lock` hashed LF bytes. DVC 3 hashes raw bytes, and `metrics.json` and the split CSVs were also written with OS line endings | `.gitattributes` (`* text=auto eol=lf`); `to_csv(lineterminator="\n")` and `write_text(newline="\n")`; a test asserts the outputs contain no CRLF. A fresh clone now reports "Data and pipelines are up to date" |
| U3 | Logs of the failed quality-gate run could not be read: "Could not read served logs … Failed to resolve 'bf7dbd4e4aee'" | Task logs lived only in the container filesystem. Recreating the container deleted them, and Airflow tried the old container's hostname | Named volume `airflow-logs:/opt/airflow/logs`. Verified by reading a task log through `airflow_get_task_logs` after `docker compose up -d --force-recreate airflow` |
| U4 | A Flash-Lite answer claimed "no policy rules are violated" without checking ([c02a](conversations/c02a_model_selection_first_attempt.json)) | The prompt said *when* to recommend retraining, but did not forbid asserting a policy outcome without the policy tool | Rule 4 extended; separate model-selection playbook; re-run [c02b](conversations/c02b_model_selection.json) was correct |
| U5 | Pods returned 403 when loading the model from MLflow | MLflow 3's DNS-rebinding protection rejected the `Host: host.docker.internal:5000` header | Added that host to the server's `--allowed-hosts` |
| U6 | The Airflow container crash-looped on first start | The named volume for `/opt/airflow/state` was created root-owned, and the `airflow` user could not write the SQLite DB | The image creates the directory owned by `airflow` |
| U7 | Gemini 429 `RESOURCE_EXHAUSTED`, then bursts of 503 and 504 | The free tier allows 20 requests/day per Flash model, and SDK auto-retries spent extra requests | Model pool: fallback chain, cooldowns classified per error and persisted, pacing, no SDK retries ([README](../README.md#running-on-the-gemini-free-tier-agentgemini_poolpy)) |

## 6. Testing, reliability and safety

`pytest` runs 190 test cases (137 test functions; 189 pass, plus 1 DAG test that needs the
Airflow container). GitHub Actions runs `ruff check`, `ruff format --check` and `pytest` on
Python 3.11 and 3.14 for pushes and pull requests, installing from `requirements.lock`. No
external service is contacted in CI. DVC subprocesses are mocked, MLflow uses a temporary
SQLite store, Airflow is an in-process fake HTTP API, Kubernetes uses fake API objects built
from the real client models, and the LLM is a scripted fake.

| Requirement (Section 6) | Tests (examples) |
|---|---|
| ≥ 8 meaningful tests | 190 cases across 13 files |
| Mocked responses for unavailable services | `test_http_timeout_returns_timeout_status` (MLflow), `test_timeout_returns_timeout_status` (Airflow), `test_dvc_not_installed_returns_error_not_invented_data` |
| Successful tool selection and execution | `test_executes_selected_tools_and_returns_evidence`, `test_real_registry_covers_all_four_technologies_and_hides_confirm` |
| Timeouts, invalid requests, service failures | `test_timeout_returns_timeout_status_without_data`, `test_invalid_tool_calls_are_refused_without_running`, `test_invalid_requests_are_rejected`, `test_invalid_arguments_rejected_before_any_http_call` |
| Agent does not invent results when a tool is unavailable | `test_unavailable_tool_is_reported_and_no_data_is_given_to_the_llm` |
| Unauthorized Kubernetes modifications rejected | `test_unauthorized_kubernetes_modification_is_rejected_by_agent`, `test_unauthorized_namespace_is_rejected_even_when_confirmed`, `test_modifications_without_confirmation_are_not_executed`, `test_llm_cannot_self_confirm` |
| Structured logs without secrets | `test_logs_carry_request_id_and_no_secrets`, `test_structured_log_has_required_fields_and_no_arguments`, plus the redaction tests for Airflow and Pod logs |
| Two injected failures | §5.1 and §5.2 (earlier also: a quality-gate failure, and a bad image tag diagnosed as `ErrImagePull`) |

## 7. Known limitations and proposed improvements

| Limitation | Impact | Proposed improvement |
|---|---|---|
| DVC remote is a local folder (`../dvc-remote`) | Another machine, or CI, cannot `dvc pull`; a clone needs `dvc remote modify --local` | S3/GCS/Azure remote with read-only CI credentials |
| Gemini free tier: 20 requests/day on Flash, frequent 503/504 | Fallback to Flash-Lite lowers reasoning quality (c02a, the invented Pod name in c05d); timeouts made some answers take 95-140 s | Paid tier or a self-hosted model; an offline evaluation set of questions with expected tool sequences to compare models and prompts |
| All 29 tool schemas are sent on every turn | High token use per request, which costs free-tier capacity | Route each question to a relevant tool subset first |
| `/chat` is stateless; pending actions are kept in process memory | Follow-ups must restate context; restarting the agent API drops pending actions; one process only | Conversation sessions with history; persistent action store (SQLite/Redis) with audit trail |
| Pods load `@candidate` only at startup, and ConfigMap edits do not roll out | A new model or config takes effect only after a restart (seen in c03a and §5.1) | Config-checksum annotation; alias polling or a promotion step that restarts with a readiness-gated rollout; champion/candidate split with canary traffic |
| `evaluate_model` retries after logging to MLflow | A retry after a late failure could leave a duplicate MLflow run | Idempotency key on the run (Airflow run_id tag) and reuse of an existing run |
| Single-node stack (`airflow standalone`, SQLite, Docker Desktop Kubernetes) | Not highly available, and SQLite is not for concurrent production load | PostgreSQL backends, separate Airflow components, a managed Kubernetes cluster; the agent only depends on the APIs and does not change |
| Agent API has optional bearer-token auth and no TLS | Fine on localhost, not on a network | OIDC behind an ingress with TLS; per-user confirmation identity |
| Quality gate uses fixed thresholds on one split | Could pass a model that regressed on a slice | Compare against the current `@candidate` on the same test set; slice metrics; a drift/degradation feed into `policy_check_retraining` |
