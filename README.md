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

## Development log

| Step | Change |
|---|---|
| 1 | Project scaffolding following the assignment's recommended layout |
| 2 | Shared `ToolResult` schema, `@tool` wrapper with structured logging, DVC tools (Task 1) and 17 tests |
| 3 | Local `.venv` with editable install (`.[dev,dvc]`); ruff lint/format and pytest all passing |
