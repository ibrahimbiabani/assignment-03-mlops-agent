"""DVC tool tests. Git/DVC are never executed: subprocess.run is replaced by a fake."""

import json
import logging
import subprocess

import pytest

from agent.schemas import ToolStatus
from agent.tools import dvc_tools

DVC_FILE = """\
outs:
- md5: 3f2a9c1e0b7d4e5f6a7b8c9d0e1f2a3b
  size: 2048
  hash: md5
  path: iris.csv
"""
SHA = "a4c92e1f00000000000000000000000000000000"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_REPO_PATH", str(tmp_path))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "iris.csv.dvc").write_text(DVC_FILE)
    # Files under DVC's own cache dir must be ignored.
    (tmp_path / ".dvc").mkdir()
    (tmp_path / ".dvc" / "ignored.dvc").write_text(DVC_FILE)
    return tmp_path


@pytest.fixture
def fake_run(monkeypatch):
    """Map a command prefix (tuple) to stdout, an exception, or a non-zero exit."""
    responses: dict[tuple[str, ...], object] = {}
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        for prefix, resp in responses.items():
            if tuple(args[: len(prefix)]) == prefix:
                if isinstance(resp, BaseException):
                    raise resp
                code, out, err = resp if isinstance(resp, tuple) else (0, resp, "")
                return subprocess.CompletedProcess(args, code, out, err)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(dvc_tools.subprocess, "run", run)
    return responses, calls


def test_list_tracked_datasets_reads_dvc_files(repo):
    result = dvc_tools.list_tracked_datasets()

    assert result.status == ToolStatus.SUCCESS
    assert result.data["count"] == 1
    ds = result.data["datasets"][0]
    assert ds["path"] == "data/iris.csv"
    assert ds["md5"] == "3f2a9c1e0b7d4e5f6a7b8c9d0e1f2a3b"


def test_get_dataset_version_returns_hash_commit_and_tags(repo, fake_run):
    responses, _ = fake_run
    responses[("git", "log")] = f"{SHA}\x1fa4c92e1\x1f2026-11-01T10:00:00+00:00\x1fAdd iris v2\n"
    responses[("git", "tag")] = "data-v2.1\n"

    result = dvc_tools.get_dataset_version("data/iris.csv")

    assert result.ok
    assert result.data["outs"][0]["md5"] == "3f2a9c1e0b7d4e5f6a7b8c9d0e1f2a3b"
    assert result.data["commit"]["short_sha"] == "a4c92e1"
    assert result.data["commit"]["tags"] == ["data-v2.1"]


def test_check_data_changes_clean_and_modified(repo, fake_run):
    responses, _ = fake_run
    responses[("dvc", "status")] = "{}"
    assert dvc_tools.check_data_changes().data["up_to_date"] is True

    changed = {"data/iris.csv.dvc": [{"changed outs": {"data/iris.csv": "modified"}}]}
    responses[("dvc", "status")] = json.dumps(changed)
    result = dvc_tools.check_data_changes()
    assert result.data["up_to_date"] is False
    assert result.data["changes"] == changed


def test_compare_dataset_versions_summarises_dvc_diff(repo, fake_run):
    responses, calls = fake_run
    responses[("dvc", "diff")] = json.dumps(
        {"added": [], "deleted": [], "modified": [{"path": "data/iris.csv"}], "renamed": []}
    )

    result = dvc_tools.compare_dataset_versions("data-v1.0", "data-v2.1")

    assert result.ok
    assert result.data["summary"]["modified"] == 1
    assert calls[-1] == ["dvc", "diff", "--json", "data-v1.0", "data-v2.1"]


def test_reproduction_steps_use_recorded_revision(repo, fake_run):
    responses, _ = fake_run
    responses[("git", "rev-parse")] = SHA + "\n"
    responses[("git", "show")] = DVC_FILE

    result = dvc_tools.get_reproduction_steps("data/iris.csv", "data-v2.1")

    assert result.ok
    assert result.data["commit"] == SHA
    assert result.data["steps"][0] == f"git checkout {SHA} -- data/iris.csv.dvc"


@pytest.mark.parametrize("rev", ["--output=/tmp/x", "", "a b", "x;rm -rf /"])
def test_invalid_revision_is_rejected_before_running(repo, fake_run, rev):
    _, calls = fake_run

    result = dvc_tools.compare_dataset_versions(rev, "HEAD")

    assert result.status == ToolStatus.INVALID_REQUEST
    assert result.data is None
    assert calls == []


@pytest.mark.parametrize("path", ["../secrets.csv", "/etc/passwd", "C:/Windows/x"])
def test_path_outside_repo_is_rejected(repo, path):
    result = dvc_tools.get_dataset_version(path)
    assert result.status == ToolStatus.INVALID_REQUEST


def test_unapproved_command_is_rejected(repo, fake_run):
    _, calls = fake_run
    with pytest.raises(dvc_tools.ToolRejected):
        dvc_tools.run_command(["dvc", "gc", "--force"])
    assert calls == []


def test_timeout_returns_timeout_status_without_data(repo, fake_run):
    responses, _ = fake_run
    responses[("dvc", "status")] = subprocess.TimeoutExpired(["dvc", "status"], 30)

    result = dvc_tools.check_data_changes()

    assert result.status == ToolStatus.TIMEOUT
    assert result.data is None
    assert "timed out" in result.error


def test_dvc_not_installed_returns_error_not_invented_data(repo, fake_run):
    responses, _ = fake_run
    responses[("dvc", "status")] = FileNotFoundError("dvc")

    result = dvc_tools.check_data_changes()

    assert result.status == ToolStatus.ERROR
    assert result.data is None
    assert "not found" in result.error


def test_command_failure_surfaces_stderr(repo, fake_run):
    responses, _ = fake_run
    responses[("dvc", "diff")] = (1, "", "ERROR: unknown revision 'nope'")

    result = dvc_tools.compare_dataset_versions("nope", "HEAD")

    assert result.status == ToolStatus.ERROR
    assert "unknown revision" in result.error


def test_structured_log_has_required_fields_and_no_arguments(repo, fake_run, caplog):
    responses, _ = fake_run
    responses[("dvc", "diff")] = "{}"
    caplog.set_level(logging.INFO, logger="mlops_agent.tools")

    dvc_tools.compare_dataset_versions("secret-token-abc", "HEAD")

    entry = json.loads(caplog.records[-1].getMessage())
    assert entry["tool"] == "dvc_compare_dataset_versions"
    assert entry["status"] == "success"
    assert {"request_id", "duration_ms"} <= entry.keys()
    assert "secret-token-abc" not in caplog.text
