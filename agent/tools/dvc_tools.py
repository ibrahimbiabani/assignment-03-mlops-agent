"""DVC data-management tools (Task 1).

All provenance comes from DVC metadata (.dvc files, `dvc status`, `dvc diff`)
and Git history. Only the read-only commands in APPROVED_COMMANDS can run.
"""

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agent.tools.base import InvalidToolRequest, ToolError, ToolRejected, ToolTimeout, tool

APPROVED_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("dvc", "status"),
        ("dvc", "diff"),
        ("git", "log"),
        ("git", "tag"),
        ("git", "show"),
        ("git", "rev-parse"),
    }
)
DEFAULT_TIMEOUT_S = 30
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^-]{0,99}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_SEP = "\x1f"


def repo_root() -> Path:
    return Path(os.environ.get("MLOPS_REPO_PATH", ".")).resolve()


def run_command(args: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> str:
    """Run an approved DVC/Git command without a shell and return its stdout."""
    if tuple(args[:2]) not in APPROVED_COMMANDS:
        raise ToolRejected(f"command not approved: {' '.join(args[:2])}")
    try:
        proc = subprocess.run(
            args,
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"'{args[0]}' executable not found; is it installed?") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeout(f"'{' '.join(args[:2])}' timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise ToolError(f"'{' '.join(args[:2])}' failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _validate_rev(rev: str) -> str:
    if not isinstance(rev, str) or not _REV_RE.match(rev):
        raise InvalidToolRequest(f"invalid git revision: {rev!r}")
    return rev


def _dvc_file_for(path: str) -> str:
    """Map 'data/x.csv' or 'data/x.csv.dvc' to the repo-relative .dvc file path."""
    if not path or not isinstance(path, str):
        raise InvalidToolRequest("dataset path is required")
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or path.startswith("-") or _DRIVE_RE.match(path):
        raise InvalidToolRequest(f"path must be relative to the repository: {path!r}")
    return str(p) if p.suffix == ".dvc" else f"{p}.dvc"


def _parse_dvc_file(text: str, dvc_file: str) -> list[dict[str, Any]]:
    meta = yaml.safe_load(text) or {}
    parent = PurePosixPath(dvc_file).parent
    return [
        {
            "path": str(parent / out["path"]),
            "dvc_file": dvc_file,
            "md5": out.get("md5"),
            "size": out.get("size"),
            "nfiles": out.get("nfiles"),
        }
        for out in meta.get("outs", [])
    ]


@tool("dvc_list_tracked_datasets")
def list_tracked_datasets() -> dict[str, Any]:
    """List every dataset tracked by a .dvc file, with its content hash."""
    root = repo_root()
    datasets: list[dict[str, Any]] = []
    for f in sorted(root.rglob("*.dvc")):
        rel = f.relative_to(root)
        if rel.parts[0] in {".dvc", ".git"}:
            continue
        datasets.extend(_parse_dvc_file(f.read_text(encoding="utf-8"), rel.as_posix()))
    return {"datasets": datasets, "count": len(datasets)}


@tool("dvc_get_dataset_version")
def get_dataset_version(path: str) -> dict[str, Any]:
    """Current hash of a dataset plus the Git commit and tags that recorded it."""
    dvc_file = _dvc_file_for(path)
    full = repo_root() / dvc_file
    if not full.is_file():
        raise InvalidToolRequest(f"no DVC tracking file found: {dvc_file}")
    outs = _parse_dvc_file(full.read_text(encoding="utf-8"), dvc_file)

    log = run_command(
        ["git", "log", "-1", f"--format=%H{_SEP}%h{_SEP}%cI{_SEP}%s", "--", dvc_file]
    ).strip()
    commit = None
    if log:
        sha, short, date, subject = log.split(_SEP)
        tags = run_command(["git", "tag", "--points-at", sha]).split()
        commit = {"sha": sha, "short_sha": short, "date": date, "message": subject, "tags": tags}
    return {"dvc_file": dvc_file, "outs": outs, "commit": commit}


@tool("dvc_check_data_changes")
def check_data_changes() -> dict[str, Any]:
    """Report whether tracked data in the workspace differs from what DVC recorded."""
    status = json.loads(run_command(["dvc", "status", "--json"]) or "{}")
    return {"up_to_date": not status, "changes": status}


@tool("dvc_compare_dataset_versions")
def compare_dataset_versions(rev_a: str, rev_b: str) -> dict[str, Any]:
    """Compare tracked data between two Git revisions using `dvc diff`."""
    diff = json.loads(
        run_command(["dvc", "diff", "--json", _validate_rev(rev_a), _validate_rev(rev_b)]) or "{}"
    )
    summary = {k: len(v) for k, v in diff.items() if isinstance(v, list)}
    return {"rev_a": rev_a, "rev_b": rev_b, "summary": summary, "diff": diff}


@tool("dvc_get_reproduction_steps")
def get_reproduction_steps(path: str, rev: str) -> dict[str, Any]:
    """Explain how to restore a dataset exactly as it was at `rev`.

    The steps are built from the .dvc file recorded at that revision,
    so they only reference a version that actually exists. Nothing is
    checked out by this tool.
    """
    dvc_file = _dvc_file_for(path)
    sha = run_command(["git", "rev-parse", "--verify", f"{_validate_rev(rev)}^{{commit}}"]).strip()
    recorded = run_command(["git", "show", f"{sha}:{dvc_file}"])
    outs = _parse_dvc_file(recorded, dvc_file)
    return {
        "dvc_file": dvc_file,
        "commit": sha,
        "outs": outs,
        "steps": [
            f"git checkout {sha} -- {dvc_file}",
            f"dvc pull {dvc_file}  # fetches from the remote and checks out the data",
        ],
    }
