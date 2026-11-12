#!/usr/bin/env bash
# Reproducible dataset checkout demo (Task 1). Writes a command-by-command transcript.
#
#   bash scripts/dvc_restore_demo.sh docs/evidence/01_dvc_reproducible_checkout.txt
#
# Part A clones the committed HEAD into a temporary folder (empty DVC cache) and restores
# data-v1.0 and data-v2.0 from the DVC remote only. Part B restores v1 in this workspace and
# returns to the current version. Part C shows `dvc repro` has nothing to rerun.
# Expects the remote at ../dvc-remote (the committed default) and dvc on PATH or in .venv.
set -euo pipefail
OUT="${1:?usage: dvc_restore_demo.sh OUTPUT_FILE}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
export PATH="$REPO/.venv/Scripts:$REPO/.venv/bin:$PATH"
REMOTE="$(cd "$REPO/../dvc-remote" && (pwd -W 2>/dev/null || pwd))"
CLONE="$(mktemp -d)/clone"

run() { echo; echo "\$ $*"; "$@" 2>&1; }
describe() {
  echo; echo "\$ (rows and md5 of data/raw/breast_cancer.csv)"
  python -c "import pandas as p; print('rows:', len(p.read_csv('data/raw/breast_cancer.csv')))"
  md5sum data/raw/breast_cancer.csv
  echo "md5 recorded in .dvc: $(grep -m1 'md5:' data/raw/breast_cancer.csv.dvc | awk '{print $NF}')"
}

{
echo "# Part A: fresh clone, data restored only from the DVC remote (empty local cache)"
echo; echo "\$ git clone <this repository> <temporary folder>"
git clone --quiet "$REPO" "$CLONE"
cd "$CLONE"
run git log --oneline -3
run git tag -l
# The committed remote URL is relative to the original checkout; point the clone at it.
echo; echo "\$ dvc remote modify --local localremote url <absolute path of ../dvc-remote>"
dvc remote modify --local localremote url "$REMOTE"
run dvc list . data/raw --dvc-only

echo; echo "## Restore data-v1.0"
run git checkout --quiet data-v1.0
run git rev-parse --short HEAD
run dvc pull
describe
run dvc status

echo; echo "## Back to the current version (main; dataset data-v2.0)"
run git checkout --quiet main
run git rev-parse --short HEAD
run git describe --tags --abbrev=0 HEAD
run dvc pull
describe
run dvc status

echo; echo "## What changed between the versions"
run dvc diff data-v1.0 data-v2.0
run git log --oneline data-v1.0..data-v2.0 -- data/raw/breast_cancer.csv.dvc

echo; echo "# Part B: in the working repository (local cache), restore v1 and return"
cd "$REPO"
run git checkout data-v1.0 -- data/raw/breast_cancer.csv.dvc
run dvc checkout data/raw/breast_cancer.csv.dvc
describe
run git checkout HEAD -- data/raw/breast_cancer.csv.dvc
run dvc checkout data/raw/breast_cancer.csv.dvc
describe
run dvc status

echo; echo "# Part C: the DVC pipeline is reproducible (nothing to rerun on unchanged inputs)"
run dvc repro
run dvc metrics show
} > "$OUT"
rm -rf "$(dirname "$CLONE")"
echo "wrote $OUT"
