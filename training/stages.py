"""Offline pipeline stages run by `dvc repro` (see dvc.yaml and params.yaml).

    python -m training.stages preprocess --data data/raw/breast_cancer.csv --out data/processed
    python -m training.stages train --train data/processed/train.csv --model models/model.joblib
    python -m training.stages evaluate --model models/model.joblib --test data/processed/test.csv

This pipeline makes data -> split -> model -> metrics reproducible and versioned by DVC
(dvc.lock pins the hash of every input and output). It has no MLflow or registry side
effects: logging, the quality gate and promotion happen only in the Airflow DAG.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from training import train


def _params(path: Path, section: str) -> dict[str, Any]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(section, {})


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="DVC pipeline stages")
    p.add_argument("--params", type=Path, default=Path("params.yaml"))
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("preprocess", help="validate the raw data, then split it")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("train")
    s.add_argument("--train", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    s = sub.add_parser("evaluate")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--test", type=Path, required=True)
    s.add_argument("--metrics", type=Path, default=Path("metrics.json"))
    args = p.parse_args(argv)

    if args.stage == "preprocess":
        cfg = _params(args.params, "preprocess")
        train.validate_dataset(args.data)
        train.preprocess(
            args.data, args.out, test_size=cfg.get("test_size", 0.2), seed=cfg.get("seed", 42)
        )
    elif args.stage == "train":
        train.train_model(args.train, args.model, _params(args.params, "train"))
    else:
        metrics = train.compute_metrics(joblib.load(args.model), pd.read_csv(args.test))
        args.metrics.write_text(
            json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
