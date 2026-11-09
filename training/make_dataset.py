"""Create the raw dataset tracked by DVC (Wisconsin breast-cancer data from scikit-learn).

python -m training.make_dataset                  # all 569 rows
python -m training.make_dataset --rows 400       # a smaller earlier version
"""

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/breast_cancer.csv"))
    p.add_argument("--rows", type=int, default=None)
    args = p.parse_args()

    df = load_breast_cancer(as_frame=True).frame
    df.columns = [c.replace(" ", "_") for c in df.columns]
    if args.rows:
        df = df.sample(n=args.rows, random_state=0).sort_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
