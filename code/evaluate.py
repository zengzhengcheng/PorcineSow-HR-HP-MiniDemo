"""Re-score a CSV with the same 5-fold schedule used by train_baseline.py.

Useful for sanity-checking that a downstream environment matches the
reference fold-level metrics shipped in results/cv_summary.csv.

Usage:
    python code/evaluate.py --config code/config_baseline.yaml
    python code/evaluate.py --config code/config_baseline.yaml --reference results/cv_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from utils.io import load_config, load_dataset, split_xy


def fold_metrics(x: np.ndarray, y: np.ndarray, cfg: dict) -> pd.DataFrame:
    kf = KFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
        model = HistGradientBoostingRegressor(**cfg["model"]["params"])
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "r2": r2_score(y[test_idx], pred),
                "mae": mean_absolute_error(y[test_idx], pred),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
            }
        )
    return pd.DataFrame(rows)


def compare(current: pd.DataFrame, reference_path: Path) -> None:
    ref = pd.read_csv(reference_path)
    if set(ref["fold"]) != set(current["fold"]):
        print("warning: fold sets differ; cannot compare per-fold")
        return
    merged = ref.merge(current, on="fold", suffixes=("_ref", "_now"))
    print("\nper-fold delta vs reference (|delta| > 0.005 highlighted):")
    print(f"{'fold':>4} {'r2_ref':>8} {'r2_now':>8} {'dR2':>8}  "
          f"{'mae_ref':>8} {'mae_now':>8} {'dMAE':>8}")
    for _, r in merged.iterrows():
        dr = r["r2_now"] - r["r2_ref"]
        dm = r["mae_now"] - r["mae_ref"]
        mark = "  *" if abs(dr) > 0.005 else ""
        print(f"{int(r['fold']):>4} {r['r2_ref']:>8.4f} {r['r2_now']:>8.4f} "
              f"{dr:>+8.4f}  {r['mae_ref']:>8.4f} {r['mae_now']:>8.4f} "
              f"{dm:>+8.4f}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-score with same fold schedule")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--reference", type=Path, default=None,
                    help="reference cv_summary.csv to compare against")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_path = Path(args.data) if args.data else (ROOT / cfg["data_path"])
    df = load_dataset(data_path)
    x, y = split_xy(df)

    metrics = fold_metrics(x, y, cfg)
    print(metrics.to_string(index=False))
    print(f"\nmean R^2: {metrics['r2'].mean():.4f}  "
          f"mean MAE: {metrics['mae'].mean():.4f}")

    if args.reference is not None:
        compare(metrics, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
