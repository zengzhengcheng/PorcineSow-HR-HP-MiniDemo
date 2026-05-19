"""Train the gradient-boosted tree baseline with 5-fold cross-validation.

Usage:
    python code/train_baseline.py --config code/config_baseline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from utils.io import FEATURE_COLUMNS, TARGET_COLUMN, load_config, load_dataset, split_xy

ESTIMATORS = {"HistGradientBoostingRegressor": HistGradientBoostingRegressor}


def build_model(cfg_model: dict):
    name = cfg_model["estimator"]
    if name not in ESTIMATORS:
        raise ValueError(f"unsupported estimator: {name}")
    return ESTIMATORS[name](**cfg_model.get("params", {}))


def cross_validate(x: np.ndarray, y: np.ndarray, cfg: dict):
    kf = KFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )
    rows = []
    oof_pred = np.zeros_like(y)
    oof_mask = np.zeros_like(y, dtype=bool)
    fitted_models = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
        model = build_model(cfg["model"])
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
        oof_pred[test_idx] = pred
        oof_mask[test_idx] = True
        fitted_models.append(model)
    return rows, oof_pred, oof_mask, fitted_models


def save_summary(rows: list, out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    mean = df[["r2", "mae", "rmse"]].mean()
    std = df[["r2", "mae", "rmse"]].std()
    df.to_csv(out_dir / "cv_summary.csv", index=False)

    lines = ["# 5-fold cross-validation summary", ""]
    lines.append("| fold | n_train | n_test | R^2 | MAE | RMSE |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['n_train']} | {r['n_test']} | "
            f"{r['r2']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} |"
        )
    lines.append(
        f"| mean | - | - | {mean['r2']:.4f} | {mean['mae']:.4f} | {mean['rmse']:.4f} |"
    )
    lines.append(
        f"| std  | - | - | {std['r2']:.4f} | {std['mae']:.4f} | {std['rmse']:.4f} |"
    )
    (out_dir / "cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("observed HP (kcal / 5 min)")
    ax.set_ylabel("predicted HP (kcal / 5 min)")
    ax.set_title("Out-of-fold predictions, 5-fold CV")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(
    fitted_models: list, x: np.ndarray, y: np.ndarray, out_path: Path
) -> None:
    rng = np.random.default_rng(42)
    importances = np.zeros(len(FEATURE_COLUMNS))
    for model in fitted_models:
        result = permutation_importance(
            model, x, y, n_repeats=5, random_state=rng.integers(0, 2**31), n_jobs=1
        )
        importances += result.importances_mean
    importances /= len(fitted_models)
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.barh(np.array(FEATURE_COLUMNS)[order], importances[order])
    ax.set_xlabel("permutation importance (mean R^2 drop)")
    ax.set_title("Feature importance (averaged over folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="5-fold CV baseline for sow 5-min HP prediction")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None, help="override data_path from config")
    ap.add_argument("--out", type=Path, default=None, help="override output_dir from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_path = Path(args.data) if args.data else (ROOT / cfg["data_path"])
    out_dir = Path(args.out) if args.out else (ROOT / cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data: {data_path}")
    df = load_dataset(data_path)
    print(f"rows: {len(df)}  features: {len(FEATURE_COLUMNS)}  target: {TARGET_COLUMN}")
    x, y = split_xy(df)

    rows, oof_pred, oof_mask, fitted_models = cross_validate(x, y, cfg)
    summary = save_summary(rows, out_dir)

    r2_mean = summary["r2"].mean()
    r2_std = summary["r2"].std()
    mae_mean = summary["mae"].mean()
    print(f"5-fold R^2: {r2_mean:.4f} +/- {r2_std:.4f}")
    print(f"5-fold MAE: {mae_mean:.4f} kcal/5min")

    plot_pred_vs_true(y[oof_mask], oof_pred[oof_mask], out_dir / "kfold_pred_vs_true.png")
    plot_feature_importance(fitted_models, x, y, out_dir / "feature_importance.png")
    print(f"results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
