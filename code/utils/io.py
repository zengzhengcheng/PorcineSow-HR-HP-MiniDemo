"""I/O and preprocessing helpers for the minimal sow dataset."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


FEATURE_COLUMNS: List[str] = [
    "hr_mean",
    "hr_sd",
    "rmssd",
    "pnn50",
    "gyro_mean",
    "angle_sd",
    "dz_sd",
    "active_fraction",
    "frac_rest",
    "weight_kg",
    "hour_of_day",
    "is_fasting_day",
]
TARGET_COLUMN: str = "hp_kcal_5min"
ID_COLUMNS: List[str] = ["animal_id", "period_in_animal", "day_in_period", "time_offset_s"]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return df


def split_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return x, y
