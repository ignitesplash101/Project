from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRAIN_END = "2019-Q4"
VAL_END = "2021-Q4"


def period_to_order(period_key: str) -> int:
    year, quarter = period_key.split("-Q")
    return int(year) * 4 + (int(quarter) - 1)


def period_to_float(period_key: str) -> float:
    return period_to_order(period_key) / 4.0


def evaluate_sets(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def temporal_split(df: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END):
    order_series = df["PeriodKey"].apply(period_to_order)
    train_mask = order_series <= period_to_order(train_end)
    val_mask = (order_series > period_to_order(train_end)) & (order_series <= period_to_order(val_end))
    test_mask = order_series > period_to_order(val_end)
    return df[train_mask], df[val_mask], df[test_mask]


def add_temporal_features(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    df = df.sort_values([group_col, "PeriodKey"]).copy()
    df["PeriodNum"] = df["PeriodKey"].astype(str).apply(period_to_float)
    for lag in [1, 4]:
        df[f"{target_col}_lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    df[f"{target_col}_growth_qoq"] = (df[target_col] / df[f"{target_col}_lag1"] - 1) * 100
    df[f"{target_col}_growth_yoy"] = (df[target_col] / df[f"{target_col}_lag4"] - 1) * 100
    df[f"{target_col}_ma4q"] = (
        df.groupby(group_col)[target_col]
        .rolling(window=4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{target_col}_std4q"] = (
        df.groupby(group_col)[target_col]
        .rolling(window=4, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df
