#!/usr/bin/env python
# coding: utf-8

# # 02 - Hedonic Indices and Diagnostics
# 

# ## 0. Environment Setup

# In[1]:


from __future__ import annotations

from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

NOTEBOOK_DIR = Path.cwd()
DATA_DIR = NOTEBOOK_DIR

def period_to_order(period_key: str) -> int:
    year, quarter = period_key.split("-Q")
    return int(year) * 4 + (int(quarter) - 1)

def period_to_float(period_key: str) -> float:
    return period_to_order(period_key) / 4.0

def evaluate_sets(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }

def temporal_split(df: pd.DataFrame, train_end: str, val_end: str):
    order_series = df["PeriodKey"].apply(period_to_order)
    train_mask = order_series <= period_to_order(train_end)
    val_mask = (order_series > period_to_order(train_end)) & (order_series <= period_to_order(val_end))
    test_mask = order_series > period_to_order(val_end)
    return df[train_mask], df[val_mask], df[test_mask]

MODEL_FACTORIES = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
}


# In[2]:


TRAIN_END = "2019-Q4"
VAL_END = "2021-Q4"
TRAIN_END_ORDER = period_to_order(TRAIN_END)
VAL_END_ORDER = period_to_order(VAL_END)


# ## 1. Load processed datasets

# In[3]:


MAIN_FEATURES_PATH = DATA_DIR / "main_features.parquet"
MESH_FEATURES_PATH = DATA_DIR / "mesh_quarter_features.csv"

main_df = pd.read_parquet(MAIN_FEATURES_PATH)
mesh_panel_raw = pd.read_csv(MESH_FEATURES_PATH)

main_df["Mesh250m"] = main_df["Mesh250m"].astype(str)
main_df.loc[main_df["Mesh250m"].str.lower() == "nan", "Mesh250m"] = np.nan
main_df["WardName"] = main_df["Municipality_en"].fillna(main_df["Municipality"]).fillna("Unknown")

mesh_panel_raw["Mesh250m"] = mesh_panel_raw["Mesh250m"].astype(str)
mesh_panel_raw.loc[mesh_panel_raw["Mesh250m"].str.lower() == "nan", "Mesh250m"] = np.nan

print(f"Transactions loaded: {len(main_df):,}")
print(f"Mesh quarters loaded: {len(mesh_panel_raw):,}")
print(f"Time span: {main_df['PeriodKey'].min()} -> {main_df['PeriodKey'].max()}")


# ## 2. Hedonic price indices

# In[4]:


hedonic_df = main_df[[
    "LogPrice", "AreaSqM", "BuildingAge", "Municipality", "Municipality_en", "PeriodKey",
    "Structure", "Structure_en", "Type", "Type_en", "Use", "Use_en", "Mesh250m", "WardName"
]].copy()

hedonic_df["Ward"] = hedonic_df["WardName"]
hedonic_df.drop(columns=["WardName"], inplace=True)

hedonic_df["StructureFeat"] = hedonic_df["Structure_en"].fillna(hedonic_df["Structure"]).fillna("Unknown")
hedonic_df["TypeFeat"] = hedonic_df["Type_en"].fillna(hedonic_df["Type"]).fillna("Unknown")
hedonic_df["UseFeat"] = hedonic_df["Use_en"].fillna(hedonic_df["Use"]).fillna("Unknown")

period_levels_all = sorted(hedonic_df["PeriodKey"].astype(str).unique())
ward_levels_all = sorted(hedonic_df["Ward"].astype(str).unique())
structure_levels_all = sorted(hedonic_df["StructureFeat"].astype(str).unique())
type_levels_all = sorted(hedonic_df["TypeFeat"].astype(str).unique())
use_levels_all = sorted(hedonic_df["UseFeat"].astype(str).unique())

categorical_levels = {
    "PeriodKey": (period_levels_all, True),
    "Ward": (ward_levels_all, False),
    "StructureFeat": (structure_levels_all, False),
    "TypeFeat": (type_levels_all, False),
    "UseFeat": (use_levels_all, False),
}

for col, (cats, ordered) in categorical_levels.items():
    hedonic_df[col] = pd.Categorical(hedonic_df[col], categories=cats, ordered=ordered)

print(f"Hedonic sample after cleaning: {len(hedonic_df):,}")


# In[5]:


def fit_hedonic_model(data: pd.DataFrame):
    return smf.ols(
        "LogPrice ~ np.log(AreaSqM) + BuildingAge + C(Ward) + C(PeriodKey) + C(TypeFeat) + C(StructureFeat) + C(UseFeat)",
        data=data
    ).fit()


def make_hedonic_predictions(model, data: pd.DataFrame) -> pd.DataFrame:
    preds = data.copy()
    preds["PredictedLogPrice"] = model.predict(data)
    preds["PredictedPrice"] = np.exp(preds["PredictedLogPrice"])
    return preds


def build_index_panel(panel_df: pd.DataFrame,
                      group_col: str,
                      value_col: str,
                      count_col: str | None = None,
                      min_periods: int = 4,
                      min_total_obs: int | None = 20,
                      group_label: str | None = None,
                      fallback_min_periods: int | None = None,
                      fallback_min_total_obs: int | None = None,
                      allow_relax: bool = True) -> pd.DataFrame:
    """Aggregate fitted values into an index while enforcing coverage thresholds."""
    records: list[pd.DataFrame] = []
    group_label = group_label or group_col
    default_relaxed_periods = max(2, min_periods - 1)
    default_relaxed_total = None if min_total_obs is None else max(4, min_total_obs // 2)
    relaxed_periods = fallback_min_periods if fallback_min_periods is not None else default_relaxed_periods
    relaxed_total = fallback_min_total_obs if fallback_min_total_obs is not None else default_relaxed_total

    for group, group_df in panel_df.groupby(group_col):
        unique_periods = group_df["PeriodKey"].nunique()
        total_obs = float(group_df[count_col].sum()) if count_col and count_col in group_df.columns else None

        def passes_requirements(min_period_threshold: int, min_obs_threshold: int | None) -> bool:
            if unique_periods < min_period_threshold:
                return False
            if min_obs_threshold is None or total_obs is None:
                return True
            return total_obs >= min_obs_threshold

        if not passes_requirements(min_periods, min_total_obs):
            if not allow_relax or not passes_requirements(relaxed_periods, relaxed_total):
                continue

        group_df = group_df.copy()
        group_df["PeriodOrder"] = group_df["PeriodKey"].astype(str).apply(period_to_order)
        group_df = group_df.sort_values("PeriodOrder")

        base_series = group_df[value_col]
        valid_mask = np.isfinite(base_series) & (base_series > 0)
        if not valid_mask.any():
            continue
        anchor_idx = base_series[valid_mask].index[0]
        base_value = float(base_series.loc[anchor_idx])
        base_period_key = group_df.loc[anchor_idx, "PeriodKey"]

        group_df["Index"] = group_df[value_col] / base_value * 100.0
        group_df["BasePeriod"] = base_period_key
        group_df[group_label] = group

        keep_cols = [group_label, "PeriodKey", value_col, "Index", "BasePeriod", "PeriodOrder"]
        if count_col and count_col in group_df.columns:
            keep_cols.append(count_col)
        records.append(group_df[keep_cols])

    if not records:
        return pd.DataFrame(columns=[group_label, "PeriodKey", value_col, "Index", "BasePeriod", "PeriodOrder"])

    return pd.concat(records, ignore_index=True).sort_values([group_label, "PeriodOrder"]).reset_index(drop=True)


def aggregate_predictions(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ward = (
        pred_df.groupby(["Ward", "PeriodKey"], dropna=False)
        .agg(PredictedPrice=("PredictedPrice", "median"), Transactions=("PredictedPrice", "size"))
        .reset_index()
    )
    mesh = (
        pred_df.dropna(subset=["Mesh250m"])
        .groupby(["Mesh250m", "PeriodKey"], dropna=False)
        .agg(PredictedPrice=("PredictedPrice", "median"), Transactions=("PredictedPrice", "size"))
        .reset_index()
    )
    return ward, mesh


# In[6]:


hedonic_model_full = fit_hedonic_model(hedonic_df)
hedonic_predictions_full = make_hedonic_predictions(hedonic_model_full, hedonic_df)

baseline_period = period_levels_all[0]
hedonic_index_overall = pd.DataFrame([
    {
        "PeriodKey": period,
        "Index": float(np.exp(hedonic_model_full.params.get(f"C(PeriodKey)[T.{period}]", 0.0)) * 100.0) if period != baseline_period else 100.0,
        "Effect": hedonic_model_full.params.get(f"C(PeriodKey)[T.{period}]", 0.0) if period != baseline_period else 0.0,
        "BaselinePeriod": baseline_period,
    }
    for period in period_levels_all
])

period_orders = hedonic_df["PeriodKey"].astype(str).apply(period_to_order)
hedonic_train_df = hedonic_df[period_orders <= TRAIN_END_ORDER].copy()
hedonic_model_train = fit_hedonic_model(hedonic_train_df)
hedonic_predictions_train = make_hedonic_predictions(hedonic_model_train, hedonic_df)

ward_predicted_full, mesh_predicted_full = aggregate_predictions(hedonic_predictions_full)
ward_predicted_train, mesh_predicted_train = aggregate_predictions(hedonic_predictions_train)

ward_hedonic_index_full = build_index_panel(
    ward_predicted_full,
    group_col="Ward",
    value_col="PredictedPrice",
    count_col="Transactions",
    min_periods=4,
    min_total_obs=20,
    group_label="Ward",
    fallback_min_periods=3,
    fallback_min_total_obs=12,
    allow_relax=True,
)
mesh_hedonic_index_full = build_index_panel(
    mesh_predicted_full,
    group_col="Mesh250m",
    value_col="PredictedPrice",
    count_col="Transactions",
    min_periods=4,
    min_total_obs=12,
    group_label="Mesh250m",
    fallback_min_periods=3,
    fallback_min_total_obs=6,
    allow_relax=True,
)

ward_hedonic_index = build_index_panel(
    ward_predicted_train,
    group_col="Ward",
    value_col="PredictedPrice",
    count_col="Transactions",
    min_periods=4,
    min_total_obs=20,
    group_label="Ward",
    fallback_min_periods=3,
    fallback_min_total_obs=12,
    allow_relax=True,
)
mesh_hedonic_index = build_index_panel(
    mesh_predicted_train,
    group_col="Mesh250m",
    value_col="PredictedPrice",
    count_col="Transactions",
    min_periods=4,
    min_total_obs=12,
    group_label="Mesh250m",
    fallback_min_periods=3,
    fallback_min_total_obs=6,
    allow_relax=True,
)

ward_hedonic_index_full.rename(columns={"PredictedPrice": "WardPredictedPrice", "Index": "WardHedonicIndexFull"}, inplace=True)
mesh_hedonic_index_full.rename(columns={"PredictedPrice": "MeshPredictedPrice", "Index": "MeshHedonicIndexFull"}, inplace=True)
ward_hedonic_index.rename(columns={"PredictedPrice": "WardPredictedPrice", "Index": "WardHedonicIndex"}, inplace=True)
mesh_hedonic_index.rename(columns={"PredictedPrice": "MeshPredictedPrice", "Index": "MeshHedonicIndex"}, inplace=True)


# In[7]:


print(f"Full-sample ward hedonic series: {ward_hedonic_index_full['Ward'].nunique()} wards")
print(f"Full-sample mesh hedonic series: {mesh_hedonic_index_full['Mesh250m'].nunique()} meshes")
print(f"Train-based ward hedonic series: {ward_hedonic_index['Ward'].nunique()} wards")
print(f"Train-based mesh hedonic series: {mesh_hedonic_index['Mesh250m'].nunique()} meshes")

fig_overall = px.line(
    hedonic_index_overall,
    x="PeriodKey",
    y="Index",
    title="Overall Hedonic Price Index (full sample, base = 100)",
    markers=True,
)
fig_overall.update_layout(template="plotly_dark", xaxis_title="Quarter", yaxis_title="Index")
fig_overall


# In[8]:


fig_all_ward_index = px.line(
    ward_hedonic_index_full.sort_values(["Ward", "PeriodKey"]),
    x="PeriodKey",
    y="WardHedonicIndexFull",
    color="Ward",
    template="plotly_dark",
    title="Ward Hedonic Index (Full Sample)"
)
fig_all_ward_index


# ## 3. Save hedonic index tables
# 

# In[9]:


hedonic_index_overall.to_csv(DATA_DIR / "hedonic_index_overall.csv", index=False)
ward_hedonic_index_full.drop(columns=["PeriodOrder"], errors="ignore").to_csv(
    DATA_DIR / "hedonic_index_by_ward.csv", index=False
)
mesh_hedonic_index_full.drop(columns=["PeriodOrder"], errors="ignore").to_csv(
    DATA_DIR / "hedonic_index_by_mesh.csv", index=False
)
ward_hedonic_index.drop(columns=["PeriodOrder"], errors="ignore").to_csv(
    DATA_DIR / "hedonic_index_by_ward_trainmodel.csv", index=False
)
mesh_hedonic_index.drop(columns=["PeriodOrder"], errors="ignore").to_csv(
    DATA_DIR / "hedonic_index_by_mesh_trainmodel.csv", index=False
)
print("Saved hedonic index tables.")

