from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from . import config
from .utils import evaluate_sets, temporal_split


# keep factory definitions here so hyperparameters stay in one place
MODEL_FACTORIES = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ),
}

TREE_MODELS = {"RandomForest", "LightGBM"}


def prepare_splits(panel: pd.DataFrame, feature_cols: List[str], target_col: str):
    # drop rows lacking the target or any requested feature so splits stay aligned
    feature_cols = [c for c in feature_cols if c in panel.columns]
    train_df, val_df, test_df = temporal_split(panel.dropna(subset=[target_col]))
    splits = {
        "train": train_df.dropna(subset=feature_cols + [target_col]).copy(),
        "val": val_df.dropna(subset=feature_cols + [target_col]).copy(),
        "test": test_df.dropna(subset=feature_cols + [target_col]).copy(),
    }
    return splits, feature_cols


def train_regression_models(
    panel: pd.DataFrame,
    level_name: str,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, pd.DataFrame], List[str]]:
    # produce train/val/test splits and fit each classical model
    splits, feature_cols = prepare_splits(panel, feature_cols, target_col)
    results, prediction_frames, trained_models = [], [], {}

    for name, factory in MODEL_FACTORIES.items():
        train_df = splits["train"]
        if train_df.empty:
            continue
        model = factory()
        model.fit(train_df[feature_cols], train_df[target_col])
        preds = {split: model.predict(split_df[feature_cols]) for split, split_df in splits.items()}
        metrics = {
            split: evaluate_sets(split_df[target_col], preds[split])
            for split, split_df in splits.items()
            if not split_df.empty
        }
        results.append(
            {
                "Model": name,
                **{
                    f"{split}_{metric}": values[metric]
                    for split, values in metrics.items()
                    for metric in ["mae", "rmse", "r2"]
                },
            }
        )
        for split, split_df in splits.items():
            if split_df.empty:
                continue
            payload = {
                "Level": level_name,
                "Model": name,
                "Split": split,
                "PeriodKey": split_df["PeriodKey"].values,
                "Actual": split_df[target_col].values,
                "Predicted": preds[split],
            }
            for col in ["Ward", "WardLat", "WardLon", "Mesh250m", "MeshLat", "MeshLon"]:
                payload[col] = split_df.get(col, pd.Series(np.nan, index=split_df.index)).values
            prediction_frames.append(pd.DataFrame(payload))
        model_path = config.OUTPUT_DIR / f"{level_name.lower()}_model_{name.lower()}.pkl"
        joblib.dump(model, model_path)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df["Level"] = level_name
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return results_df, predictions_df, trained_models, splits, feature_cols


def save_tree_shap_outputs(
    level_name: str,
    model_name: str,
    model,
    split_df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
    target_col: str,
) -> None:
    if split_df.empty:
        return
    explainer = shap.TreeExplainer(model)
    sample = split_df.sample(n=min(600, len(split_df)), random_state=42)
    shap_values = explainer.shap_values(sample[feature_cols])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_abs = np.abs(shap_values)
    summary_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "MeanAbsSHAP": shap_abs.mean(axis=0),
            "MedianAbsSHAP": np.median(shap_abs, axis=0),
            "StdAbsSHAP": shap_abs.std(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    total_mean_abs = summary_df["MeanAbsSHAP"].sum()
    summary_df["ContributionShare"] = (
        summary_df["MeanAbsSHAP"] / total_mean_abs if total_mean_abs else 0
    )
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    predictions = model.predict(sample[feature_cols])
    if predictions.ndim > 1:
        predictions = predictions.ravel()
    actual = sample[target_col].to_numpy()
    residuals = actual - predictions

    # record per-observation shap values along with prediction context
    local_records = []
    sample_reset = sample.reset_index(drop=True)
    for i in range(len(sample_reset)):
        meta_cols = [
            "PeriodKey",
            "Ward",
            "Mesh250m",
            "WardLat",
            "WardLon",
            "MeshLat",
            "MeshLon",
        ]
        meta = {col: sample_reset.iloc[i].get(col) for col in meta_cols if col in sample_reset.columns}
        base = {
            "ObservationIndex": int(i),
            "Model": model_name,
            "Level": level_name,
            "Split": split_name,
            "Actual": float(actual[i]),
            "Predicted": float(predictions[i]),
            "Residual": float(residuals[i]),
            **meta,
        }
        for feature_idx, feature in enumerate(feature_cols):
            local_records.append(
                {
                    **base,
                    "Feature": feature,
                    "FeatureValue": float(sample_reset.iloc[i][feature]),
                    "SHAPValue": float(shap_values[i, feature_idx]),
                }
            )
    local_df = pd.DataFrame(local_records)
    local_df.to_csv(
        config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_local.csv",
        index=False,
    )
    metadata_path = config.SHAP_DIR / f"{level_name.lower()}_{model_name.lower()}_shap_metadata.json"
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "expected_value": float(expected_value),
                "feature_cols": feature_cols,
                "n_samples": len(sample_reset),
                "target_col": target_col,
                "split": split_name,
            },
            fh,
            indent=2,
        )


def export_tree_shap(
    level_name: str,
    models: Dict[str, object],
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str,
):
    for model_name, model in models.items():
        if model_name not in TREE_MODELS:
            continue
        save_tree_shap_outputs(
            level_name,
            model_name,
            model,
            splits["val"],
            feature_cols,
            "val",
            target_col,
        )
