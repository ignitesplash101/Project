from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config

TRAIN_END = "2019-Q4"


def period_to_order(period_key: str) -> int:
    year, quarter = period_key.split("-Q")
    return int(year) * 4 + (int(quarter) - 1)


def fit_hedonic_model(data: pd.DataFrame):
    formula = (
        "LogPrice ~ np.log(AreaSqM) + BuildingAge + C(Ward) + "
        "C(PeriodKey) + C(TypeFeat) + C(StructureFeat) + C(UseFeat)"
    )
    return smf.ols(formula, data=data).fit()


def make_hedonic_predictions(model, data: pd.DataFrame) -> pd.DataFrame:
    preds = data.copy()
    preds["PredictedLogPrice"] = model.predict(data)
    preds["PredictedPrice"] = np.exp(preds["PredictedLogPrice"])
    return preds


def aggregate_predictions(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def build_index_panel(
    panel_df: pd.DataFrame,
    group_col: str,
    value_col: str,
    count_col: str | None = None,
    min_periods: int = 4,
    min_total_obs: int | None = 20,
    group_label: str | None = None,
    fallback_min_periods: int | None = None,
    fallback_min_total_obs: int | None = None,
    allow_relax: bool = True,
) -> pd.DataFrame:
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


def _prepare_hedonic_dataframe(main_df: pd.DataFrame) -> pd.DataFrame:
    hedonic_df = main_df[
        [
            "LogPrice",
            "AreaSqM",
            "BuildingAge",
            "Municipality",
            "Municipality_en",
            "PeriodKey",
            "Structure",
            "Structure_en",
            "Type",
            "Type_en",
            "Use",
            "Use_en",
            "Mesh250m",
            "WardName",
        ]
    ].copy()

    hedonic_df["Ward"] = hedonic_df["WardName"]
    hedonic_df.drop(columns=["WardName"], inplace=True)
    hedonic_df["StructureFeat"] = hedonic_df["Structure_en"].fillna(hedonic_df["Structure"]).fillna("Unknown")
    hedonic_df["TypeFeat"] = hedonic_df["Type_en"].fillna(hedonic_df["Type"]).fillna("Unknown")
    hedonic_df["UseFeat"] = hedonic_df["Use_en"].fillna(hedonic_df["Use"]).fillna("Unknown")
    return hedonic_df


def load_inputs() -> pd.DataFrame:
    if not config.MAIN_FEATURES_PARQUET.exists():
        raise FileNotFoundError(config.MAIN_FEATURES_PARQUET)
    main_df = pd.read_parquet(config.MAIN_FEATURES_PARQUET)
    main_df["Mesh250m"] = main_df["Mesh250m"].astype(str)
    main_df.loc[main_df["Mesh250m"].str.lower() == "nan", "Mesh250m"] = np.nan
    main_df["WardName"] = main_df["Municipality_en"].fillna(main_df["Municipality"]).fillna("Unknown")
    return main_df


def _build_overall_index(model, periods: list[str]) -> pd.DataFrame:
    baseline_period = periods[0]
    rows = []
    for period in periods:
        if period == baseline_period:
            rows.append({"PeriodKey": period, "Index": 100.0, "Effect": 0.0, "BaselinePeriod": baseline_period})
            continue
        key = f"C(PeriodKey)[T.{period}]"
        effect = float(model.params.get(key, 0.0))
        rows.append(
            {
                "PeriodKey": period,
                "Index": float(np.exp(effect) * 100.0),
                "Effect": effect,
                "BaselinePeriod": baseline_period,
            }
        )
    return pd.DataFrame(rows)


def generate_hedonic_tables(save: bool = True) -> Dict[str, pd.DataFrame]:
    main_df = load_inputs()
    hedonic_df = _prepare_hedonic_dataframe(main_df)
    period_levels = sorted(hedonic_df["PeriodKey"].astype(str).unique())

    model_full = fit_hedonic_model(hedonic_df)
    predictions_full = make_hedonic_predictions(model_full, hedonic_df)

    train_mask = hedonic_df["PeriodKey"].astype(str).apply(period_to_order) <= period_to_order(TRAIN_END)
    hedonic_train = hedonic_df[train_mask].copy()
    model_train = fit_hedonic_model(hedonic_train)
    predictions_train = make_hedonic_predictions(model_train, hedonic_train)

    ward_full, mesh_full = aggregate_predictions(predictions_full)
    ward_train, mesh_train = aggregate_predictions(predictions_train)

    ward_full_idx = build_index_panel(
        ward_full,
        group_col="Ward",
        value_col="PredictedPrice",
        count_col="Transactions",
        min_periods=4,
        min_total_obs=20,
        group_label="Ward",
        fallback_min_periods=3,
        fallback_min_total_obs=12,
    ).rename(columns={"PredictedPrice": "WardPredictedPrice", "Index": "WardHedonicIndexFull"})

    mesh_full_idx = build_index_panel(
        mesh_full,
        group_col="Mesh250m",
        value_col="PredictedPrice",
        count_col="Transactions",
        min_periods=4,
        min_total_obs=12,
        group_label="Mesh250m",
        fallback_min_periods=3,
        fallback_min_total_obs=6,
    ).rename(columns={"PredictedPrice": "MeshPredictedPrice", "Index": "MeshHedonicIndexFull"})

    ward_train_idx = build_index_panel(
        ward_train,
        group_col="Ward",
        value_col="PredictedPrice",
        count_col="Transactions",
        min_periods=4,
        min_total_obs=20,
        group_label="Ward",
        fallback_min_periods=3,
        fallback_min_total_obs=12,
    ).rename(columns={"PredictedPrice": "WardPredictedPrice", "Index": "WardHedonicIndex"})

    mesh_train_idx = build_index_panel(
        mesh_train,
        group_col="Mesh250m",
        value_col="PredictedPrice",
        count_col="Transactions",
        min_periods=4,
        min_total_obs=12,
        group_label="Mesh250m",
        fallback_min_periods=3,
        fallback_min_total_obs=6,
    ).rename(columns={"PredictedPrice": "MeshPredictedPrice", "Index": "MeshHedonicIndex"})

    hedonic_overall = _build_overall_index(model_full, period_levels)

    tables = {
        "overall": hedonic_overall,
        "ward_full": ward_full_idx,
        "mesh_full": mesh_full_idx,
        "ward_train": ward_train_idx,
        "mesh_train": mesh_train_idx,
    }

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        tables["overall"].to_csv(config.HEDONIC_FILES["overall"], index=False)
        tables["ward_full"].to_csv(config.HEDONIC_FILES["ward_full"], index=False)
        tables["mesh_full"].to_csv(config.HEDONIC_FILES["mesh_full"], index=False)
        tables["ward_train"].to_csv(config.HEDONIC_FILES["ward_train"], index=False)
        tables["mesh_train"].to_csv(config.HEDONIC_FILES["mesh_train"], index=False)

    return tables


def run():
    tables = generate_hedonic_tables(save=True)
    print("Saved hedonic tables:")
    for key, path in config.HEDONIC_FILES.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    run()
