"""
Generate lightweight demo data and outputs for the CODE package.

This builds tiny synthetic ward + mesh panels so the workflow and dashboard
can run quickly without the full MLIT dataset. Re-run this script to refresh
all demo artifacts in CODE/data and CODE/outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "CODE" / "data"
OUTPUT_DIR = REPO_ROOT / "CODE" / "outputs"
SHAP_PLOT_DIR = OUTPUT_DIR / "shap_plots" / "mesh"


def period_list() -> List[str]:
    # keep enough quarters so lag-4 features exist across train/val/test splits
    periods = []
    for year in range(2018, 2023):
        for q in range(1, 5):
            if year == 2022 and q > 1:
                continue
            periods.append(f"{year}-Q{q}")
    return periods


def period_order(period_key: str) -> int:
    year, quarter = period_key.split("-Q")
    return int(year) * 4 + (int(quarter) - 1)


def build_main_features() -> pd.DataFrame:
    periods = period_list()
    demo_meshes = [
        {
            "ward": "Chiyoda Ward",
            "city": "Tokyo",
            "ward_lat": 35.68,
            "ward_lon": 139.76,
            "mesh": "53394631",
            "mesh_lat": 35.69,
            "mesh_lon": 139.77,
            "price_base": 550_000,
            "price_step": 8_500,
        },
        {
            "ward": "Chiyoda Ward",
            "city": "Tokyo",
            "ward_lat": 35.68,
            "ward_lon": 139.76,
            "mesh": "53394632",
            "mesh_lat": 35.70,
            "mesh_lon": 139.74,
            "price_base": 540_000,
            "price_step": 7_500,
        },
        {
            "ward": "Aoba Ward",
            "city": "Sendai",
            "ward_lat": 38.27,
            "ward_lon": 140.87,
            "mesh": "58405001",
            "mesh_lat": 38.27,
            "mesh_lon": 140.87,
            "price_base": 320_000,
            "price_step": 6_000,
        },
        {
            "ward": "Aoba Ward",
            "city": "Sendai",
            "ward_lat": 38.27,
            "ward_lon": 140.87,
            "mesh": "58405002",
            "mesh_lat": 38.29,
            "mesh_lon": 140.85,
            "price_base": 310_000,
            "price_step": 5_500,
        },
    ]

    rows = []
    for mesh_info in demo_meshes:
        for idx, period in enumerate(periods):
            price_per_sqm = mesh_info["price_base"] + idx * mesh_info["price_step"]
            area = 60 + (idx % 3) * 5
            building_age = 5 + idx
            rows.append(
                {
                    "WardName": mesh_info["ward"],
                    "Municipality": mesh_info["ward"],
                    "Municipality_en": mesh_info["ward"],
                    "Mesh250m": mesh_info["mesh"],
                    "PeriodKey": period,
                    "PricePerSqM": price_per_sqm,
                    "TradePriceValue": price_per_sqm * area,
                    "AreaSqM": area,
                    "BuildingAge": building_age,
                    "Latitude": mesh_info["mesh_lat"],
                    "Longitude": mesh_info["mesh_lon"],
                    "MeshLat": mesh_info["mesh_lat"],
                    "MeshLon": mesh_info["mesh_lon"],
                }
            )
    return pd.DataFrame(rows)


def build_mesh_features(main_df: pd.DataFrame) -> pd.DataFrame:
    mesh_features = (
        main_df.groupby(["Mesh250m", "PeriodKey"], dropna=False)
        .agg(
            mesh_transaction_count=("PricePerSqM", "count"),
            mesh_median_ppsqm=("PricePerSqM", "median"),
            mesh_mean_ppsqm=("PricePerSqM", "mean"),
            mesh_price_std=("PricePerSqM", "std"),
            mesh_price_iqr=("PricePerSqM", lambda s: s.quantile(0.75) - s.quantile(0.25)),
            mesh_avg_age=("BuildingAge", "mean"),
            mesh_avg_area=("AreaSqM", "mean"),
            MeshLat=("MeshLat", "first"),
            MeshLon=("MeshLon", "first"),
        )
        .reset_index()
    )
    mesh_features["PeriodNum"] = mesh_features["PeriodKey"].apply(period_order)
    return mesh_features


def build_hedonic_tables(main_df: pd.DataFrame, mesh_features: pd.DataFrame):
    hedonic_dir = {
        "overall": DATA_DIR / "hedonic_index_overall.csv",
        "ward_full": DATA_DIR / "hedonic_index_by_ward.csv",
        "ward_train": DATA_DIR / "hedonic_index_by_ward_trainmodel.csv",
        "mesh_full": DATA_DIR / "hedonic_index_by_mesh.csv",
        "mesh_train": DATA_DIR / "hedonic_index_by_mesh_trainmodel.csv",
    }

    # overall index: simple linear growth anchored at first period
    periods = sorted(main_df["PeriodKey"].unique(), key=period_order)
    base_period = periods[0]
    overall_rows = []
    for i, period in enumerate(periods):
        overall_rows.append(
            {
                "PeriodKey": period,
                "Index": 100 + i * 2,
                "Effect": i * 0.02,
                "BaselinePeriod": base_period,
            }
        )
    pd.DataFrame(overall_rows).to_csv(hedonic_dir["overall"], index=False)

    # ward hedonic (full + train-model flavors)
    ward_group = main_df.groupby(["WardName", "PeriodKey"], dropna=False)["PricePerSqM"].median().reset_index()
    ward_group = ward_group.rename(columns={"WardName": "Ward", "PricePerSqM": "WardPredictedPrice"})
    ward_group["PeriodOrder"] = ward_group["PeriodKey"].apply(period_order)
    ward_group["BasePeriod"] = base_period
    ward_group["Transactions"] = 5
    ward_group["WardHedonicIndexFull"] = ward_group.groupby("Ward")["WardPredictedPrice"].transform(
        lambda s: s / s.iloc[0] * 100
    )
    ward_train = ward_group.rename(columns={"WardHedonicIndexFull": "WardHedonicIndex"}).copy()

    ward_group.to_csv(hedonic_dir["ward_full"], index=False)
    ward_train.to_csv(hedonic_dir["ward_train"], index=False)

    # mesh hedonic (full + train-model flavors)
    mesh_group = mesh_features[["Mesh250m", "PeriodKey", "mesh_median_ppsqm"]].rename(
        columns={"mesh_median_ppsqm": "MeshPredictedPrice"}
    )
    mesh_group["PeriodOrder"] = mesh_group["PeriodKey"].apply(period_order)
    mesh_group["BasePeriod"] = base_period
    mesh_group["Transactions"] = 3
    mesh_group["MeshHedonicIndexFull"] = mesh_group.groupby("Mesh250m")["MeshPredictedPrice"].transform(
        lambda s: s / s.iloc[0] * 100
    )
    mesh_train = mesh_group.rename(columns={"MeshHedonicIndexFull": "MeshHedonicIndex"}).copy()

    mesh_group.to_csv(hedonic_dir["mesh_full"], index=False)
    mesh_train.to_csv(hedonic_dir["mesh_train"], index=False)


def build_price_index(mesh_features: pd.DataFrame):
    rows = []
    for _, row in mesh_features.iterrows():
        base = (
            mesh_features.loc[mesh_features["Mesh250m"] == row["Mesh250m"], "mesh_median_ppsqm"].iloc[0]
        )
        rows.append(
            {
                "Mesh250m": row["Mesh250m"],
                "PeriodKey": row["PeriodKey"],
                "PriceIndex": row["mesh_median_ppsqm"] / base * 100,
                "Latitude": row["MeshLat"],
                "Longitude": row["MeshLon"],
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "mesh_quarterly_price_index.csv", index=False)


def build_demo_outputs(main_df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # aggregated ward + mesh panels
    ward_panel = (
        main_df.groupby(["WardName", "PeriodKey"], dropna=False)
        .agg(
            Actual=("PricePerSqM", "median"),
            WardLat=("Latitude", "median"),
            WardLon=("Longitude", "median"),
        )
        .reset_index()
        .rename(columns={"WardName": "Ward"})
    )
    ward_panel["City"] = ward_panel["Ward"].apply(lambda w: "Sendai" if "Sendai" in w else "Tokyo")

    mesh_panel = (
        main_df.groupby(["Mesh250m", "PeriodKey"], dropna=False)
        .agg(
            Actual=("PricePerSqM", "median"),
            Ward=("WardName", "first"),
            WardLat=("Latitude", "first"),
            WardLon=("Longitude", "first"),
            MeshLat=("Latitude", "first"),
            MeshLon=("Longitude", "first"),
        )
        .reset_index()
    )
    mesh_panel["City"] = mesh_panel["Ward"].apply(lambda w: "Sendai" if "Sendai" in w else "Tokyo")

    def split_label(period_key: str) -> str:
        order = period_order(period_key)
        if order <= period_order("2019-Q4"):
            return "train"
        if order <= period_order("2021-Q4"):
            return "val"
        return "test"

    # simple deterministic predictions for demo purposes
    def make_preds(df: pd.DataFrame, level: str) -> pd.DataFrame:
        df = df.copy()
        df["Predicted"] = df["Actual"] * 1.03
        df["Level"] = level
        df["Model"] = "LinearRegression"
        df["Split"] = df["PeriodKey"].apply(split_label)
        return df

    ward_preds = make_preds(ward_panel, "Ward")
    mesh_preds = make_preds(mesh_panel, "Mesh")

    # results summary (dummy metrics)
    def results_frame(level: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Model": "LinearRegression",
                    "train_mae": 1000,
                    "train_rmse": 1500,
                    "train_r2": 0.98,
                    "val_mae": 1200,
                    "val_rmse": 1600,
                    "val_r2": 0.97,
                    "test_mae": 1300,
                    "test_rmse": 1700,
                    "test_r2": 0.96,
                    "Level": level,
                }
            ]
        )

    model_results = pd.concat([results_frame("Ward"), results_frame("Mesh")], ignore_index=True)
    model_results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    ward_preds.to_csv(OUTPUT_DIR / "ward_predictions_detailed.csv", index=False)
    mesh_preds.to_csv(OUTPUT_DIR / "mesh_predictions_detailed.csv", index=False)

    viz_frames = []
    for df in [ward_preds, mesh_preds]:
        copy = df.copy()
        copy["Latitude"] = copy.get("MeshLat", copy["WardLat"])
        copy["Longitude"] = copy.get("MeshLon", copy["WardLon"])
        viz_frames.append(copy)
    pd.concat(viz_frames, ignore_index=True).to_csv(OUTPUT_DIR / "model_predictions_viz.csv", index=False)

    # lightweight SHAP explanation and plots for the dashboard
    feature_names = ["PriceLag1", "Transactions", "AvgAge"]
    values = np.array(
        [
            [0.1, -0.05, 0.02],
            [0.12, -0.03, 0.01],
            [0.08, -0.02, 0.03],
            [0.15, -0.04, 0.00],
        ]
    )
    shap_exp = shap.Explanation(
        values=values,
        base_values=np.array([0.5] * values.shape[0]),
        data=np.array(
            [
                [1.0, 10, 20],
                [1.1, 12, 22],
                [0.9, 9, 18],
                [1.2, 11, 21],
            ]
        ),
        feature_names=feature_names,
    )
    shap_path = OUTPUT_DIR / "mesh_linearregression_val_shap.pkl"
    shap_path.write_bytes(pickle.dumps(shap_exp))

    def write_shap_plot(city: str, kind: str):
        plt.clf()
        fig, ax = plt.subplots()
        vals = values.mean(axis=0)
        if kind == "bar":
            ax.bar(feature_names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel("Mean SHAP value")
        else:
            for j, feat in enumerate(feature_names):
                ax.scatter(shap_exp.data[:, j], shap_exp.values[:, j], label=feat)
            ax.legend()
            ax.set_xlabel("Feature value")
            ax.set_ylabel("SHAP value")
        ax.set_title(f"{city} - LinearRegression ({kind})")
        fname = SHAP_PLOT_DIR / f"mesh_linearregression_val_{city.lower()}_{kind}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)

    for city in ["tokyo", "sendai"]:
        for kind in ["bar", "beeswarm"]:
            write_shap_plot(city, kind)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_df = build_main_features()
    mesh_features = build_mesh_features(main_df)

    # core data tables
    main_df.to_parquet(DATA_DIR / "main_features.parquet", index=False)
    main_df.to_csv(DATA_DIR / "main_features.csv", index=False)
    mesh_features.to_csv(DATA_DIR / "mesh_quarter_features.csv", index=False)
    build_price_index(mesh_features)
    build_hedonic_tables(main_df, mesh_features)

    # demo outputs for dashboards/tests
    build_demo_outputs(main_df)
    print("Demo data and outputs written to CODE/data and CODE/outputs.")


if __name__ == "__main__":
    main()
