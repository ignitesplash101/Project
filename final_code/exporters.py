from __future__ import annotations

import pandas as pd

from . import config


def export_artifacts(
    ward_results: pd.DataFrame,
    mesh_results: pd.DataFrame,
    lstm_results: pd.DataFrame,
    ward_predictions: pd.DataFrame,
    mesh_predictions: pd.DataFrame,
    lstm_predictions: pd.DataFrame,
):
    # combine leaderboard entries from every model family
    all_results = pd.concat([ward_results, mesh_results, lstm_results], ignore_index=True, sort=False)
    all_results.to_csv(config.MODEL_RESULTS_CSV, index=False)

    # ward csv includes both classical and lstm predictions
    ward_predictions_full = pd.concat([ward_predictions, lstm_predictions], ignore_index=True, sort=False)
    ward_predictions_full.to_csv(config.WARD_PREDICTIONS_CSV, index=False)

    # mesh predictions currently only come from classical models
    mesh_predictions.to_csv(config.MESH_PREDICTIONS_CSV, index=False)

    # assemble the viz table with lat/lon fields for mapping tools
    viz_frames = []
    if not ward_predictions_full.empty:
        ward_viz = ward_predictions_full.copy()
        ward_viz["Latitude"] = ward_viz["WardLat"]
        ward_viz["Longitude"] = ward_viz["WardLon"]
        viz_frames.append(ward_viz)
    if not mesh_predictions.empty:
        mesh_viz = mesh_predictions.copy()
        mesh_viz["Latitude"] = mesh_viz["MeshLat"].fillna(mesh_viz["WardLat"])
        mesh_viz["Longitude"] = mesh_viz["MeshLon"].fillna(mesh_viz["WardLon"])
        viz_frames.append(mesh_viz)

    if viz_frames:
        model_predictions_viz = pd.concat(viz_frames, ignore_index=True, sort=False)
        model_predictions_viz.to_csv(config.MODEL_VIZ_CSV, index=False)
