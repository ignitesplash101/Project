Hedonic Forecast - final_code package
=====================================

This folder is the CLI-first pipeline for rebuilding panels, models, SHAP explainability, and report tables.

Folder overview
---------------
final_code/
- config.py              shared path registry
- data_loader.py         loads parquet/CSV inputs and normalises types
- panels.py              rebuilds ward + mesh feature panels (lags, hedonic joins, missing flags)
- models.py              classical regressors (linear, rf, lightgbm) plus SHAP helpers
- lstm_model.py          torch LSTM sequence pipeline for wards and meshes
- run_workflow.py        one command to refresh everything (models + SHAP + exports)
- exporters.py           writes dashboard CSV inputs and viz table
- reporting.py           produces report-ready tables (accuracy, feature importance, granularity deltas)
- experiments.py         runs train-fraction efficiency sweeps for ward/mesh panels
- data/                  required inputs (main_features, mesh_quarter_features, hedonic_index_*.csv)
- outputs/               artifacts (results/predictions, SHAP CSVs + PNGs, reports)

Required data files
-------------------
Place these under `final_code/data/` before running:
1) main_features.parquet
2) main_features.csv
3) mesh_quarter_features.csv
4) Hedonic tables:
   - hedonic_index_overall.csv
   - hedonic_index_by_ward.csv
   - hedonic_index_by_ward_trainmodel.csv
   - hedonic_index_by_mesh.csv
   - hedonic_index_by_mesh_trainmodel.csv

Regenerate hedonic tables (optional)
------------------------------------
If any hedonic file is missing or stale:
```
python -m final_code.hedonic_indices
```
This reads `main_features.parquet` / `mesh_quarter_features.csv` and rewrites all `hedonic_index_*.csv` in place.

End-to-end workflow
-------------------
Primary entry point:
```
python -m final_code.run_workflow
```
This command:
1) Rebuilds ward/mesh panels (hedonic fallbacks + missing flags).
2) Trains Linear Regression, Random Forest, and LightGBM for both levels.
3) Trains ward and mesh Torch LSTMs.
4) Writes prediction/metric CSVs plus `model_results.csv`.
5) Exports SHAP summaries, local CSVs, and bar/beeswarm PNGs under `outputs/shap_outputs/` and `outputs/shap_plots/`.
   - LSTM SHAP now splits by city on any split with data (Sendai mesh uses test if validation lacks samples).

Environment toggles for faster iterations:
* `EXPORT_TREE_SHAP=0`  skip RandomForest/LightGBM SHAP plots.
* `EXPORT_LINEAR_SHAP=0`  skip LinearRegression SHAP plots.
* `EXPORT_LSTM_SHAP=0`  skip Torch LSTM SHAP generation.

Example (skip all SHAP, keep models/predictions):
```
EXPORT_TREE_SHAP=0 EXPORT_LINEAR_SHAP=0 EXPORT_LSTM_SHAP=0 python -m final_code.run_workflow
```

After clearing outputs
----------------------
If you remove `final_code/outputs/*`, rerun:
```
python -m final_code.run_workflow
```
to regenerate all artifacts before opening the report or dashboard.

Report tables
-------------
Generate report tables after the workflow:
```
python -m final_code.reporting --reports all
```
Outputs (under `outputs/reports/`):
- feature_importance_summary.csv
- accuracy_summary.csv
- granularity_comparison.csv
- efficiency_ward.csv
- efficiency_mesh.csv

Efficiency scaling study
------------------------
Reproduce the train-fraction runtime tables:
```
python -m final_code.experiments --levels Ward Mesh --fractions 0.25 0.5 0.75 1.0
```
Saves `efficiency_ward.csv` and `efficiency_mesh.csv` under `outputs/reports/`.

Where outputs go
----------------
- outputs/model_results.csv              combined leaderboard (ward + mesh, classical + LSTM)
- outputs/ward_predictions_detailed.csv  classical + LSTM ward predictions
- outputs/mesh_predictions_detailed.csv  classical + LSTM mesh predictions
- outputs/model_predictions_viz.csv      merged table for dashboards/maps
- outputs/shap_outputs/                  SHAP CSVs + metadata for each level/model (city splits, val/test)
- outputs/shap_plots/<level>/            SHAP bar & beeswarm PNGs (Tokyo/Sendai subsets included)
- outputs/reports/                       data tables cited in the written report

Need help?
----------
See `final_code/WORKFLOW.md` for a narrative walkthrough. Each script logs the files it writes so you can confirm which artifacts were refreshed.
