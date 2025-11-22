Hedonic Forecast - CODE package
================================

This folder is the CLI-first pipeline for rebuilding panels, models, SHAP explainability, and the Streamlit dashboard. It is the only code deliverable inside `teamXXXfinal.zip` (alongside DOC and README).

Data disclaimer
---------------
- Only synthetic, downsampled demo data ships in `CODE/data/` to keep runtime small. The full MLIT Real Estate Transaction Price dataset is **not bundled** due to submission constraints; fetch it yourself from the MLIT API and process locally if you want full fidelity.
- If you do not have MLIT data, just use the included demo inputs—run the workflow/Streamlit steps below and the dashboard will function with the synthetic sample.

Folder overview
---------------
CODE/
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
- visualisation/         Streamlit dashboard (uses CODE/outputs by default)

Data
----
- Demo data is already included under `CODE/data` (synthetic, tiny). Regenerate it with:
  ```
  python CODE/demo_setup.py
  ```
- To use full MLIT data: download quarterly CSVs from https://www.land.mlit.go.jp/webland_english/ (no API key), clean dates/units, map coordinates to JIS 250m meshes, and aggregate to ward/mesh-quarter panels. Place the cleaned outputs in `CODE/data/` with these names:
  1) main_features.parquet
  2) main_features.csv
  3) mesh_quarter_features.csv
  4) Hedonic tables:
     - hedonic_index_overall.csv
     - hedonic_index_by_ward.csv
     - hedonic_index_by_ward_trainmodel.csv
     - hedonic_index_by_mesh.csv
     - hedonic_index_by_mesh_trainmodel.csv
  The exploration notebooks are not part of the submission; only the cleaned tables above are needed.

Environment setup (conda, recommended)
--------------------------------------
1) Create and activate the environment from `environment.yml`:
   ```
   conda env create -f environment.yml
   conda activate dva
   ```
2) Optional (only if you want to train LSTMs): install CPU PyTorch after activation:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   Otherwise, set `SKIP_LSTM=1` when running the workflow to skip Torch.

Quick MLIT API guide
--------------------
- The Real Estate Transaction Price API (`TradeListSearch`) is documented online: https://www.land.mlit.go.jp/webland/api.html (parameters, area codes, examples).
- Example pull (Tokyo 2023, all quarters):
  ```
  curl -o data/raw_tokyo_2023.json "https://www.land.mlit.go.jp/webland/api/TradeListSearch?from=20231&to=20234&city=13101"
  ```
  Adjust `from`/`to` for the year/quarter span and `city` for your municipality code.
- Convert and stack the JSON into CSV, clean units/dates, map lat/lon to JIS 250m meshes, then aggregate to the tables listed in the Data section before running `python -m CODE.run_workflow`.

Workflow (fast path)
--------------------
Use env var `SKIP_LSTM=1` to keep the demo quick:
```
SKIP_LSTM=1 python -m CODE.run_workflow
```
This rebuilds panels, trains classical models, writes metrics/predictions, and exports SHAP for trees/linear models. Outputs land in `CODE/outputs/`.

Speed toggles
-------------
- `EXPORT_TREE_SHAP=0`  skip RandomForest/LightGBM SHAP plots.
- `EXPORT_LINEAR_SHAP=0`  skip LinearRegression SHAP plots.
- `EXPORT_LSTM_SHAP=0`  skip Torch LSTM SHAP generation.
- `SKIP_LSTM=1`         skip LSTM training entirely (recommended for demo data).
*Note:* `python -m CODE.run_workflow` regenerates metrics, predictions, SHAP outputs, and report tables in one shot; you generally do not need to run reporting or efficiency scripts separately unless you want only those tables.

Reports and artifacts
---------------------
- outputs/model_results.csv              combined leaderboard (ward + mesh, classical + LSTM)
- outputs/ward_predictions_detailed.csv  classical + LSTM ward predictions
- outputs/mesh_predictions_detailed.csv  classical + LSTM mesh predictions
- outputs/model_predictions_viz.csv      merged table for dashboards/maps
- outputs/shap_outputs/                  SHAP CSVs + metadata for each level/model (city splits, val/test)
- outputs/shap_plots/<level>/            SHAP bar & beeswarm PNGs (Tokyo/Sendai subsets included)
- outputs/reports/                       data tables cited in the report (regen via `python -m CODE.reporting --reports all`)

Local visualisation (Streamlit)
-------------------------------
- Run with the demo artifacts (no full data needed):
  ```
  cd CODE/visualisation
  streamlit run streamlit_2.py
  ```
  The app auto-detects `CODE/outputs` and the demo geojson.
- If you regenerate outputs, rerun `SKIP_LSTM=1 python -m CODE.run_workflow` first, then launch Streamlit.
- A Streamlit Community Cloud deployment is linked via the poster QR for quick access.

Report and poster
-----------------
- `team090report.pdf` and `team090poster.pdf` are included for grading; both are generated from the pipeline artifacts and dashboard visuals.

End-to-end with your own data
-----------------------------
1) Download MLIT CSVs (public, no key) and clean/aggregate into the required tables under `CODE/data`.
2) `SKIP_LSTM=1 python -m CODE.run_workflow` (add SHAP toggles if desired).
3) Launch the dashboard: `cd CODE/visualisation && streamlit run streamlit_2.py`.
