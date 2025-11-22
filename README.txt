Hedonic Forecast - CODE package
=====================================

This folder is the CLI-first pipeline for rebuilding panels, models, SHAP explainability, and report tables. It contains only what is delivered in the graded `teamXXXfinal.zip` (CODE + DOC + README); no external copies or duplicate code.

Data disclaimer
---------------
- The repo ships **only synthetic, heavily downsampled demo data** under `CODE/data/` so the pipeline and Streamlit demo run fast. The full MLIT Real Estate Transaction Price dataset is **not included** due to submission constraints; to reproduce full results you must pull the data yourself from the MLIT API and process it locally.
- If you do not have access to the full MLIT data, you can still run the demo: use the included `CODE/data` and launch the Streamlit app as described below.

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

Required data files
-------------------
Demo data is already included under `CODE/data` (synthetic, tiny).

If you download and swap in the full MLIT-derived data yourself, place these under `CODE/data/`:
1) main_features.parquet
2) main_features.csv
3) mesh_quarter_features.csv
4) Hedonic tables:
   - hedonic_index_overall.csv
   - hedonic_index_by_ward.csv
   - hedonic_index_by_ward_trainmodel.csv
   - hedonic_index_by_mesh.csv
   - hedonic_index_by_mesh_trainmodel.csv
To build these from the public MLIT Real Estate Transaction Price API: download the quarterly CSVs from https://www.land.mlit.go.jp/webland_english/ (no key required), then run your prep scripts to clean dates/units, map coordinates to JIS 250m meshes, and aggregate to ward/mesh-quarter panels. The pipeline expects cleaned outputs in the formats above; the notebooks used for exploration are not part of the submission.

MLIT download guide (API)
-------------------------------
1) Review the official online docs for the Real Estate Transaction Price API (`TradeListSearch` endpoint): https://www.land.mlit.go.jp/webland/api.html
2) Pull quarterly records per city with those parameters, e.g.:
   ```
   curl -o data/raw_tokyo_2023.json "https://www.land.mlit.go.jp/webland/api/TradeListSearch?from=20231&to=20234&city=13101"
   ```
   Adjust `from`/`to` for the quarter span and `city` for the municipality code you need.
3) Convert/stack the JSON responses into CSV, clean numeric fields and dates, map lat/lon to 250m JIS meshes, and aggregate to the input tables listed above before running `python -m CODE.run_workflow`.

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

End-to-end workflow
-------------------
Primary entry point (skip LSTM for faster processing):
```
SKIP_LSTM=1 python -m CODE.run_workflow
```
This command:
1) Rebuilds ward/mesh panels (hedonic fallbacks + missing flags).
2) Trains Linear Regression, Random Forest, and LightGBM for both levels.
3) Trains ward and mesh Torch LSTMs (omit when `SKIP_LSTM=1` or if torch is unavailable).
4) Writes prediction/metric CSVs plus `model_results.csv`.
5) Exports SHAP summaries, local CSVs, and bar/beeswarm PNGs under `outputs/shap_outputs/` and `outputs/shap_plots/`.
   - LSTM SHAP now splits by city on any split with data (Sendai mesh uses test if validation lacks samples).

Environment toggles for faster iterations:
* `EXPORT_TREE_SHAP=0`  skip RandomForest/LightGBM SHAP plots.
* `EXPORT_LINEAR_SHAP=0`  skip LinearRegression SHAP plots.
* `EXPORT_LSTM_SHAP=0`  skip Torch LSTM SHAP generation.
* `SKIP_LSTM=1`  skip LSTM training entirely (handy for the included demo data).
*Note:* `python -m CODE.run_workflow` regenerates the full set of outputs (metrics, predictions, SHAP, and report tables); you normally do not need to run reporting/efficiency scripts separately.

Example (skip all SHAP, keep models/predictions):
```
EXPORT_TREE_SHAP=0 EXPORT_LINEAR_SHAP=0 EXPORT_LSTM_SHAP=0 SKIP_LSTM=1 python -m CODE.run_workflow
```

After clearing outputs
----------------------
If you remove `CODE/outputs/*`, rerun:
```
python -m CODE.run_workflow
```
to regenerate all artifacts before opening the report or dashboard.

Report tables
-------------
Generate report tables after the workflow:
```
python -m CODE.reporting --reports all
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
python -m CODE.experiments --levels Ward Mesh --fractions 0.25 0.5 0.75 1.0
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
*Visualisation:* the Streamlit app in `visualisation/streamlit_2.py` looks for artifacts under `CODE/outputs`. A tiny demo geojson (`visualisation/mesh250_all_quarters_demo.geojson`) is included to keep the map responsive.

Local visualisation (Streamlit)
-------------------------------
- Use the prebuilt demo artifacts (no full data needed):  
  ```
  cd CODE/visualisation
  streamlit run streamlit_2.py
  ```  
  The app auto-detects `CODE/outputs` and the demo geojson.
- If you regenerate outputs, run `SKIP_LSTM=1 python -m CODE.run_workflow` first, then launch Streamlit from `visualisation/`.
- A Streamlit Community Cloud deployment is also available; the link/QR is provided on the poster for quick access.


For full rebuilds with full data: prepare MLIT downloads → clean and aggregate into the expected CSV/Parquet tables → run `SKIP_LSTM=1 python -m CODE.run_workflow` → launch the Streamlit app.
