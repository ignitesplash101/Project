Hedonic Forecast - team090final README
======================================

QUICKSTART
----------
```
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
SKIP_LSTM=1 python -m CODE.run_workflow
cd CODE/visualisation && streamlit run streamlit_2.py
```

DESCRIPTION
-----------
This is a simple CLI-first pipeline (in `CODE/`) to rebuild ward/mesh panels, train classical + LSTM models, export SHAP explainability files and other metrics, and power a Streamlit dashboard using real estate prices in Japan. 
The zip deliverable `team090final.zip` includes this `README.txt`, `DOC/`, and the `CODE/` folder. 
The  `DOC/` folder contains the final report and poster `team090report.pdf` and `team090poster.pdf`. For more details on the project approach and methodologies, please refer to either of these files.
*Only synthetic demo data is provided here for the submission; the full MLIT Real Estate Transaction Price data is excluded due to project submission constraints. 
To reproduce full results, you will need to download MLIT data via the public API and process them locally.

INSTALLATION
------------
1) Create and activate a conda env from `environment.yml` (choose a name):
   ```
   conda env create -f environment.yml -n <env_name>
   conda activate <env_name>
   ```
2) Optional (only if training LSTMs):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   Otherwise set `SKIP_LSTM=1` to skip Torch.
3) Data:
   - Demo data is already under `CODE/data` (synthetic, tiny).
   - If you fetch MLIT data yourself, place cleaned files in `CODE/data/`:
     * main_features.parquet / main_features.csv
     * mesh_quarter_features.csv
     * hedonic_index_overall.csv
     * hedonic_index_by_ward.csv
     * hedonic_index_by_ward_trainmodel.csv
     * hedonic_index_by_mesh.csv
     * hedonic_index_by_mesh_trainmodel.csv
     MLIT API docs: https://www.land.mlit.go.jp/webland/api.html  
     Example pull:
     ```
     curl -o data/raw_tokyo_2023.json "https://www.land.mlit.go.jp/webland/api/TradeListSearch?from=20231&to=20234&city=13101"
     ```
     Clean/stack JSON to CSV, map lat/lon to 250m JIS meshes, then aggregate to the tables above.

EXECUTION
---------
Fast path (skip LSTM):
```
SKIP_LSTM=1 python -m CODE.run_workflow
```
This rebuilds panels, trains classical models, writes metrics/predictions, and exports SHAP based on the sample data. 
Outputs from this process will land in `CODE/outputs/`. 

Streamlit (uses `CODE/outputs` and the bundled demo geojson), to initialize the dashboard:
```
cd CODE/visualisation
streamlit run streamlit_2.py
```
If you are unable to run this locally, the `team090poster.pdf` file contains a QR code that allows you access to an online Streamlit community hosted version of the dashboard.
Please feel free to use that instead.

Key outputs:
- `outputs/model_results.csv` — leaderboard
- `outputs/model_predictions_viz.csv` — dashboard inputs
- `outputs/ward_predictions_detailed.csv`, `outputs/mesh_predictions_detailed.csv`
- `outputs/shap_outputs/`, `outputs/shap_plots/`
- `outputs/reports/` — report tables (`python -m CODE.reporting --reports all` if needed)

Submission packaging
--------------------
- Zip name: `team090final.zip`.
- Contents:
  - `README.txt` (this file): DESCRIPTION, INSTALLATION, EXECUTION, optional 1-minute unlisted demo video URL.
  - `DOC/`: `team090report.pdf` and `team090poster.pdf`.
  - `CODE/`: only the necessary files for the demo.

Getting real MLIT data (optional)
-----------------------------------------------
- Not required for the demo. The full MLIT Real Estate Transaction Price data is **not included**.
- If you still want to reproduce full results, you must obtain your own API access and pull data from the official endpoint (`TradeListSearch`): https://www.land.mlit.go.jp/webland/api.html
- Example pull (Tokyo 2023 all quarters):
  ```
  curl -o data/raw_tokyo_2023.json "https://www.land.mlit.go.jp/webland/api/TradeListSearch?from=20231&to=20234&city=13101"
  ```
- Then clean/stack JSON to CSV, map lat/lon to 250m JIS meshes, and aggregate to the required tables in `CODE/data`:
  main_features.parquet/.csv, mesh_quarter_features.csv, and the hedonic_index_*.csv files listed in INSTALLATION.
- Please note that API key access, data cleaning, coordinate-to-mesh mapping, and aggregation are all manual steps.