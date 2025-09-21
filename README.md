# MLIT API Demo Workspace

This repository bundles a lightweight helper (`notebooks/mlit_api_demo.py`) and notebook (`notebooks/test_notebook.ipynb`) that exercise every documented MLIT Real Estate Information Library endpoint. Use it as a playground for experimenting with the official APIs or as a starting point for Sendai-focused analyses.

## Quick start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # or pip install pandas requests python-dotenv nbformat
   ```
2. **Provide credentials**
   ```bash
   echo MLIT_API_KEY="your-key" > .env
   ```
3. **Launch the demo notebook**
   ```bash
   cd notebooks
   jupyter lab test_notebook.ipynb
   ```

The helper exposes `mlit_api_demo.list_endpoints()` which returns metadata (required parameters, defaults, descriptions) for all endpoints — transactions, municipalities, appraisal reports, and the 20+ map/facility tile feeds.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/mlit_api_demo.py` | Endpoint catalogue, validation layer, and thin wrapper functions. |
| `notebooks/test_notebook.ipynb` | Sequential demo that calls every endpoint with sensible defaults and renders sample frames. |
| `initial_scoping.md` | Narrative guide covering API conventions and Sendai-focused use cases. |
| `notebooks/sendai_mlit_exploration.ipynb` | Exploratory analysis notebook (original Sendai workflow). |

## Notes

- Defaults target Miyagi prefecture / Sendai tiles (`z=13`, `x=7301`, `y=3152`). Override them to explore other regions.
- Responses containing GeoJSON are flattened so that the feature properties appear in the first DataFrame columns and the geometry is kept as `_geometry_coordinates`.
- The helper returns raw dictionaries/lists — persist interesting payloads with `Path('filename.json').write_text(json.dumps(payload, ensure_ascii=False))`.
