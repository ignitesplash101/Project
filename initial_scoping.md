# Sendai Land & Mobility API Field Guide

## 1. Project Narrative
- Build a replicable Sendai-focused analytics workflow that blends land market activity with everyday mobility context.
- Use Japan MLIT Real Estate Information Library endpoints to assemble three core storylines: ward-level price shifts, transaction clustering near transit, and parcel-level valuation signals.
- Pair the engineering workbook (`notebooks/sendai_mlit_exploration.ipynb`) with this written guide so every milestone (proposal, progress report, final package) cites the exact API contract, helper design, and cached evidence.

## 2. Workflow At a Glance
1. **Authenticate** with a valid `MLIT_API_KEY` stored in `.env` or exported in the shell.
2. **Call** the REST endpoints with guarded helpers that log, validate, and cache the results under `notebooks/data/`.
3. **Normalize** prices, parcel sizes, and categorical codes so Plotly charts and summary tables stay stable across runs.
4. **Visualize** Sendai transactions, prices, and land tiles to surface hotspot wards and accessibility gaps.

## 3. Environment & Setup
### 3.1 Python packages
```bash
pip install pandas requests plotly python-dotenv nbformat
```

### 3.2 Credentials and configuration
```python
from pathlib import Path
Path('.env').write_text('MLIT_API_KEY="paste-your-key"\n', encoding='utf-8')
```
The notebook resolves the key with `load_env_variable('MLIT_API_KEY')`. If the variable is absent, a descriptive `RuntimeError` is raised before any HTTP call is made.

### 3.3 Directory layout and caching
| Path | Description |
| --- | --- |
| `notebooks/data/raw/` | Cached API payloads (`trades_*`, `appraisal_*`, `land_price_*`). |
| `notebooks/data/reference/` | Lookup tables such as `municipalities_04.json`. |
| `notebooks/sendai_mlit_exploration.ipynb` | Primary exploratory notebook. |
| `notebooks/sendai_mlit_exploration.py` | Script export for command-line execution. |

Re-running the notebook refreshes caches in place. Version control carefully to avoid bloating commits with large JSON snapshots.

## 4. MLIT Real Estate Information Library Reference
### 4.1 Shared conventions
- **Base URL**: `https://www.reinfolib.mlit.go.jp/ex-api/external`
- **Authentication**: include `Ocp-Apim-Subscription-Key: <MLIT_API_KEY>` in every request header.
- **Language**: pass `language=en` for English output (keys may remain Japanese for some endpoints).
- **Timeout**: helpers set `timeout=30` seconds; adjust if you observe throttling.
- **Empty responses**: HTTP `204` or `404` indicate “no data for that slice”; keep calm and continue.

Generic request pattern:
```python
import os
import requests

BASE_URL = 'https://www.reinfolib.mlit.go.jp/ex-api/external'
HEADERS = {'Ocp-Apim-Subscription-Key': os.environ['MLIT_API_KEY']}
params = {...}
response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=30)
response.raise_for_status()
payload = response.json()
```

```
GET /ex-api/external/{ENDPOINT}?key=value&... HTTP/1.1
Host: www.reinfolib.mlit.go.jp
Ocp-Apim-Subscription-Key: YOUR-KEY-HERE
Accept: application/json
```

### 4.2 XIT001 – Real Estate Transaction Price Information (Trades)
**Purpose**: Capture closed real estate transactions by quarter with price, land use, planning constraints, and geographies. Powers price trendlines, transaction mix analysis, and ward-level comparisons.

**Typical use cases**
- Compute median price per square metre by ward and quarter.
- Track transaction counts by property type.
- Identify Sendai-specific records via `MunicipalityCode` filtering.

**Raw GET string**
```text
https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001?area=04&year=2024&quarter=4&priceClassification=01&language=en
```

**Request payload (query params as JSON)**
```json
{
  "area": "04",
  "year": "2024",
  "quarter": "4",
  "priceClassification": "01",
  "language": "en"
}
```
Add optional keys `city` (e.g., `04101`) or `station` (e.g., `040010`) to narrow the slice.

**Key query parameters**
| Parameter | Required | Example | Notes |
| --- | --- | --- | --- |
| `area` | Yes | `04` | Prefecture code (Miyagi = 04). |
| `year` | Yes | `2024` | Calendar year. |
| `quarter` | Yes | `1`–`4` | Quarter number. |
| `priceClassification` | Optional | `01` | `01` = transaction prices, `02` = contract quotes, omit for both. |
| `city` | Optional | `04101` | Municipality code (Sendai wards). |
| `station` | Optional | `040010` | Station code for rail-focused pulls. |
| `language` | Optional | `en` | English text where available. |

**Python helper signature**
```python
def fetch_trade_data(pref_code: str, *,
                     city_code: str | None = None,
                     station_code: str | None = None,
                     price_classification: str | None = '01',
                     start_year: int = 2020, start_quarter: int = 1,
                     end_year: int | None = None, end_quarter: int | None = None,
                     language: str = 'en') -> list[dict]:
    """Loop across year/quarter ranges, enforce at least one location filter,
    attach default Year/Quarter fields, and return a flat list ready for pandas."""
```
Helper highlights:
- Validates quarter bounds (1–4) and chronological order.
- Calls `call_mlit_api('XIT001', params)` per quarter and adds fallback fields (`Year`, `Quarter`, `PriceClassification`).
- Caches payloads to `notebooks/data/raw/trades_{pref_code}_{period}.json`.

**Response excerpt**
```json
{
  "Prefecture": "Miyagi Prefecture",
  "Municipality": "Aoba Ward, Sendai City",
  "MunicipalityCode": "4101",
  "DistrictName": "Akebonomachi",
  "PriceCategory": "Real Estate Transaction Price Information",
  "Type": "Residential Land(Land Only)",
  "TradePrice": "22000000",
  "PricePerUnit": "250000",
  "LandShape": "Semi-rectangular Shaped",
  "Area": "290",
  "UnitPrice": "75000",
  "CityPlanning": "Category I Exclusively Low-story Residential Zone",
  "CoverageRatio": "50",
  "FloorAreaRatio": "80",
  "Year": "2007",
  "Quarter": "1",
  "PriceClassification": "01"
}
```

**Notebook integration**
- Loaded into `transactions_df` for descriptive stats and Plotly line/bar charts.
- Optional Sendai-only filter keeps wards `{'4101','4102','4103','4104','4105'}` when `APPLY_SENDAI_FILTER=True`.

### 4.3 XIT002 – Prefecture Municipality Directory
**Purpose**: Provide human-readable names for municipality codes; essential for ward-level reporting and map legends.

**Raw GET string**
```text
https://www.reinfolib.mlit.go.jp/ex-api/external/XIT002?area=04&language=en
```

**Request payload**
```json
{
  "area": "04",
  "language": "en"
}
```

**Key query parameters**
| Parameter | Required | Example | Notes |
| --- | --- | --- | --- |
| `area` | Yes | `04` | Prefecture code. |
| `language` | Optional | `en` | English or Japanese labels. |

**Python helper signature**
```python
def fetch_municipalities(pref_code: str, *, language: str = 'en') -> list[dict]:
    """Single XIT002 call; cached under notebooks/data/reference/."""
```

**Response excerpt**
```json
[
  { "id": "04100", "name": "Sendai City" },
  { "id": "04101", "name": "Aoba Ward" },
  { "id": "04102", "name": "Miyagino Ward" },
  { "id": "04103", "name": "Wakabayashi Ward" }
]
```

**Notebook integration**
- Joined to trades for ward-level summaries.
- Drives table outputs in proposal write-ups.

### 4.4 XCT001 – Appraisal Report Information
**Purpose**: Retrieve official land appraisal sheets with rich context (surrounding land use, infrastructure availability, nearest station, coverage ratios). Ideal for validating trends and providing narrative colour.

**Raw GET string**
```text
https://www.reinfolib.mlit.go.jp/ex-api/external/XCT001?area=04&division=00&year=2024&language=en
```

**Request payload**
```json
{
  "area": "04",
  "division": "00",
  "year": "2024",
  "language": "en"
}
```

**Key query parameters**
| Parameter | Required | Example | Notes |
| --- | --- | --- | --- |
| `area` | Yes | `04` | Prefecture code. |
| `division` | Yes | `00` | Land use division (e.g., `00` = residential). |
| `year` | Yes | `2024` | Appraisal year. |
| `language` | Optional | `en` | English output is partial; many keys remain Japanese. |

**Python helper signature**
```python
def fetch_appraisal_records(year: int, pref_code: str, *,
                            division: str = '00', language: str = 'en') -> list[dict]:
    """Wraps XCT001, caches the response, and surfaces valuation context."""
```

**Response excerpt**
```json
{
  "価格時点": "2024",
  "標準地番号 市区町村コード 県コード": "4",
  "標準地番号 市区町村コード 市区町村コード": "101",
  "標準地番号 地域名": "仙台青葉",
  "標準地番号 用途区分": "住宅地",
  "標準地番号 連番": "1",
  "１㎡当たりの価格": "302000",
  "路線価 年": "2023",
  "路線価 相続税路線価": "225000",
  "路線価 倍率": "0"
}
```

**Notebook integration**
- Parsed into `appraisal_df`; price strings are cleaned with `parse_numeric_value`.
- Supports cross-checking transaction medians and writing narrative callouts (e.g., higher valuations near subway corridors).

### 4.5 XPT002 – Land Price Point Tiles (GeoJSON)
**Purpose**: Fetch parcel-level valuation points as slippy tiles. Enables interactive maps, spatial clustering, and animation across years.

**Raw GET string**
```text
https://www.reinfolib.mlit.go.jp/ex-api/external/XPT002?response_format=geojson&z=13&x=7301&y=3152&year=2024&useCategoryCode=00,03,05
```

**Request payload**
```json
{
  "response_format": "geojson",
  "z": "13",
  "x": "7301",
  "y": "3152",
  "year": "2024",
  "useCategoryCode": "00,03,05"
}
```
Add `priceClassification` if you need to separate transaction versus contract valuations.

**Key query parameters**
| Parameter | Required | Example | Notes |
| --- | --- | --- | --- |
| `response_format` | Yes | `geojson` | Choose between GeoJSON and vector tiles. |
| `z` | Yes | `13` | Zoom level (Web Mercator). |
| `x` | Yes | `7301` | Tile x index. |
| `y` | Yes | `3152` | Tile y index. |
| `year` | Yes | `2024` | Data year. |
| `useCategoryCode` | Optional | `00,03,05` | Comma-separated land use categories. |
| `priceClassification` | Optional | `01` | Mirrors XIT001 but rarely required. |

**Python helper signature**
```python
def collect_land_price_features(*, center_lat: float, center_lon: float,
                                zoom: int, year: int, tile_radius: int = 0,
                                price_classification: str | None = None,
                                use_category_codes: str | None = None,
                                response_format: str = 'geojson') -> dict:
    """Compute tile indices, iterate a (2r+1)^2 grid, and return
    `{'geojson': FeatureCollection, 'dataframe': DataFrame, 'tiles': list}`."""
```
Helper highlights:
- Uses `slippy_tile_index`/`slippy_tile_bounds` for deterministic coverage.
- Logs feature counts per tile and caches GeoJSON under `notebooks/data/raw/`.
- Downstream cells convert unit-bearing strings (e.g., `"77,700(?/?)"`) via `parse_numeric_value` for reliable markers.

**Response excerpt**
```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [140.84022045135498, 38.28235738797571]
  },
  "properties": {
    "location_number_ja": "川内明神横丁１５番１外",
    "area_division_name_ja": "市街化区域",
    "city_code": "04101",
    "u_road_distance_to_nearest_station_name_ja": "700m",
    "building_structure_name_ja": "W（木造）",
    "regulations_use_category_name_ja": "第二種住居地域"
  }
}
```

**Notebook integration**
- Concatenates all tiles into `land_price_df` and derives `price_per_sqm` plus `estimated_site_price`.
- Map cells drop rows with missing numeric values before calling Plotly, eliminating the invalid-marker crash seen in the first draft notebook.

### 4.6 Shared helper utilities
- `load_env_variable(name, env_path='.env')`: resolves secrets locally while keeping `.env` out of version control.
- `call_mlit_api(endpoint, params)`: central logging/error handling, treats 204/404 as empty datasets, and raises on unexpected statuses.
- `parse_numeric_value(value)`: strips thousands separators and unit suffixes (`(?/?)`, `(?)`), returning floats or `None`.
- `slippy_tile_index` / `slippy_tile_bounds`: translate lon/lat to Web Mercator indices and bounding boxes for tile iteration.

## 5. Data Flow and Cleaning Logic
1. **Fetch and cache**: Every helper writes the raw payload to disk so proposal work continues offline or during API downtime.
2. **DataFrame ingestion**: JSON payloads convert to pandas DataFrames with consistent column names (`Year`, `Quarter`, `MunicipalityCode`, etc.).
3. **Numeric coercion**: Key numeric columns (`TradePrice`, `LandArea`, `u_current_years_price_ja`, `u_cadastral_ja`) are converted via `pd.to_numeric(..., errors='coerce')` or `parse_numeric_value`.
4. **Derived metrics**:
   - `price_per_sqm` for transactions (`TradePrice / LandArea`).
   - `deal_quarter` string labels (`YYYY Q#`) for chart axes.
   - `estimated_site_price` for land tiles (`current_price * parcel_area_sqm`).
5. **Filtering**: Configurable ward list, year range, price range, and parcel size filters keep visuals focused on Sendai.
6. **Visualization safeguards**: Map-building cells drop rows with missing marker attributes before invoking Plotly, preventing NaN-related crashes.

## 6. Notebook Outputs & Early Insights
- 74,000+ trade records retrieved for Miyagi (2007Q1–2025Q4) with caching to avoid re-downloading.
- 888 land price features (2023–2025) covering a 3×3 tile grid centred on Sendai Station; ready for animation across years.
- Appraisal records confirm valuation levels near Sendai’s subway corridor, supporting comparisons against transaction medians.
- Municipality lookup confirms five Sendai wards (`Aoba`, `Miyagino`, `Wakabayashi`, `Izumi`, `Taihaku`) as the primary filters for proposal analyses.

## 7. Validation & Quality Checks
- Compare computed median trade prices to MLIT annual summaries to ensure order-of-magnitude agreement.
- Spot-check ward codes (`MunicipalityCode`) using XIT002 output.
- Inspect random land tile features to verify station distance strings parse correctly when converted to kilometres.
- Ensure map figures render without warnings after any parameter change—failed markers usually indicate unparsed numeric text.

## 8. Next Steps
1. Extend the proposal narrative with insights from the latest cached payloads (highlight price pockets, year-over-year changes, and proximity patterns).
2. Decide whether to ingest Sendai GTFS feeds; helpers already support joining additional accessibility measures.
3. Build progress-report visuals directly from cached JSON so reviewers can reproduce results without live API calls.
4. Document any rate-limit or outage observations during mid-term checkpoints.


### 4.6 Map & Facility Tile APIs (XPT001, XPT002, XKT001-XKT025)

These endpoints share the same slippy-tile semantics. `mlit_api_demo.EndpointSpec` seeds demo-friendly defaults of
`response_format='geojson'`, `z=13`, `x=7301`, `y=3152` (the Sendai Station tile). Override those values to move the
window or tighten filters.

| Code | What it returns | Required parameters | Optional parameters |
| --- | --- | --- | --- |
| XPT001 | Transaction price points (tile feed) | response_format, z, x, y, from, to | priceClassification, landTypeCode |
| XPT002 | Land price survey points | response_format, z, x, y, year | priceClassification, useCategoryCode |
| XKT001 | City planning areas / classifications | response_format, z, x, y | None |
| XKT002 | City planning zoning (用途地域) | response_format, z, x, y | None |
| XKT003 | Location optimisation plan layers | response_format, z, x, y | None |
| XKT004 | Elementary school districts | response_format, z, x, y | administrativeAreaCode |
| XKT005 | Junior high school districts | response_format, z, x, y | administrativeAreaCode |
| XKT006 | School facilities | response_format, z, x, y | None |
| XKT007 | Nursery / kindergarten facilities | response_format, z, x, y | None |
| XKT010 | Medical institutions | response_format, z, x, y | None |
| XKT011 | Welfare facilities | response_format, z, x, y | administrativeAreaCode, welfareFacilityClassCode, welfareFacilityMiddleClassCode, welfareFacilityMinorClassCode |
| XKT013 | Future population (250 m mesh) | response_format, z, x, y | None |
| XKT014 | Fire / quasi-fire zones | response_format, z, x, y | None |
| XKT015 | Station passenger volumes | response_format, z, x, y | None |
| XKT016 | Disaster hazard areas | response_format, z, x, y | administrativeAreaCode |
| XKT017 | Libraries | response_format, z, x, y | administrativeAreaCode |
| XKT018 | Municipal offices & community centres | response_format, z, x, y | None |
| XKT019 | Natural park areas | response_format, z, x, y | prefectureCode, districtCode |
| XKT020 | Large-scale landfill / embankment risk | response_format, z, x, y | None |
| XKT021 | Landslide prevention districts | response_format, z, x, y | prefectureCode, administrativeAreaCode |
| XKT022 | Steep slope collapse hazard areas | response_format, z, x, y | prefectureCode, administrativeAreaCode |
| XKT023 | District plans | response_format, z, x, y | None |
| XKT024 | High-use districts | response_format, z, x, y | None |
| XKT025 | Liquefaction propensity map | response_format, z, x, y | None |

See `notebooks/mlit_api_demo.py` for the complete spec catalogue used by the demo notebook, and adjust the defaults as
needed for production workloads.

## Heilmeier Catechism

1. **What are you trying to do?** Build a repeatable, well-documented workflow that pulls every publicly documented MLIT land, facilities, and transaction endpoint, cleans the payloads, and exposes them through a lightweight notebook so Sendai-focused analysts can blend property markets with mobility and hazard context without bespoke scripting.
2. **How is it done today, and what are the limits of current practice?** Teams usually wire one or two endpoints manually, cache ad-hoc CSV exports, and stop at transaction prices. Those pipelines are brittle (parameter changes, Japanese-only keys) and ignore the rich spatial tiles for schools, medical sites, zoning, and hazards, so cross-domain reasoning stalls.
3. **What is new in your approach, and why will it be successful?** We catalogued all 27 MLIT endpoints, captured required parameters and defaults in code, validated them by running the demo notebook end-to-end, and normalised both JSON and GeoJSON payloads for pandas. Input validation and encoded Japanese strings keep the plumbing stable, letting analysts focus on insight generation.
4. **Who cares?** Sendai urban planners, resilience researchers, real-estate analysts, and mobility practitioners who need defensible evidence to argue for zoning tweaks, transit upgrades, or hazard mitigation.
5. **If you are successful, what difference will it make, and how will you measure it?** Success means stakeholders can reproducibly answer �where should we intervene next?� in hours rather than weeks. We will measure progress via (a) complete API coverage (already demonstrated in `test_notebook.ipynb`), (b) turnaround time for two showcase analyses (ward-level housing dynamics and station-area resilience), and (c) structured feedback from at least two domain partners on clarity and usability.
6. **What are the risks and payoffs?** Risks include API throttling, schema shifts, or untranslated fields that slow non-Japanese analysts. The payoff is a sustainable data spine for Sendai and a template other Japanese cities can clone with minimal edits.
7. **How much will it cost?** Apart from staff time (approximately 80 graduate-research hours), expenses are negligible: API access is free, storage lives in the existing repo, and compute runs on personal machines or campus notebook servers.
8. **How long will it take?** Remaining deliverables�two analysis narratives, polished visuals, and a final decision-maker brief�fit within the semester (~6 weeks). The ingestion and validation foundation is complete.
9. **What are the midterm and final exams to check for success?** Midterm: a working notebook/dashboard bundle that renders the two showcase analyses with fresh pulls and documented parameters. Final: a reproducible release containing the helper module, scripted smoke tests across endpoints, the analysis notebooks, and a short decision-maker brief demonstrating how the insights informed or confirmed policy choices.
