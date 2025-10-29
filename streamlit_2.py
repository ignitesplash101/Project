# =====================================================
# streamlit_mesh_dashboard_cityfix.py
# Mesh-250m dashboard with proper Tokyo/Sendai filtering
# =====================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import json
import time
import itertools
import numpy as np

st.set_page_config(page_title="Japan Housing Dashboard", layout="wide")
st.title("🏙️ Japan Housing Dashboard — Mesh-250 m Dynamic Visualization")

# --------------------------
# City definitions
# --------------------------
TOKYO_WARDS = {f"131{str(i).zfill(2)}" for i in range(1,24)}  # 13101..13123
SENDAI_WARDS = {"04101","04102","04103","04104","04105"}

# fallback bounding boxes (min_lat, max_lat, min_lon, max_lon)
BBOXES = {
    "Tokyo":  (35.40, 35.92, 139.40, 139.95),
    "Sendai": (38.15, 38.40, 140.70, 141.05),
}

CENTER = {
    "Tokyo":  dict(latitude=35.68, longitude=139.76, zoom=9),
    "Sendai": dict(latitude=38.27, longitude=140.87, zoom=10),
}

# =====================================================
# 1) Load or create placeholder mesh data
# =====================================================
@st.cache_data
def load_predictions():
    try:
        df_pred = pd.read_csv("model_predictions_viz.csv")
    except Exception as e:
        st.error(f"❌ Failed to load model_predictions_viz.csv: {e}")
        df_pred = pd.DataFrame()
    return df_pred

df_pred = load_predictions()



@st.cache_data
def load_mesh_data():

    gdf = gpd.read_file("mesh250_all_quarters.geojson")

    # Ensure numeric types
    for col in [
        "mesh_mean_ppsqm", "mesh_median_ppsqm", "mesh_transaction_count",
        "mesh_avg_age", "mesh_avg_area", "PriceIndex"
    ]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Normalize MunicipalityCode if present
    if "MunicipalityCode" in gdf.columns:
        gdf["MunicipalityCode"] = gdf["MunicipalityCode"].astype(str).str.zfill(5)

    return gdf

gdf = load_mesh_data()

# =====================================================
# 2) Sidebar Controls
# =====================================================
st.sidebar.title("🏠 Japan Housing Forecast")
st.sidebar.caption("Mesh-250 m granularity · Tokyo · Sendai")

city = st.sidebar.selectbox("Select City", ["Tokyo", "Sendai"])
# Dynamically populate model list
model_list = sorted(df_pred["Model"].unique())
model = st.sidebar.selectbox("Select Model", model_list)
quarters = sorted(gdf["PeriodKey"].unique())
period = st.sidebar.select_slider("Select Quarter", options=quarters, value=quarters[0])
animate = st.sidebar.checkbox("Auto-play Animation", value=False)
metric = st.sidebar.selectbox(
    "Color by Metric",
    [
        "PriceIndex",
        "mesh_mean_ppsqm",
        "mesh_median_ppsqm",
        "mesh_transaction_count",
        "mesh_avg_age",
        "mesh_avg_area",
    ],
    index=0,
)

show_shap = st.sidebar.checkbox("Show SHAP Panel", value=True)
show_leaderboard = st.sidebar.checkbox("Show Leaderboard", value=True)

st.sidebar.markdown("---")
st.sidebar.download_button("📥 Export CSV", data=b"", file_name="forecasts.csv")

# =====================================================
# 3) KPI Section — Focused on Mesh-250 m Level
# =====================================================

# Load model metrics
df_results = pd.read_csv("model_results.csv")

# Filter only Mesh level models
mesh_models = df_results[df_results["Level"].str.lower() == "mesh"]

if not mesh_models.empty:
    best_model_mesh = mesh_models.loc[mesh_models["test_rmse"].idxmin()]
    best_model_name = best_model_mesh["Model"]
    best_model_rmse = best_model_mesh["test_rmse"]
else:
    best_model_name = "N/A"
    best_model_rmse = float("nan")

# =====================================================
# Coverage (based on mesh_quarterly_price_index.csv)
# =====================================================
try:
    price_index_df = pd.read_csv("mesh_quarterly_price_index.csv")
    # Unique meshes that have valid price index values
    coverage = price_index_df['Latitude'].notnull().sum()
    total = len(price_index_df["Latitude"])
    coverage_pct = (coverage / total * 100) if total > 0 else 0
except Exception as e:
    st.warning(f"⚠️ Unable to load mesh_quarterly_price_index.csv ({e})")
    coverage, total, coverage_pct = 0, 0, 0

# =====================================================
# Display KPIs
# =====================================================
kpi_cols = st.columns(2)
kpi_cols[0].metric(
    "Best Model for Mesh-250 m (RMSE)",
    best_model_name,
    f"{best_model_rmse:.0f}" if pd.notnull(best_model_rmse) else "N/A"
)
kpi_cols[1].metric(
    "Data Coverage (Mesh250m)",
    f"{coverage_pct:.1f}%",
    f"{coverage}/{total}"
)


# =====================================================
# City Filter helpers
# =====================================================
def filter_city(frame: gpd.GeoDataFrame, city_name: str) -> gpd.GeoDataFrame:
    """Filter meshes to Tokyo or Sendai using MunicipalityCode when available, else bbox."""
    if "MunicipalityCode" in frame.columns:
        if city_name == "Tokyo":
            return frame[frame["MunicipalityCode"].isin(TOKYO_WARDS)]
        else:
            return frame[frame["MunicipalityCode"].isin(SENDAI_WARDS)]
    # bbox fallback
    lat_min, lat_max, lon_min, lon_max = BBOXES[city_name]
    return frame[
        (frame["Latitude"] >= lat_min) & (frame["Latitude"] <= lat_max) &
        (frame["Longitude"] >= lon_min) & (frame["Longitude"] <= lon_max)
    ]

# =====================================================
# 4) Render one map frame for (city, period, model)
# =====================================================
def render_period(city_name: str, period_key: str, model_name: str):
    # Filter base mesh data for that quarter + city
    frame = gdf[gdf["PeriodKey"] == period_key].copy()
    frame = filter_city(frame, city_name)

    # Merge predictions for same model + quarter
    preds = df_pred[
        (df_pred["Model"].str.strip().str.lower() == model_name.strip().lower())
        & (df_pred["PeriodKey"] == period_key)
    ].copy()

    # Clean up Mesh ID formatting on both sides
    if "Mesh250m" in frame.columns:
        frame["Mesh250m"] = (
            frame["Mesh250m"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False)
        )

    if "Mesh250m" in preds.columns:
        preds["Mesh250m"] = (
            preds["Mesh250m"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False)
        )

    # Merge using both Mesh250m + PeriodKey if available
    if "Mesh250m" in preds.columns and "Mesh250m" in frame.columns:
        frame = frame.merge(
            preds[["Mesh250m", "PeriodKey", "Predicted"]],
            on=["Mesh250m", "PeriodKey"],
            how="left"
        )

    if frame.empty:
        st.warning(f"No data for {city_name} ({model_name}) in {period_key}")
        return

    # Format tooltip values (2 dp)
    for col in [
        "PriceIndex", "mesh_median_ppsqm",
        "mesh_mean_ppsqm", "mesh_avg_age",
        "mesh_avg_area", "Predicted"
    ]:
        if col in frame.columns:
            frame[f"{col}_str"] = frame[col].apply(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else "-"
            )

    geojson = json.loads(frame.to_json())
    max_val = frame[metric].max() or 1e-6  # Avoid div-by-zero

    # Green (cheap) → Red (expensive)
    color_expr = f"[255*(properties.{metric}/{max_val}), 255*(1 - properties.{metric}/{max_val}), 0]"

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        stroked=False,
        filled=True,
        opacity=0.8,
        get_fill_color=color_expr,
        pickable=True,
    )

    vs = pdk.ViewState(**CENTER[city_name])

    tooltip = {
        "html": (
            "<b>Mesh ID:</b> {Mesh250m}<br>"
            "<b>Quarter:</b> {PeriodKey}<br>"
            "<b>Predicted Median ¥/m²:</b> {Predicted_str}<br>"
            "<b>Actual Median ¥/m²:</b> {mesh_median_ppsqm_str}<br>"
            "<b>Mean ¥/m²:</b> {mesh_mean_ppsqm_str}<br>"
            "<b>Avg Age:</b> {mesh_avg_age_str}<br>"
            "<b>Avg Area:</b> {mesh_avg_area_str}<br>"
            "<b>Price Index:</b> {PriceIndex_str}"
        ),
        "style": {"backgroundColor": "white", "color": "black"}
    }

    deck = pdk.Deck(layers=[layer], initial_view_state=vs, tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)




# =====================================================
# 5) Main Layout — Map + Right Panels
# =====================================================
map_col, right_col = st.columns([1.6, 1])

with map_col:
    st.subheader("Mesh-250 m Dynamic Map")
    st.caption("Color by selected metric; hover for mesh-level stats.")
    if animate:
        placeholder = st.empty()
        for q in quarters:
            with placeholder.container():
                st.markdown(f"🕒 Quarter: `{q}` — {city} — Color by `{metric}`")
                render_period(city, q, model)   # ✅ Use q instead of period
            time.sleep(1.2)
    else:
        st.markdown(f"### 🕒 Quarter: `{period}` — {city} — Color by `{metric}`")
        render_period(city, period, model)
with right_col:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    if show_shap:
        st.subheader("Feature Importance (SHAP) Dummy Data")
        shap_data = pd.DataFrame(
            {"Feature": ["Area", "Age", "Type"], "Importance": [0.35, 0.25, 0.18]}
        )
        fig = px.bar(shap_data, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

if show_leaderboard:
    st.subheader("Model Leaderboard")

    # Load model results
    df_results = pd.read_csv("model_results.csv")

    # Select level (Ward or Mesh)
    levels = df_results["Level"].unique().tolist()[::-1]
    level_choice = st.radio("Select Evaluation Level", levels, horizontal=True)

    # Filter for level
    filtered = df_results[df_results["Level"] == level_choice]

    # Sort by best R² descending
    filtered = filtered.sort_values(by="test_r2", ascending=False)

    # Pretty format numbers
    numeric_cols = ["test_mae", "test_rmse", "test_r2"]
    for col in numeric_cols:
        filtered[col] = filtered[col].apply(lambda x: f"{x:,.3f}" if pd.notnull(x) else "-")

    # Display table
    st.dataframe(
        filtered[["Model", "test_mae", "test_rmse", "test_r2"]],
        hide_index=True,
        use_container_width=True,
    )

    # Highlight top model
    top_model = filtered.iloc[0]["Model"]
    top_r2 = filtered.iloc[0]["test_r2"]
    st.success(f"🏆 **Top Model:** {top_model} (R² = {top_r2})")


