import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Global Development Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background-color: #f9fbfd; }
h1, h2, h3 { color: #1f4e79; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Global Development Categorization Dashboard")
st.caption("Interactive ML dashboard for global socio-economic analysis")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "World_development_mesurement.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "model")
COUNTRY_COL = "Country"

# =========================
# LOAD DATA WITH ERROR HANDLING
# =========================
@st.cache_data
def load_data():
    """Load and validate dataset"""
    try:
        df = pd.read_excel(DATA_PATH)
        if df.empty:
            st.error("❌ Dataset is empty!")
            st.stop()
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {DATA_PATH}")
        st.info("Make sure 'World_development_mesurement.xlsx' is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

df = load_data()


# =========================
# LOAD MODELS WITH ERROR HANDLING
# =========================
@st.cache_resource
def load_models():
    """Load preprocessor objects"""
    try:
        scaler  = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"),   "rb"))
        imputer = pickle.load(open(os.path.join(MODEL_DIR, "imputer.pkl"),  "rb"))
        features = pickle.load(open(os.path.join(MODEL_DIR, "features.pkl"), "rb"))
        return scaler, imputer, features
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e.filename}")
        st.info("Run 'train_model.py' first to generate model files")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

scaler, imputer, features = load_models()

# =========================
# PREPROCESS DATA — matches train_model.py cleaning logic exactly
# =========================
@st.cache_data
def get_clean_data(_df, _features):
    """
    Clean data the same way train_model.py did:
    - Explicit special-char removal for mixed-type columns
    - Convert to numeric
    - Drop rows with any missing feature value
    Returns two independent copies: one for raw display, one for transforming.
    """
    df_clean = _df.copy()

    # BUG FIX 5: Use the SAME specific cleaning as train_model.py
    # (not a generic regex that behaves differently)
    special_char_cols = ['Business Tax Rate', 'GDP', 'Health Exp/Capita',
                         'Tourism Inbound', 'Tourism Outbound']
    for col in special_char_cols:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )

    # Convert all feature columns to numeric
    for col in _features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Drop rows missing any feature
    df_clean = df_clean.dropna(subset=_features).reset_index(drop=True)

    return df_clean.copy(), df_clean.copy()

df_raw, df_processed = get_clean_data(df, features)


# =========================
# TRANSFORM DATA FOR CLUSTERING
# BUG FIX 2: Order now matches train_model.py exactly:
#   impute → log transform → scale
# (original app.py was doing log → impute → scale which is wrong)
# =========================
@st.cache_data
def transform_data(_df, _features, _imputer, _scaler):
    try:
        data = _df[_features].copy()

        # STEP 1: Impute first (matches train_model.py)
        X_imputed = _imputer.transform(data.values)
        df_imputed = pd.DataFrame(X_imputed, columns=_features)

        # STEP 2: Log transform after imputing (matches train_model.py)
        log_cols = ['Business Tax Rate', 'Hours to do Tax', 'Tourism Outbound',
                    'Energy Usage', 'GDP', 'Health Exp/Capita', 'Tourism Inbound',
                    'CO2 Emissions', 'Days to Start Business', 'Population Total',
                    'Lending Interest']
        for col in log_cols:
            if col in df_imputed.columns:
                df_imputed[col] = np.log1p(df_imputed[col])

        # STEP 3: Scale
        X_scaled = _scaler.transform(df_imputed.values)
        return X_scaled

    except Exception as e:
        st.error(f"❌ Transformation error: {str(e)}")
        st.stop()

X = transform_data(df_processed, features, imputer, scaler)


# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("🎛 Dashboard Controls")

k = st.sidebar.slider("🔁 Number of Clusters (k)", 2, 6, 3)

selected_country = st.sidebar.selectbox(
    "🌍 Select Country",
    sorted(df_processed[COUNTRY_COL].unique())
)

selected_features = st.sidebar.multiselect(
    "📊 Select Indicators",
    features,
    default=features[:4]
)

# =========================
# LIVE KMEANS TRAINING (CACHED)
# =========================
@st.cache_data
def run_kmeans(_X, n_clusters):
    """Train KMeans with caching to avoid recomputation"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(_X)
    return labels

cluster_labels = run_kmeans(X, k)
df_processed["Cluster"] = cluster_labels
df_raw["Cluster"] = cluster_labels

# =========================
# CLUSTER LABELING (GDP-BASED)
# =========================
cluster_means = df_processed.groupby("Cluster")["GDP"].mean()
cluster_order = cluster_means.sort_values().index

label_pool = [
    "Under-Developed",
    "Developing",
    "Developed",
    "High Income",
    "Very High Income",
    "Elite"
]

cluster_map = dict(zip(cluster_order, label_pool[:k]))
df_processed["Development Level"] = df_processed["Cluster"].map(cluster_map)
df_raw["Development Level"] = df_raw["Cluster"].map(cluster_map)

# =========================
# METRICS
# =========================
sil_score = round(silhouette_score(X, df_processed["Cluster"]), 3)

k1, k2, k3 = st.columns(3)
k1.metric("🌍 Countries", len(df_processed))
k2.metric("🟢 Developed %",
          round((df_processed["Development Level"] == "Developed").mean() * 100, 2))
k3.metric("📈 Silhouette Score", sil_score)

st.divider()

# =========================
# 🌍 WORLD MAP
# =========================
st.subheader("🌍 Global Development Map")

world_map = px.choropleth(
    df_processed,
    locations="Country",
    locationmode="country names",
    color="Development Level",
    hover_name="Country",
    color_discrete_sequence=px.colors.qualitative.Set2
)

st.plotly_chart(world_map, use_container_width=True)

st.divider()

# =========================
# 📍 COUNTRY DRILLDOWN MAP
# =========================
st.subheader("📍 Country-Level Drilldown")

drill_df = df_processed[df_processed[COUNTRY_COL] == selected_country]

if drill_df.empty:
    st.warning(f"⚠️ No data available for {selected_country}")
else:
    drill_map = px.choropleth(
        drill_df,
        locations="Country",
        locationmode="country names",
        color="Development Level",
        hover_name="Country",
        scope="world"
    )
    st.plotly_chart(drill_map, use_container_width=True)

st.divider()

# =========================
# 📊 CATEGORY TABLES
# =========================
st.subheader("📊 Country Categorization")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🔴 Under-Developed")
    under_dev = df_processed[df_processed["Development Level"] == "Under-Developed"][[COUNTRY_COL]]
    if under_dev.empty:
        st.info("No countries in this category")
    else:
        st.dataframe(under_dev, height=260)

with c2:
    st.markdown("### 🟡 Developing")
    developing = df_processed[df_processed["Development Level"] == "Developing"][[COUNTRY_COL]]
    if developing.empty:
        st.info("No countries in this category")
    else:
        st.dataframe(developing, height=260)

with c3:
    st.markdown("### 🟢 Developed")
    developed = df_processed[df_processed["Development Level"] == "Developed"][[COUNTRY_COL]]
    if developed.empty:
        st.info("No countries in this category")
    else:
        st.dataframe(developed, height=260)

st.divider()

# =========================
# 📈 PIE CHART
# =========================
st.subheader("📈 Development Category Distribution")

category_counts = df_processed["Development Level"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

pie_fig = px.pie(
    category_counts,
    names="Category",
    values="Count",
    hole=0.4
)

st.plotly_chart(pie_fig, use_container_width=True)

st.divider()

# =========================
# 📌 RADAR CHART (SAFE & SCALED)
# =========================
st.subheader("📌 Relative Performance (Country vs Cluster Average)")

country_row = df_processed[df_processed[COUNTRY_COL] == selected_country]

if country_row.empty:
    st.warning(f"⚠️ {selected_country} data is incomplete and was skipped during cleaning.")
else:
    idx = country_row.index[0]
    country_scaled = X[idx]

    cluster_id = cluster_labels[idx]
    cluster_avg_scaled = X[cluster_labels == cluster_id].mean(axis=0)

    if selected_features:
        feat_indices = [features.index(f) for f in selected_features]

        radar_df = pd.DataFrame({
            "Indicator": selected_features,
            "Country (Scaled)": [country_scaled[i] for i in feat_indices],
            "Cluster Avg (Scaled)": [cluster_avg_scaled[i] for i in feat_indices]
        })

        radar_fig = px.line_polar(
            radar_df.melt(id_vars="Indicator"),
            r="value",
            theta="Indicator",
            color="variable",
            line_close=True,
            title=f"Relative Performance: {selected_country} vs Cluster {cluster_id}"
        )
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Please select at least one indicator from the sidebar.")

# =========================
# 📊 DISTRIBUTION PLOT
# =========================
st.subheader("📊 Indicator Distribution")

dist_feature = st.selectbox("Select Indicator", features)

box_fig = px.box(
    df_processed,
    y=dist_feature,
    color="Development Level",
    points="all"
)

st.plotly_chart(box_fig, use_container_width=True)

st.divider()

# =========================
# 📌 SINGLE COUNTRY ANALYSIS (USING RAW DATA)
# BUG FIX 1: row_raw was used but never defined — fixed here
# =========================
st.subheader("📌 Single Country Analysis")

global_mean_raw = df_raw[features].mean()

# BUG FIX 1: Define row_raw from the selected country in df_raw
country_raw_rows = df_raw[df_raw[COUNTRY_COL] == selected_country]
if not country_raw_rows.empty:
    row_raw = country_raw_rows[features].iloc[0]
else:
    row_raw = pd.Series([np.nan] * len(features), index=features)

def improvement_label(value, mean):
    if pd.isna(value) or pd.isna(mean):
        return "No Data"
    if value < mean * 0.8:
        return "Needs Improvement"
    elif value > mean * 1.2:
        return "Good"
    else:
        return "Average"

analysis_df = pd.DataFrame([
    {
        "Indicator": f,
        "Value": round(row_raw[f], 2) if pd.notna(row_raw[f]) else "N/A",
        "Global Average": round(global_mean_raw[f], 2) if pd.notna(global_mean_raw[f]) else "N/A",
        "Status": improvement_label(row_raw[f], global_mean_raw[f])
    } for f in features
])

st.dataframe(analysis_df, use_container_width=True)

# =========================
# ⬇️ DOWNLOAD REPORT
# =========================
st.download_button(
    "⬇️ Download Country Report (CSV)",
    data=analysis_df.to_csv(index=False),
    file_name=f"{selected_country}_development_report.csv"
)

st.success("Interactive clustering & exploration complete")
