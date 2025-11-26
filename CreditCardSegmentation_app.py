# ----------------------------------------------------------
# CreditCardSegmentation_app.py
# Full Streamlit app with ALL graphs
# ----------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------- Styling ----------
sns.set(context="notebook", style="whitegrid")
pd.set_option("display.max_columns", 100)

st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title { font-size: 30px; font-weight: 700; }
    .subtitle { font-size: 16px; color: #666; margin-bottom: 15px; }
    .metric-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        text-align: center;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper functions ----------

def load_data(path_or_file):
    """Load CSV from path or uploaded file."""
    return pd.read_csv(path_or_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-style cleaning: drop CUST_ID, numeric, fill NaN, drop duplicates."""
    df = df.copy()

    if "CUST_ID" in df.columns:
        df = df.drop("CUST_ID", axis=1)

    df = df.apply(pd.to_numeric, errors="coerce")

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    df = df.drop_duplicates()

    return df


def scale_features(df: pd.DataFrame):
    """StandardScaler on numeric columns."""
    df_num = df.select_dtypes(include=["number"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    return df_num, X_scaled


def choose_k(X_scaled, k_min=2, k_max=10):
    """Compute inertia + silhouette for k in [k_min..k_max] and choose best."""
    ks = list(range(k_min, k_max + 1))
    inertias = []
    sils = []
    best_k = None
    best_sil = -1

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        sils.append(sil)

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return best_k, ks, inertias, sils


def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    return km, labels


def pca_plot(X_scaled, labels):
    """2D PCA scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Cluster Plot")
    return fig


def cluster_profile(df_with_clusters, cluster_col="Cluster"):
    """Mean of key columns per cluster."""
    key_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT"]
    key_cols = [c for c in key_cols if c in df_with_clusters.columns]
    if not key_cols:
        return None
    return df_with_clusters.groupby(cluster_col)[key_cols].mean().round(2)


def corr_heatmap(df: pd.DataFrame):
    """Correlation heatmap."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def boxplot_feature(df, feature, cluster_col="Cluster"):
    """Boxplot of one feature by cluster."""
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(data=df, x=cluster_col, y=feature, ax=ax)
    ax.set_title(f"{feature} by {cluster_col}")
    return fig

# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    use_sample = st.checkbox(
        "Use sample dataset (CC_General.csv in this folder)",
        value=True,
    )

    auto_k = st.checkbox("Automatically choose best k (silhouette)", value=True)
    manual_k = st.slider("Or choose k manually:", 2, 10, 4)

    show_eda = st.checkbox("Show EDA heatmap", value=True)
    show_boxplots = st.checkbox("Show boxplots per cluster", value=True)

    st.markdown("---")
    st.caption("Extended from your Jupyter notebook 💚")


# ---------- Main title ----------

st.markdown(
    "<div class='big-title'>💳 Credit Card Customer Segmentation</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Upload data → clean → scale → choose k → cluster → interpret segments.</div>",
    unsafe_allow_html=True,
)

# ---------- Load data ----------

df_raw = None

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "CC_General.csv")

st.write("Sample file location:", sample_path)

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
elif use_sample:
    if os.path.exists(sample_path):
        df_raw = load_data(sample_path)
    else:
        st.error("Sample file NOT FOUND. Put CC_General.csv in the same folder.")
        st.stop()

if df_raw is None:
    st.error("No dataset loaded! Upload a CSV or tick 'Use sample dataset'.")
    st.stop()

# Basic info
n_rows, n_cols = df_raw.shape
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"<div class='metric-card'>Rows<br><b>{n_rows}</b></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card'>Columns<br><b>{n_cols}</b></div>",
        unsafe_allow_html=True,
    )

st.write("### 👀 Raw Data Preview")
st.dataframe(df_raw.head(), use_container_width=True)

# ---------- EDA Heatmap ----------

if show_eda:
    st.write("### 🔍 Correlation Heatmap (EDA)")
    fig_corr = corr_heatmap(df_raw)
    st.pyplot(fig_corr)

# ---------- Clean & scale ----------

st.write("### 🧹 Cleaning & Scaling")
df_clean = clean_data(df_raw)
df_num, X_scaled = scale_features(df_clean)
st.write("Shape after cleaning:", df_clean.shape)

# ---------- Choose k ----------

st.write("### 🔢 Choosing Number of Clusters (k)")

if auto_k:
    best_k, ks, inertias, sils = choose_k(X_scaled)
    k_to_use = best_k
    st.success(f"Best k by silhouette score: **{best_k}**")

    # Elbow plot
    fig_elbow, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(ks, inertias, marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Plot (K-Means)")
    st.pyplot(fig_elbow)

    # Silhouette plot
    fig_sil, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(ks, sils, marker="o")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette vs k")
    st.pyplot(fig_sil)

else:
    k_to_use = manual_k
    st.info(f"Using manual k = **{k_to_use}**")

# ---------- Final clustering ----------

st.write("### 🎯 Final Clustering")

km, labels = run_kmeans(X_scaled, k_to_use)
df_clusters = df_clean.copy()
df_clusters["Cluster"] = labels

st.write("#### Cluster Sizes")
st.write(df_clusters["Cluster"].value_counts().sort_index())

st.write("#### PCA Scatter Plot of Clusters")
fig_pca = pca_plot(X_scaled, labels)
st.pyplot(fig_pca)

st.write("#### Cluster Profile (Mean values)")
profile = cluster_profile(df_clusters)
if profile is not None:
    st.dataframe(profile)
else:
    st.write("Profile columns not found in dataset.")

# ---------- Boxplots per cluster ----------

if show_boxplots:
    st.write("### 📦 Boxplots per Cluster")

    features_to_plot = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT"]
    features_to_plot = [f for f in features_to_plot if f in df_clusters.columns]

    for feat in features_to_plot:
        st.write(f"#### {feat}")
        fig_box = boxplot_feature(df_clusters, feat)
        st.pyplot(fig_box)

# ---------- Download ----------

st.write("### 💾 Download Clustered Dataset")
csv = df_clusters.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV with clusters", csv, "credit_card_clusters.csv")
