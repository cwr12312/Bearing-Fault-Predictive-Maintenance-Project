"""Dataset Explorer — EDA views (class balance, correlation, PCA/t-SNE/UMAP, stats).

Uses the project's REAL preprocessed data files
(data/preprocessed_augmented_dataset_scaled.csv) whenever they are present,
so every statistic shown here matches the actual dataset used to train and
evaluate the six models. Falls back to an illustrative synthetic dataset
only if those files are missing.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import (
    PROJECT_SHORT_TITLE,
    COLORS,
    PLOTLY_TEMPLATE,
    ALL_FEATURES,
    DATASET_INFO,
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
    DATA_DIR,
)
from utils.styling import (
    inject_css,
    page_header,
    section_title,
    footer,
    render_sidebar,
)
from utils.preprocessing import generate_synthetic_dataset


# --------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=f"Dataset Explorer · {PROJECT_SHORT_TITLE}",
    layout="wide",
)

inject_css()
render_sidebar(active="dataset_explorer")
page_header(
    "Dataset Explorer",
    "Exploratory data analysis of the CWRU-derived feature space",
)


# --------------------------------------------------------------------------
# DATASET PATH
# --------------------------------------------------------------------------

# This defines the path to your real preprocessed training dataset. If this
# file isn't present in data/, get_dataset() below falls back to a synthetic
# stand-in so the page still renders something meaningful.
real_path = DATA_DIR / "preprocessed_augmented_dataset_scaled.csv"


# --------------------------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_dataset():
    """
    Loads the real augmented training dataset shipped with the project.

    The CSV stores `fault_encoded` (0–9) rather than the string class name,
    so the encoded values are mapped back to the class names defined in
    config.CLASS_NAMES.
    """

    if real_path.exists():
        df = pd.read_csv(real_path)

        # Convert encoded fault labels to their corresponding class names
        df["fault_class"] = df["fault_encoded"].map(
            lambda i: CLASS_NAMES[int(i)]
        )

        return df, True

    # Fallback dataset if the real file is unavailable — generates an
    # illustrative dataset with the same class balance/shape so the page
    # still has something meaningful to plot (utils.preprocessing.generate_synthetic_dataset).
    return generate_synthetic_dataset(
        n_per_class=230,
        seed=11,
    ), False


# Load the dataset (cached — only re-read/regenerated when the underlying file changes).
df, is_real = get_dataset()


# --------------------------------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------------------------------

section_title("Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Samples", f"{len(df):,}")
c2.metric("Features", str(len(ALL_FEATURES)))
c3.metric("Classes", str(df["fault_class"].nunique()))
c4.metric(
    "Missing Values",
    str(int(df[ALL_FEATURES].isna().sum().sum())),
)


# --------------------------------------------------------------------------
# DOCUMENTED DATASET FACTS
# --------------------------------------------------------------------------

# Collapsible reference panel of the dataset facts documented in the
# project's Interim Report (config.DATASET_INFO) — source, sample counts,
# operating condition, etc.
with st.expander("Documented Dataset Facts (from Interim Report)"):
    for k, v in DATASET_INFO.items():
        st.markdown(f"- **{k}:** {v}")


# --------------------------------------------------------------------------
# CLASS DISTRIBUTION
# --------------------------------------------------------------------------

st.write("")

tab1, = st.tabs(["Class Distribution"])

with tab1:
    section_title("Class Distribution")

    # Count how many samples fall into each of the 10 fault classes.
    dist = df["fault_class"].value_counts().reset_index()
    dist.columns = ["Class", "Count"]

    # Convert technical class names to display names where available
    dist["Class"] = dist["Class"].map(
        lambda c: CLASS_DISPLAY_NAMES.get(c, c)
    )

    fig = px.bar(
        dist,
        x="Class",
        y="Count",
        color="Class",
        template=PLOTLY_TEMPLATE,
    )

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_tickangle=-30,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------

footer()