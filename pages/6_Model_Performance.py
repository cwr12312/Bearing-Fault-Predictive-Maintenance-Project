"""Model Performance — confusion matrix and evaluation metrics.

Evaluates each model on the project's real held-out test set when available.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from config import (
    PROJECT_SHORT_TITLE,
    PLOTLY_TEMPLATE,
    MODEL_NAMES,
    CLASS_NAMES,
    ALL_FEATURES,
    DATA_DIR,
)
from utils.styling import (
    inject_css,
    page_header,
    section_title,
    footer,
    render_sidebar,
)
from utils.model_loader import get_model
from utils.preprocessing import generate_synthetic_dataset, scale_features
from utils.data_loader import get_row


# --------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=f"Model Performance · {PROJECT_SHORT_TITLE}",
    layout="wide",
)

inject_css()
render_sidebar(active="model_performance")
page_header(
    "Model Performance",
    "Deep dive into any single model's evaluation behaviour",
)


# --------------------------------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------------------------------

model_name = st.selectbox("Select model", MODEL_NAMES, index=1)
wrapper = get_model(model_name)  # Cached load of the model's weights/architecture, or None on failure.
official = get_row(model_name)  # Pre-computed metrics row for this model from data/model_metrics.csv.

if wrapper is None:
    st.error(f"{model_name} failed to load. Showing official metrics only.")
else:
    st.success(f"{model_name} loaded successfully. Live evaluation is available below.")


# --------------------------------------------------------------------------
# OFFICIAL METRICS
# --------------------------------------------------------------------------
# The pre-recorded metrics for this model from the comparison workbook,
# shown regardless of whether the model itself loaded successfully.

c1, c2, c3, c4 = st.columns(4)

c1.metric("Official Accuracy", f"{official['accuracy'] * 100:.2f}%")
c2.metric("Official Precision", f"{official['precision'] * 100:.2f}%")
c3.metric("Official Recall", f"{official['recall'] * 100:.2f}%")
c4.metric("Official F1 Score", f"{official['f1_score'] * 100:.2f}%")


# --------------------------------------------------------------------------
# LIVE EVALUATION
# --------------------------------------------------------------------------
# Everything below only runs if the model actually loaded, since it needs a
# working `wrapper` to call .predict() on the held-out test set.

if wrapper is not None:

    @st.cache_data(show_spinner=False)
    def eval_set():
        """
        Load the project's real held-out test set when available.
        Falls back to a synthetic dataset only if the test file is missing.
        """

        real_path = DATA_DIR / "test_set_scaled.csv"

        if real_path.exists():
            df = pd.read_csv(real_path)

            X = df[ALL_FEATURES].values.astype(np.float32)
            y = df["fault_encoded"].astype(int).values

            return X, y

        # Fallback only if the real test dataset is unavailable — build a
        # small synthetic labeled set and scale it the same way real data would be.
        df = generate_synthetic_dataset(n_per_class=40, seed=99)
        X = scale_features(df[ALL_FEATURES])
        y = df["fault_class"].map(
            lambda c: CLASS_NAMES.index(c)
        ).values

        return X, y


    # Load evaluation data and run prediction — this is a REAL model inference
    # pass over the (real or fallback) test set, not a cached/pre-computed result.
    X_eval, y_eval = eval_set()
    y_pred, y_probs = wrapper.predict(X_eval)


    # ----------------------------------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------------------------------

    tab1, = st.tabs(["Confusion Matrix"])

    with tab1:
        section_title("Confusion Matrix")

        # Rows = true class, columns = predicted class; diagonal = correct predictions.
        cm = confusion_matrix(
            y_eval,
            y_pred,
            labels=range(len(CLASS_NAMES)),
        )

        fig = px.imshow(
            cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
        )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted",
            yaxis_title="True",
        )

        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------

footer()