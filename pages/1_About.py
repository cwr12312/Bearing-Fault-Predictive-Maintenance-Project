"""About page  project metadata, objectives, methodology, architecture, credits.
Positioned near the top of the navigation, directly after "App"."""

import streamlit as st

# Static, hard-coded project metadata pulled from config.py — nothing on this
# page depends on live data or model inference.
from config import (
    PROJECT_SHORT_TITLE, PROJECT_TITLE, PROJECT_DESCRIPTION, OBJECTIVES, METHODOLOGY,
    SYSTEM_ARCHITECTURE, DATASET_INFO, DEVELOPMENT_TOOLS,
    MODEL_REGISTRY,
)
from utils.styling import inject_css, page_header, section_title, footer, render_sidebar

# Page-level Streamlit config + shared theme/sidebar, same pattern as every other page.
st.set_page_config(page_title=f"About · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
render_sidebar(active="about")
page_header("About This Project", "Capstone project metadata and technical documentation")

# Project title + one-paragraph description (config.PROJECT_TITLE / PROJECT_DESCRIPTION).
st.markdown(f"### {PROJECT_TITLE}")
st.write(PROJECT_DESCRIPTION)

st.write("")
c1, c2 = st.columns(2)
with c1:
    # Bullet list of the project's stated objectives (config.OBJECTIVES).
    section_title("Objectives")
    for o in OBJECTIVES:
        st.markdown(f"- {o}")

    # Free-text summary of how the raw vibration data becomes model input (config.METHODOLOGY).
    section_title("Methodology")
    st.write(METHODOLOGY)

with c2:
    # High-level pipeline description: Streamlit -> model registry -> preprocessing -> SHAP -> Plotly.
    section_title("System Architecture")
    st.write(SYSTEM_ARCHITECTURE)

    # Key facts about the CWRU dataset and how it was split/augmented (config.DATASET_INFO dict).
    section_title("Dataset Information")
    for k, v in DATASET_INFO.items():
        st.markdown(f"- **{k}:** {v}")

st.write("")
section_title("Deep Learning & Meta-Learning Models")
# One collapsible expander per registered model (config.MODEL_REGISTRY), showing its
# framework, a plain-language description, expected input shape, and checkpoint filename.
for name, cfg in MODEL_REGISTRY.items():
    with st.expander(f"{name}  ·  {cfg['framework']}"):
        st.write(cfg["description"])
        st.caption(f"Input shape: {cfg['input_shape']}  ·  Checkpoint: `{cfg['file'].name}`")

st.write("")
c3, c4 = st.columns(2)
with c3:
    # Comma-separated list of the libraries/tools used to build the project (config.DEVELOPMENT_TOOLS).
    section_title("Development Tools")
    st.markdown(", ".join(DEVELOPMENT_TOOLS))
footer()