"""
utils/data_loader.py
=====================
Loads the official six-model evaluation metrics (extracted from the
project's MODEL_C.xlsx comparison workbook) and provides small helpers
for ranking / best-model lookups used across the Executive Dashboard and
Model Comparison pages.
"""

import pandas as pd                       # DataFrame handling for the metrics table
import streamlit as st                    # provides @st.cache_data for caching the loaded CSV

from config import METRICS_CSV            # path to the pre-computed model_metrics.csv file


@st.cache_data(show_spinner=False)         # cache so the CSV is only read from disk once per session
def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)                                   # read the raw metrics CSV into a DataFrame
    # The official project ranking lives in the CSV "rank" column
    # (1 FBCL, 2 Meta-SGD, 3 2D CNN, 4 LSTM, 5 Transformer, 6 MAML).
    # It is used as-is so every page, chart and the Manage Agent show the same order.
    df["computed_rank"] = df["rank"].astype(int)
    df = df.sort_values("computed_rank").reset_index(drop=True)      # best -> worst
    return df


def best_model_row(df: pd.DataFrame = None) -> pd.Series:
    df = df if df is not None else load_metrics()          # use provided DataFrame or load metrics fresh if none given
    return df.sort_values("computed_rank").iloc[0]  # return the single row with the highest average score


def get_row(model_name: str, df: pd.DataFrame = None) -> pd.Series:
    df = df if df is not None else load_metrics()           # use provided DataFrame or load metrics fresh if none given
    match = df[df["model_name"] == model_name]               # filter to the row(s) matching the requested model name
    if match.empty:
        raise KeyError(f"No metrics row for model '{model_name}'")  # fail loudly if the model name isn't found
    return match.iloc[0]                                       # return the first (only) matching row
