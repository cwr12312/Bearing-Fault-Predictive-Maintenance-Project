"""
utils/benchmark.py
===================
Measures *actual* per-sample inference latency for each loaded model on
this machine, used by the Executive Dashboard and Model Performance pages.
Falls back to the estimated `inference_time_ms` column in
data/model_metrics.csv if a model failed to load or benchmarking fails.
"""

import numpy as np                                    # numeric ops + random number generation for the dummy input batch
import pandas as pd                                    # builds the results table returned to the caller
import streamlit as st                                 # provides the @st.cache_data decorator for caching results

from config import N_FEATURES                          # number of input features each model expects
from utils.model_loader import load_all_models         # loads/returns every trained model wrapper
from utils.preprocessing import load_reference_scaler  # imported for side effects / availability (scaler not used directly below)


@st.cache_data(show_spinner=False)                      # cache results so repeated calls don't re-benchmark every rerun
def benchmark_all_models(n_warmup: int = 2, n_runs: int = 10) -> pd.DataFrame:
    registry = load_all_models()                        # get dict with "models" (loaded ok) and "errors" (failed to load)
    models = registry["models"]                         # only benchmark models that loaded successfully

    rng = np.random.default_rng(0)                       # fixed seed -> reproducible dummy input across runs
    dummy = rng.normal(size=(8, N_FEATURES)).astype(np.float32)  # synthetic batch of 8 samples used purely for timing

    rows = []                                            # collects one result row per model
    for name, wrapper in models.items():                 # iterate over every successfully loaded model
        try:
            for _ in range(n_warmup):                    # warm-up passes to avoid cold-start bias (JIT/graph build, caching)
                wrapper.predict(dummy)
            times = []                                    # per-run measured latencies (ms per sample)
            for _ in range(n_runs):                       # repeat several times to get a stable median
                _, _, per_sample_ms = wrapper.timed_predict(dummy)  # run inference and capture timing
                times.append(per_sample_ms)
            rows.append({"model_name": name, "measured_inference_ms": float(np.median(times))})  # store median latency
        except Exception:
            rows.append({"model_name": name, "measured_inference_ms": np.nan})  # benchmarking failed -> mark as NaN, caller falls back to CSV estimate
    return pd.DataFrame(rows)                             # tabular results: one row per model
