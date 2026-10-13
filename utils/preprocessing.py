"""
utils/preprocessing.py
=======================
Reproduces the feature-engineering pipeline described in the Interim Report
and used by every training script (fbcl_model.py, maml.py, transfomer.py,
lstm_model.py, 2d_cnn.py):

    9 base statistical features  ->  + 10 engineered ratio/interaction terms
    ->  19-dimensional feature vector  ->  StandardScaler  ->  model input

Because the raw scaler objects (`scaler.pkl` fitted during training) were
not part of the uploaded bundle, this module fits a *reference* scaler at
app start-up from representative synthetic statistics consistent with the
dataset documentation (Table 2 / Table 4, Interim Report: CWRU, 10 balanced
classes, mean-zero / unit-variance engineered space). If the user supplies
their own `feature_scaler.pkl` / `label_encoder.pkl` in `data/`, those are
used automatically instead — see `load_reference_scaler()`.
"""

from __future__ import annotations                        # allows modern type-hint syntax (e.g. X | None) on older Python versions

import pickle                                                # loads a user-supplied scaler from disk, if present
from pathlib import Path                                     # imported for path handling (DATA_DIR usage below)

import numpy as np                                            # numeric array operations
import pandas as pd                                           # DataFrame handling throughout the pipeline
from sklearn.preprocessing import StandardScaler               # scales engineered features to zero-mean/unit-variance

from config import ALL_FEATURES, BASE_FEATURES, ENGINEERED_FEATURES, DATA_DIR  # feature name lists + data directory path


def engineer_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 9 base statistical features into the full 19-feature set
    exactly as enumerated in Table 4 of the Interim Report.

    Parameters
    ----------
    df_base : DataFrame with (at least) columns BASE_FEATURES

    Returns
    -------
    DataFrame with columns == ALL_FEATURES, in the fixed training order.
    """
    df = df_base.copy()                                        # work on a copy so the caller's DataFrame is untouched
    missing = [c for c in BASE_FEATURES if c not in df.columns]  # check that all 9 required base columns are present
    if missing:
        raise ValueError(f"Missing required base feature columns: {missing}")  # fail early with a clear message if any are missing

    eps = 1e-8  # guards against divide-by-zero on edge-case manual entries

    df["peak_to_peak"] = df["max"] - df["min"]                          # engineered feature: range between max and min
    df["rms_mean_ratio"] = df["rms"] / (df["mean"].replace(0, eps))      # engineered feature: RMS relative to mean (avoids /0)
    df["crest_form_ratio"] = df["crest"] / (df["form"].replace(0, eps))  # engineered feature: crest factor relative to form factor
    df["kurtosis_squared"] = df["kurtosis"] ** 2                          # engineered feature: emphasizes strong kurtosis (impulsiveness)
    df["skewness_abs"] = df["skewness"].abs()                             # engineered feature: magnitude of skew, direction-agnostic
    df["rms_sd_ratio"] = df["rms"] / (df["sd"].replace(0, eps))           # engineered feature: RMS relative to standard deviation
    df["max_min_ratio"] = df["max"] / (df["min"].replace(0, eps))         # engineered feature: ratio of extreme values
    df["kurtosis_crest"] = df["kurtosis"] * df["crest"]                   # engineered feature: interaction of kurtosis and crest factor
    df["mean_squared"] = df["mean"] ** 2                                   # engineered feature: squared mean (captures magnitude regardless of sign)
    df["rms_squared"] = df["rms"] ** 2                                     # engineered feature: squared RMS (proportional to signal energy)

    return df[ALL_FEATURES]                                                # return only the 19 columns, in the fixed training order


def _synthetic_reference_scaler() -> StandardScaler:
    """
    Build a StandardScaler fitted on documented, class-representative
    statistics so the pipeline is runnable end-to-end even without the
    original serialized scaler. Ranges follow the physical interpretation
    given in the Interim Report (kurtosis ~3 healthy / 5-30 faulted, RMS
    rising with severity, etc.) across the 10 balanced fault classes.
    """
    rng = np.random.default_rng(42)                             # fixed seed -> reproducible synthetic reference data
    n_per_class = 250                                             # number of synthetic samples generated per fault class
    rows = []                                                      # accumulates generated base-feature rows
    severities = [1.0, 1.6, 2.4]  # relative energy scaling for 007/014/021

    class_specs = [
        ("Ball_007", 0), ("Ball_014", 1), ("Ball_021", 2),
        ("IR_007", 0), ("IR_014", 1), ("IR_021", 2),
        ("Normal", None),
        ("OR_007_6", 0), ("OR_014_6", 1), ("OR_021_6", 2),
    ]                                                               # the 10 CWRU fault classes with their severity index
    for name, sev_idx in class_specs:                               # generate synthetic samples for each fault class
        sev = severities[sev_idx] if sev_idx is not None else 0.4    # look up severity multiplier (Normal uses a fixed low value)
        base_rms = 0.6 * sev if name != "Normal" else 0.35            # baseline RMS level scales with severity, except for Normal
        base_kurt = 3.0 if name == "Normal" else float(rng.uniform(5, 22)) * (sev / 1.6)  # Normal ~3 (Gaussian-like), faults have higher random kurtosis
        for _ in range(n_per_class):                                  # generate n_per_class synthetic rows for this class
            rms = max(0.05, rng.normal(base_rms, 0.08))                 # sampled RMS, floored to avoid non-physical negative values
            mean = rng.normal(0.0, 0.02)                                 # sampled mean, centered near zero
            sd = max(0.02, rng.normal(base_rms * 0.9, 0.05))             # sampled standard deviation, floored to stay positive
            kurt = max(1.5, rng.normal(base_kurt, base_kurt * 0.15 + 0.3))  # sampled kurtosis, floored at a physically plausible minimum
            skew = rng.normal(0.0 if name == "Normal" else 0.3, 0.4)       # sampled skewness, slight positive bias for faulted classes
            crest = max(1.0, rng.normal(3.0 if name == "Normal" else 4.5 + sev, 0.6))  # sampled crest factor, higher for more severe faults
            form = max(1.0, rng.normal(1.2, 0.1))                           # sampled form factor
            mx = rms * crest + abs(rng.normal(0, 0.05))                     # derive a plausible max value from RMS and crest factor
            mn = -mx * rng.uniform(0.7, 1.0)                                # derive a plausible min value, roughly symmetric to max
            rows.append([mx, mn, mean, sd, rms, skew, kurt, crest, form])    # store the 9 base features for this synthetic sample

    df = pd.DataFrame(rows, columns=BASE_FEATURES)                # assemble all synthetic samples into a DataFrame
    df_full = engineer_features(df)                                 # expand to the full 19-feature engineered representation
    scaler = StandardScaler()                                        # create a new scaler
    scaler.fit(df_full.values)                                        # fit it on the synthetic engineered feature distribution
    return scaler                                                      # return the fitted reference scaler


_scaler_cache: StandardScaler | None = None                       # module-level cache so the scaler is only built/loaded once per session


def load_reference_scaler() -> StandardScaler:
    """
    Loads a persisted StandardScaler from data/feature_scaler.pkl if present
    (drop your original training-time scaler there for exact reproduction),
    otherwise falls back to a synthetic, documentation-consistent reference
    scaler built on first use and cached for the session.
    """
    global _scaler_cache
    if _scaler_cache is not None:
        return _scaler_cache                                          # already loaded/built this session -> reuse cached instance

    custom_path = DATA_DIR / "feature_scaler.pkl"                     # optional path to a user-supplied, exact training-time scaler
    if custom_path.exists():
        with open(custom_path, "rb") as f:
            _scaler_cache = pickle.load(f)                              # load the user's own scaler for exact reproduction
            return _scaler_cache

    _scaler_cache = _synthetic_reference_scaler()                      # no custom scaler found -> build and cache the synthetic one
    return _scaler_cache


def scale_features(df_full: pd.DataFrame) -> np.ndarray:
    """Apply the reference StandardScaler to a full 19-feature DataFrame."""
    scaler = load_reference_scaler()                                    # get the cached/loaded scaler
    return scaler.transform(df_full[ALL_FEATURES].values.astype(np.float32))  # scale the 19 engineered features, ensuring float32 dtype


def preprocess_manual_entry(base_values: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convert a dict of the 9 base feature values (manual UI entry) into the
    engineered + scaled representation ready for any of the six models.

    Returns (engineered_unscaled_df, scaled_array)
    """
    df_base = pd.DataFrame([base_values])[BASE_FEATURES]                # wrap the single manual entry as a one-row DataFrame in the right column order
    df_full = engineer_features(df_base)                                  # expand to the 19-feature engineered representation
    scaled = scale_features(df_full)                                       # scale the engineered features
    return df_full, scaled                                                  # return both the unscaled and scaled versions


def preprocess_csv_upload(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Accepts an uploaded CSV. Two accepted layouts:
      1) Already has all 19 ALL_FEATURES columns -> used directly.
      2) Has only the 9 BASE_FEATURES columns -> engineered automatically.
    Any extra columns (e.g. a target/label column) are ignored for inference.
    """
    cols = set(df_raw.columns)                                            # set of column names present in the uploaded data
    if set(ALL_FEATURES).issubset(cols):
        df_full = df_raw[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)  # already fully engineered -> coerce to numeric, fill bad values with 0
    elif set(BASE_FEATURES).issubset(cols):
        df_base = df_raw[BASE_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)  # only base columns present -> coerce to numeric first
        df_full = engineer_features(df_base)                                # then compute the full engineered feature set
    else:
        missing = set(BASE_FEATURES) - cols                                 # figure out which base columns are missing, for the error message
        raise ValueError(
            "Uploaded CSV must contain either the 19 engineered feature columns or the 9 "
            f"base statistical columns. Missing base columns: {sorted(missing)}"
        )                                                                    # neither known layout matched -> reject with details
    scaled = scale_features(df_full)                                         # scale the engineered features
    return df_full, scaled                                                    # return both the unscaled and scaled versions


def generate_synthetic_dataset(n_per_class: int = 150, seed: int = 7) -> pd.DataFrame:
    """
    Generates an illustrative, documentation-consistent dataset (used only
    by the Dataset Explorer page for EDA visuals) since the raw CWRU CSVs
    were not part of the uploaded bundle. Distributional shape follows the
    same generative logic as `_synthetic_reference_scaler`, but returns
    unscaled features + string labels for exploration.
    """
    rng = np.random.default_rng(seed)                                       # seeded RNG for reproducible synthetic data
    severities = [1.0, 1.6, 2.4]                                              # relative energy scaling for 007/014/021 severities
    class_specs = [
        ("Ball_007", 0), ("Ball_014", 1), ("Ball_021", 2),
        ("IR_007", 0), ("IR_014", 1), ("IR_021", 2),
        ("Normal", None),
        ("OR_007_6", 0), ("OR_014_6", 1), ("OR_021_6", 2),
    ]                                                                          # the 10 CWRU fault classes with their severity index
    rows = []                                                                   # accumulates generated rows (features + label)
    for name, sev_idx in class_specs:                                          # generate synthetic samples for each fault class
        sev = severities[sev_idx] if sev_idx is not None else 0.4               # look up severity multiplier (Normal uses a fixed low value)
        base_rms = 0.6 * sev if name != "Normal" else 0.35                       # baseline RMS level scales with severity, except for Normal
        base_kurt = 3.0 if name == "Normal" else float(rng.uniform(5, 22)) * (sev / 1.6)  # Normal ~3 (Gaussian-like), faults have higher random kurtosis
        for _ in range(n_per_class):                                             # generate n_per_class synthetic rows for this class
            rms = max(0.05, rng.normal(base_rms, 0.08))                            # sampled RMS, floored to avoid non-physical negative values
            mean = rng.normal(0.0, 0.02)                                            # sampled mean, centered near zero
            sd = max(0.02, rng.normal(base_rms * 0.9, 0.05))                        # sampled standard deviation, floored to stay positive
            kurt = max(1.5, rng.normal(base_kurt, base_kurt * 0.15 + 0.3))            # sampled kurtosis, floored at a physically plausible minimum
            skew = rng.normal(0.0 if name == "Normal" else 0.3, 0.4)                  # sampled skewness, slight positive bias for faulted classes
            crest = max(1.0, rng.normal(3.0 if name == "Normal" else 4.5 + sev, 0.6))  # sampled crest factor, higher for more severe faults
            form = max(1.0, rng.normal(1.2, 0.1))                                       # sampled form factor
            mx = rms * crest + abs(rng.normal(0, 0.05))                                  # derive a plausible max value from RMS and crest factor
            mn = -mx * rng.uniform(0.7, 1.0)                                              # derive a plausible min value, roughly symmetric to max
            rows.append([mx, mn, mean, sd, rms, skew, kurt, crest, form, name])            # store the 9 base features plus the class label

    df = pd.DataFrame(rows, columns=BASE_FEATURES + ["fault_class"])            # assemble all synthetic samples + labels into a DataFrame
    engineered = engineer_features(df[BASE_FEATURES])                            # expand to the full 19-feature engineered representation
    df_out = pd.concat([engineered, df["fault_class"]], axis=1)                    # reattach the label column for exploration/plotting
    return df_out                                                                    # unscaled engineered features + string fault class label


# --------------------------------------------------------------------------
# SCALER HONESTY HELPERS
# The six models were trained on data/*_scaled.csv, which were standardised with
# the ORIGINAL training scaler. That scaler file was never shipped with the
# project, so typed/raw values are scaled with a synthetic stand-in. That is why
# the dashboard now tells the user which situation they are in.
# --------------------------------------------------------------------------
def scaler_is_original() -> bool:
    """True only when the real training-time scaler (data/feature_scaler.pkl) is present."""
    return (DATA_DIR / "feature_scaler.pkl").exists()


def looks_prescaled(df_full: pd.DataFrame) -> bool:
    """
    Heuristic: a full 19-column table whose columns are already roughly
    zero-mean / unit-variance (like data/test_set_scaled.csv) has been
    standardised before, and must NOT be scaled a second time.
    """
    try:
        v = df_full[ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    except KeyError:
        return False
    if len(v) < 20:  # too few rows to judge mean/std reliably
        return False
    means, stds = v.mean().abs(), v.std()
    return bool((means < 0.35).mean() > 0.85 and (stds.between(0.6, 1.6)).mean() > 0.85)


def preprocess_csv_upload_ex(df_raw: pd.DataFrame, assume_prescaled: bool | None = None):
    """
    Same as preprocess_csv_upload, but also reports whether the second scaling
    step was skipped because the file is already standardised.
    Returns (engineered_df, scaled_array, was_prescaled).
    """
    cols = set(df_raw.columns)
    if set(ALL_FEATURES).issubset(cols):
        df_full = df_raw[ALL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
        pre = looks_prescaled(df_full) if assume_prescaled is None else assume_prescaled
        if pre:
            return df_full, df_full.values.astype(np.float32), True
    df_full, scaled = preprocess_csv_upload(df_raw)
    return df_full, scaled, False
