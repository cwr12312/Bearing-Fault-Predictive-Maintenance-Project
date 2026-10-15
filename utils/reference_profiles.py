"""
utils/reference_profiles.py
============================
Per-class reference profiles built from the REAL training data
(data/preprocessed_augmented_dataset_scaled.csv, the data the six models were
trained on). Used to ground Root Cause Reasoning in evidence and to check
whether a sample actually resembles the class the model predicted.

Everything here is computed from that file; nothing is invented. If the file
is missing, `get_profiles()` returns None and callers fall back gracefully.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from config import ALL_FEATURES, CLASS_NAMES, DATA_DIR

_DATA_FILE = DATA_DIR / "preprocessed_augmented_dataset_scaled.csv"
_TOLERANCE = 1.5   # a sample may sit 1.5x beyond the 99th-percentile distance of its class and still count as in-range
_STD_FLOOR = 0.05  # stops near-constant features from dominating distances


@lru_cache(maxsize=1)
def get_profiles():
    """Returns dict with per-class mean/std/distance-percentiles, or None if data is unavailable."""
    if not _DATA_FILE.exists():
        return None
    try:
        df = pd.read_csv(_DATA_FILE)
        X = df[ALL_FEATURES].to_numpy(dtype=np.float64)
        y = df["fault_encoded"].to_numpy(dtype=int)
    except (OSError, KeyError, ValueError):
        return None
    means, stds, p99, p95 = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    for i, cname in enumerate(CLASS_NAMES):
        Xi = X[y == i]
        if len(Xi) < 10:
            continue
        mu, sd = Xi.mean(0), np.maximum(Xi.std(0), _STD_FLOOR)
        sub = Xi[rng.choice(len(Xi), size=min(3000, len(Xi)), replace=False)]
        d = np.sqrt((((sub - mu) / sd) ** 2).mean(1))
        means[cname], stds[cname] = mu, sd
        p95[cname], p99[cname] = float(np.percentile(d, 95)), float(np.percentile(d, 99))
    return {"mean": means, "std": stds, "p95": p95, "p99": p99,
            "lo": X.min(0), "hi": X.max(0),
            "overall_mean": X.mean(0), "overall_std": np.maximum(X.std(0), _STD_FLOOR)}


def _dist(x, mu, sd) -> float:
    return float(np.sqrt((((x - mu) / sd) ** 2).mean()))


def assess(scaled_row, predicted_class: str) -> dict | None:
    """
    Compare one scaled 19-feature sample with the training profile of the
    predicted class and with every other class.
    """
    prof = get_profiles()
    if prof is None or predicted_class not in prof["mean"]:
        return None
    x = np.asarray(scaled_row, dtype=np.float64).ravel()
    if x.shape[0] != len(ALL_FEATURES):
        return None
    dists = {c: _dist(x, prof["mean"][c], prof["std"][c]) for c in prof["mean"]}
    ranked = sorted(dists.items(), key=lambda kv: kv[1])
    d_pred = dists[predicted_class]
    key_feats = {}
    for f in ("rms", "kurtosis", "crest", "skewness", "peak_to_peak"):
        j = ALL_FEATURES.index(f)
        mu, sd = prof["mean"][predicted_class][j], prof["std"][predicted_class][j]
        key_feats[f] = {"value": float(x[j]), "class_mean": float(mu), "class_std": float(sd),
                        "z": float((x[j] - mu) / sd), "vs_overall": float(x[j])}
    return {
        "dist_pred": d_pred, "p95": prof["p95"][predicted_class], "p99": prof["p99"][predicted_class],
        "in_distribution": d_pred <= _TOLERANCE * prof["p99"][predicted_class],
        "nearest": ranked[0][0], "nearest_dist": ranked[0][1],
        "nearest_agrees": ranked[0][0] == predicted_class,
        "top3": ranked[:3], "features": key_feats,
    }


def is_reliable(scaled_row, predicted_class: str) -> bool:
    """True when the sample resembles the training data for its predicted class
    (or when no reference data is available to judge, so nothing is wrongly blocked)."""
    a = assess(scaled_row, predicted_class)
    return True if a is None else bool(a["in_distribution"])


def out_of_range_features(scaled_row, max_items: int = 5) -> list[dict]:
    """Features whose scaled value lies outside everything seen in the training data."""
    prof = get_profiles()
    if prof is None:
        return []
    x = np.asarray(scaled_row, dtype=np.float64).ravel()
    out = []
    for j, f in enumerate(ALL_FEATURES):
        if x[j] < prof["lo"][j] or x[j] > prof["hi"][j]:
            span = max(prof["hi"][j] - prof["lo"][j], 1e-9)
            excess = (prof["lo"][j] - x[j]) if x[j] < prof["lo"][j] else (x[j] - prof["hi"][j])
            out.append({"feature": f, "value": float(x[j]), "lo": float(prof["lo"][j]),
                        "hi": float(prof["hi"][j]), "excess_ratio": float(excess / span)})
    out.sort(key=lambda d: -d["excess_ratio"])
    return out[:max_items]
