"""
utils/shap_utils.py
====================
Model-agnostic explainability layer. Uses SHAP's generic `Explainer` (which
auto-selects a Permutation/Kernel algorithm) wrapped around each model's
`predict_proba`, so all six heterogeneous architectures (Keras and PyTorch)
share the same explanation code path. If SHAP cannot explain a particular
model for any reason, the caller receives a clear status instead of a
crash, and the UI falls back to permutation feature importance.
"""

from __future__ import annotations           # allows modern type-hint syntax (e.g. X | None) on older Python versions

import numpy as np                             # array handling for feature values and importances
import pandas as pd                            # builds the tabular importance output

from config import ALL_FEATURES, CLASS_NAMES  # feature names (for labeling importances) and class names (for SHAP output labels)


def _predict_fn_factory(wrapper):
    def _fn(X):
        return wrapper.predict_proba(np.asarray(X, dtype=np.float32))  # ensure SHAP's perturbed inputs are float32 before predicting
    return _fn                                                          # returns a plain callable SHAP can call repeatedly


def compute_shap_values(wrapper, background: np.ndarray, instances: np.ndarray, max_evals: int = 300):
    """
    Returns a dict:
      {"status": "ok", "values": shap.Explanation} on success, or
      {"status": "fallback", "importances": np.ndarray} using permutation importance, or
      {"status": "error", "message": str}
    """
    try:
        import shap                                                         # imported lazily; only needed when SHAP explanation is actually attempted
        predict_fn = _predict_fn_factory(wrapper)                            # wrap the model's predict_proba for SHAP to call
        masker = shap.maskers.Independent(background, max_samples=min(100, len(background)))  # defines how features are perturbed, based on a background sample
        explainer = shap.Explainer(predict_fn, masker, output_names=CLASS_NAMES)  # generic explainer, auto-picks an algorithm (e.g. Permutation)
        sv = explainer(instances, max_evals=max_evals, silent=True)            # compute SHAP values for the given instances
        return {"status": "ok", "values": sv}                                   # success -> return the SHAP explanation object
    except Exception as exc:  # noqa: BLE001
        fallback = _permutation_importance(wrapper, background, instances)      # SHAP failed -> try the lightweight fallback instead
        if fallback is not None:
            return {"status": "fallback", "importances": fallback, "message": str(exc)}  # fallback succeeded -> report it plus the original error
        return {"status": "error", "message": str(exc)}                           # both SHAP and the fallback failed -> report the error


def _permutation_importance(wrapper, background: np.ndarray, instances: np.ndarray):
    """A lightweight, dependency-free fallback: measures how much shuffling
    each feature (holding it at the background mean) changes the predicted
    probability of the winning class, averaged across the given instances."""
    try:
        base_probs = wrapper.predict_proba(instances)                    # original predicted probabilities for each instance
        base_classes = base_probs.argmax(axis=1)                          # the predicted (winning) class per instance
        n_feat = instances.shape[1]                                        # number of features to test importance for
        importances = np.zeros(n_feat)                                      # accumulator for each feature's importance score
        bg_mean = background.mean(axis=0)                                    # background mean per feature, used to "neutralize" a feature

        for f in range(n_feat):                                              # test each feature one at a time
            perturbed = instances.copy()                                       # copy so the original instances aren't modified
            perturbed[:, f] = bg_mean[f]                                        # replace this one feature with its background mean (removes its signal)
            new_probs = wrapper.predict_proba(perturbed)                         # re-predict with that feature neutralized
            drop = base_probs[np.arange(len(instances)), base_classes] - \
                   new_probs[np.arange(len(instances)), base_classes]              # how much the winning class's probability dropped
            importances[f] = np.mean(np.abs(drop))                                   # average absolute drop across instances = importance of this feature
        return importances                                                             # array of per-feature importance scores
    except Exception:
        return None                                                                      # fallback itself failed -> signal "no importance available"


def global_permutation_importance(wrapper, background: np.ndarray, sample: np.ndarray) -> pd.DataFrame:
    imp = _permutation_importance(wrapper, background, sample)             # compute per-feature importance over a representative sample
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])              # importance computation failed -> return an empty table
    df = pd.DataFrame({"feature": ALL_FEATURES, "importance": imp})          # pair each feature name with its importance score
    return df.sort_values("importance", ascending=False).reset_index(drop=True)  # sort most-important features first
