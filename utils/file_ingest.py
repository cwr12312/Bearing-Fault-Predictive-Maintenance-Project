"""
utils/file_ingest.py
=====================
Multi-format dataset ingestion for the Live Prediction page's batch-upload
flow. Accepts an uploaded file (any Streamlit `UploadedFile`) in one of the
following formats and returns a plain pandas DataFrame, exactly as
`pd.read_csv()` used to:

    .csv                 - comma-separated values
    .json                - JSON records / columns / list-of-dicts
    .h5 / .hdf5           - HDF5 (via pandas/PyTables, falling back to h5py)
    .pkl / .pickle        - a pickled pandas DataFrame / ndarray / dict
    .mat                  - MATLAB (via scipy.io.loadmat)
    .svm / .libsvm        - LibSVM / SVMLight sparse format
    .arff                 - Weka ARFF

Every loader normalises its result down to a DataFrame with plain numeric
columns wherever possible. Column *names* are preserved when the format
carries them (CSV, JSON, HDF5, ARFF, most .mat/.pkl payloads). Formats that
carry no column names at all (LibSVM/SVMLight) are mapped positionally onto
BASE_FEATURES (9 columns) or ALL_FEATURES (19 columns) — the same two
layouts `preprocess_csv_upload()` already understands — so the existing
downstream pipeline in `utils/preprocessing.py` needs no changes.

SECURITY NOTE: the .pkl/.pickle path uses Python's `pickle` module, which
can execute arbitrary code when deserialising an untrusted file. Only
upload pickle files you created yourself or otherwise trust.
"""

from __future__ import annotations               # allows modern type-hint syntax (e.g. X | None) on older Python versions

import io                                          # wraps raw bytes as file-like objects for the various parsers
import json                                        # manual JSON parsing fallback for shapes pandas can't read directly
import pickle                                       # deserializes .pkl/.pickle uploads (see security note above)
from pathlib import Path                            # used to pull the file extension out of the uploaded filename
from typing import Any                              # generic type hint for the object loaded out of a pickle file

import numpy as np                                  # array handling for positional (unnamed-column) formats
import pandas as pd                                 # the common DataFrame output type every loader returns

from config import ALL_FEATURES, BASE_FEATURES      # the two known column-name layouts used for positional mapping

# --------------------------------------------------------------------------
# File types accepted by the Streamlit uploader + human-readable summary
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = [
    "csv", "json", "h5", "hdf5", "pkl", "pickle", "mat", "svm", "libsvm", "arff",
]                                                     # extensions the Streamlit file_uploader widget will accept
SUPPORTED_FORMATS_LABEL = (
    "CSV, JSON, HDF5 (.h5/.hdf5), Pickle (.pkl/.pickle), MATLAB (.mat), "
    "LibSVM (.svm/.libsvm), ARFF"
)                                                     # human-readable string shown in UI help text / error messages


def _extension_of(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "") or ""    # safely get the filename, default to "" if missing
    return Path(name).suffix.lower().lstrip(".")        # extract extension, lowercase it, and strip the leading dot


def _positional_feature_frame(array: np.ndarray) -> pd.DataFrame:
    """
    Maps a plain numeric 2D array with no column names onto BASE_FEATURES
    (if it has 9 columns) or ALL_FEATURES (if it has 19), which is the same
    contract `preprocess_csv_upload()` expects. Anything else is returned
    with generic column names and left for `preprocess_csv_upload()` to
    reject with its normal, informative error message.
    """
    array = np.atleast_2d(array)                         # ensure at least a 2D array (handles a single flat row)
    n_cols = array.shape[1]                               # number of columns/features in the array
    if n_cols == len(BASE_FEATURES):
        columns = BASE_FEATURES                           # 9 columns -> assume the 9 base statistical features
    elif n_cols == len(ALL_FEATURES):
        columns = ALL_FEATURES                            # 19 columns -> assume the full engineered feature set
    else:
        columns = [f"col_{i}" for i in range(n_cols)]      # unknown width -> generic placeholder names, let downstream code reject it
    return pd.DataFrame(array, columns=columns)             # wrap the raw array as a labeled DataFrame


# --------------------------------------------------------------------------
# Individual format loaders
# --------------------------------------------------------------------------

def _load_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))                    # standard CSV parse from an in-memory byte buffer


def _load_json(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_json(io.BytesIO(raw))                # try pandas' built-in JSON reader first
    except ValueError:
        pass
    # Fall back to a manual parse for shapes pandas' reader is picky about,
    # e.g. {"columns": [...], "data": [[...], ...]} or a single dict of
    # scalars (one row).
    obj = json.loads(raw.decode("utf-8"))                    # decode bytes to text and parse as generic JSON
    if isinstance(obj, dict) and "data" in obj and "columns" in obj:
        return pd.DataFrame(obj["data"], columns=obj["columns"])  # {"columns": [...], "data": [[...]]} layout
    if isinstance(obj, dict):
        return pd.DataFrame([obj])                            # single flat dict -> treat as one row
    if isinstance(obj, list):
        return pd.DataFrame(obj)                              # list of records -> standard tabular conversion
    raise ValueError("Unrecognised JSON structure for a feature dataset.")  # nothing matched -> give up with a clear error


def _load_hdf5(raw: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw)                                     # in-memory buffer used by the h5py fallback path
    # Preferred path: pandas + PyTables, for HDF5 files written by
    # `DataFrame.to_hdf()`.
    try:
        with pd.HDFStore(
            "_uploaded.h5", mode="r", driver="H5FD_CORE",
            driver_core_backing_store=0, driver_core_image=raw,
        ) as store:                                             # open the HDF5 bytes purely in memory (no disk write)
            keys = store.keys()                                  # list of dataset keys stored inside the file
            if not keys:
                raise ValueError("HDF5 file contains no datasets.")
            return store[keys[0]]                                 # return the first dataset found as a DataFrame
    except Exception:
        pass
    # Fallback: raw h5py, for HDF5 files written by other tools (e.g.
    # MATLAB v7.3, numpy, custom export scripts) that store plain arrays.
    import h5py  # imported lazily; only required if this path is used

    with h5py.File(buf, "r") as f:                               # open the HDF5 buffer with the lower-level h5py API
        dataset_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]  # only keep actual dataset entries (not groups)
        if not dataset_keys:
            raise ValueError("No readable datasets found inside the HDF5 file.")
        # Prefer a dataset literally called "data"/"features"/"df" if present.
        preferred = next((k for k in dataset_keys if k.lower() in ("data", "features", "df", "x")), dataset_keys[0])  # pick the best-guess dataset name, else the first one
        array = f[preferred][()]                                    # read the dataset's contents into memory as a numpy array
        return _positional_feature_frame(np.asarray(array))          # map the unnamed array onto known feature column layouts


def _load_pickle(raw: bytes) -> pd.DataFrame:
    obj: Any = pickle.loads(raw)                                 # deserialize the pickled Python object (see security note at top of file)
    if isinstance(obj, pd.DataFrame):
        return obj                                                # already a DataFrame -> use as-is
    if isinstance(obj, dict):
        # A dict of column_name -> values, or a {"X":..., "columns": [...]}-style export.
        if "data" in obj and "columns" in obj:
            return pd.DataFrame(obj["data"], columns=obj["columns"])  # explicit data/columns layout
        try:
            return pd.DataFrame(obj)                                # try treating dict keys as column names
        except Exception:
            pass
    if isinstance(obj, (np.ndarray, list)):
        return _positional_feature_frame(np.asarray(obj, dtype=float))  # unnamed array/list -> positional column mapping
    raise ValueError(f"Unsupported object of type {type(obj).__name__} found inside the pickle file.")  # nothing recognized -> clear error


def _load_mat(raw: bytes) -> pd.DataFrame:
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(raw))                                 # parse the MATLAB .mat file into a dict of variables
    # Drop MATLAB's private/meta keys (__header__, __version__, __globals__).
    candidates = {k: v for k, v in mat.items() if not k.startswith("__")}  # keep only user-defined variables
    if not candidates:
        raise ValueError("No variables found inside the .mat file.")
    # Prefer a variable literally called data/features/X, else the largest 2D array.
    preferred_names = ("data", "features", "x", "X")
    key = next((k for k in candidates if k.lower() in [p.lower() for p in preferred_names]), None)  # look for a conventionally named variable
    if key is None:
        key = max(candidates, key=lambda k: np.asarray(candidates[k]).size)  # otherwise fall back to the largest variable by element count
    array = np.asarray(candidates[key])                             # pull out the chosen variable's array
    return _positional_feature_frame(array)                          # map it onto known feature column layouts


def _load_libsvm(raw: bytes) -> pd.DataFrame:
    from sklearn.datasets import load_svmlight_file

    with io.BytesIO(raw) as buf:
        X, _y = load_svmlight_file(buf)                              # parse LibSVM/SVMLight sparse format; labels (_y) are discarded
    return _positional_feature_frame(np.asarray(X.todense()))         # densify the sparse matrix and map onto known feature layouts


def _load_arff(raw: bytes) -> pd.DataFrame:
    from scipy.io import arff

    text = io.StringIO(raw.decode("utf-8", errors="replace"))          # decode bytes to text, replacing any bad characters
    data, meta = arff.loadarff(text)                                    # parse the Weka ARFF format (meta is unused here)
    df = pd.DataFrame(data)                                              # convert the structured numpy array into a DataFrame
    # scipy decodes ARFF string/nominal columns as bytes objects; decode them.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)  # turn byte-string cells into normal strings
    return df


_LOADERS = {
    "csv": _load_csv,                 # maps each supported extension to its loader function
    "json": _load_json,
    "h5": _load_hdf5,
    "hdf5": _load_hdf5,
    "pkl": _load_pickle,
    "pickle": _load_pickle,
    "mat": _load_mat,
    "svm": _load_libsvm,
    "libsvm": _load_libsvm,
    "arff": _load_arff,
}


def load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Reads a Streamlit `UploadedFile` of any SUPPORTED_EXTENSIONS type and
    returns a plain pandas DataFrame, ready to be handed to
    `utils.preprocessing.preprocess_csv_upload()` exactly as a CSV always
    was. Raises ValueError with a clear message on any unsupported or
    malformed file.
    """
    ext = _extension_of(uploaded_file)                          # determine the file extension from its name
    loader = _LOADERS.get(ext)                                    # look up the matching loader function
    if loader is None:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported formats: {SUPPORTED_FORMATS_LABEL}."
        )                                                          # no loader registered for this extension -> reject clearly
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()  # read the raw bytes regardless of upload object type
    try:
        df = loader(raw)                                           # run the format-specific loader
    except ValueError:
        raise                                                       # re-raise loader's own clear ValueError as-is
    except Exception as e:
        raise ValueError(f"Could not parse this .{ext} file as a feature dataset: {e}") from e  # wrap any other error in a friendlier message
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"The .{ext} file did not contain any readable rows.")  # guard against empty/invalid results
    return df                                                         # final, ready-to-use DataFrame
