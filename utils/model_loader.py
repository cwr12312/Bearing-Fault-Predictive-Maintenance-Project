"""
utils/model_loader.py
======================
Loads the six trained models (three .keras, three .pth) exactly as saved.
Handles deserialization of models saved with newer TensorFlow versions.

Why this file is more defensive than a typical model-loading module:
the three Keras models were trained/saved with a TensorFlow version that
may not exactly match the one installed at deploy time, so a plain
`tf.keras.models.load_model(...)` call can fail on config fields the
installed TF doesn't recognise (e.g. `DTypePolicy`, `quantization_config`).
`load_keras_model_compatible()` below works around this by trying several
progressively more permissive loading strategies before giving up, and
`recreate_model_from_name()` is the last resort: rebuild the architecture
from scratch and load just the weights.

The .pth (PyTorch) models don't have this version-skew problem since the
checkpoint stores the architecture hyperparameters (`arch`/`model_config`)
alongside the weights, so `load_torch_checkpoint()` can just rebuild the
exact architecture and call `load_state_dict()`.
"""

from __future__ import annotations

import time
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

from config import MODEL_REGISTRY, N_CLASSES, N_FEATURES
from utils.torch_architectures import (
    FaultTransformer,
    OptimizedmamlTransformer,
    OptimizedFBCLTransformer,
    count_boosting_heads,
)

# Import custom layers for deserialization
from utils.custom_layers import CUSTOM_LAYERS


# ============================================================
# TENSORFLOW MODEL LOADER WITH CUSTOM DESERIALIZATION
# ============================================================

def safe_deserialize_dtype_policy(config):
    """
    Handle DTypePolicy deserialization safely.
    Newer Keras/TF versions serialize the mixed-precision dtype policy as a
    {"class_name": "DTypePolicy", "config": {...}} dict; older installed
    versions may not know how to rebuild that class automatically, so this
    manually reconstructs a `Policy` object from the serialized name
    (defaulting to "float32" if none was recorded).
    """
    if isinstance(config, dict) and config.get('class_name') == 'DTypePolicy':
        return tf.keras.mixed_precision.Policy(config.get('config', {}).get('name', 'float32'))
    return config


def fix_config_for_loading(config):
    """
    Recursively fix config for loading by removing unsupported parameters.
    Walks the (potentially deeply nested) Keras model config dict/list
    structure and strips keys that newer save formats include but an
    older/mismatched TensorFlow install doesn't understand, so those
    fields don't cause the whole model load to blow up.
    """
    if isinstance(config, dict):
        # Remove problematic keys
        config.pop('quantization_config', None)
        config.pop('shared_object_id', None)
        
        # Fix DTypePolicy
        for key, value in list(config.items()):
            if isinstance(value, dict):
                fix_config_for_loading(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fix_config_for_loading(item)
    
    return config


def load_keras_model_compatible(path_str: str):
    """
    Load Keras model with compatibility handling for different TensorFlow versions.
    Uses multiple strategies to load the model.

    Tries five progressively more permissive strategies in order and
    returns as soon as one succeeds; each `except` below only prints a
    warning and falls through to the next strategy rather than raising
    immediately, since an early strategy failing is expected/normal here.
    Only if every strategy fails does this function raise, in Strategy 5.
    """
    path = Path(path_str)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")
    
    # ============================================================
    # STRATEGY 1: Load with custom objects and custom layers
    # The "happy path" — load the file as-is, registering this project's
    # own custom Keras layers (utils/custom_layers.py) so the model's
    # architecture (which references them by name) can be rebuilt.
    # ============================================================
    try:
        # Register all custom objects
        with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
            model = tf.keras.models.load_model(path_str, compile=False)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with custom objects")
        return model
    except Exception as e:
        print(f"⚠️ Custom object loading failed: {e}")
    
    # ============================================================
    # STRATEGY 2: Load without compilation (just architecture)
    # Same as Strategy 1 but additionally hands Keras a small, explicit
    # map of the *standard* Keras classes involved (Dense/Conv2D/DTypePolicy)
    # in case the custom_object_scope alone wasn't enough to resolve them.
    # ============================================================
    try:
        with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
            model = tf.keras.models.load_model(
                path_str, 
                compile=False,
                custom_objects={
                    'DTypePolicy': tf.keras.mixed_precision.Policy,
                    'Dense': tf.keras.layers.Dense,
                    'Conv2D': tf.keras.layers.Conv2D
                }
            )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with basic custom objects")
        return model
    except Exception as e:
        print(f"⚠️ Basic custom objects failed: {e}")
    
    # ============================================================
    # STRATEGY 3: Load using H5 format (legacy)
    # Drops the custom_object_scope entirely and instead relies only on
    # explicit custom_objects for common initializers/regularizers that
    # tend to trip up loading of older/legacy-format (H5-style) saves.
    # ============================================================
    try:
        model = tf.keras.models.load_model(
            path_str, 
            compile=False,
            custom_objects={
                'DTypePolicy': tf.keras.mixed_precision.Policy,
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'Zeros': tf.keras.initializers.Zeros,
                'L2': tf.keras.regularizers.L2,
            }
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with legacy format")
        return model
    except Exception as e:
        print(f"⚠️ Legacy format failed: {e}")
    
    # ============================================================
    # STRATEGY 4: Recreate model from scratch and load weights
    # Last resort before giving up entirely: rebuild the bare architecture
    # in code (see recreate_model_from_name below) using the same layer
    # shapes/hyperparameters the model was trained with, then load ONLY
    # the numeric weights from the saved file onto that fresh architecture.
    # This sidesteps config-deserialization issues completely, since no
    # config is read — only the weight tensors are.
    # ============================================================
    try:
        model = recreate_model_from_name(path.stem)
        if model is not None:
            # Try to load weights
            model.load_weights(path_str)
            print(f"✅ Recreated {path.name} from architecture and loaded weights")
            return model
    except Exception as e:
        print(f"⚠️ Recreate and load weights failed: {e}")
    
    # ============================================================
    # STRATEGY 5: Final attempt - use keras.models.load_model with ignore
    # Broadest custom_objects map yet, covering essentially every layer
    # type these six models use. If this also fails, there is no further
    # fallback — raise so the caller (load_all_models) can record the
    # real error and let the rest of the app keep running without this
    # one model.
    # ============================================================
    try:
        # Try loading with safe globals
        model = tf.keras.models.load_model(
            path_str,
            compile=False,
            custom_objects={
                'DTypePolicy': tf.keras.mixed_precision.Policy,
                'Dense': tf.keras.layers.Dense,
                'Conv2D': tf.keras.layers.Conv2D,
                'BatchNormalization': tf.keras.layers.BatchNormalization,
                'Dropout': tf.keras.layers.Dropout,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                'InputLayer': tf.keras.layers.InputLayer,
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'Zeros': tf.keras.initializers.Zeros,
                'Ones': tf.keras.initializers.Ones,
                'L2': tf.keras.regularizers.L2,
            }
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with comprehensive custom objects")
        return model
    except Exception as e:
        raise RuntimeError(f"All loading strategies failed for {path.name}: {str(e)}")


def recreate_model_from_name(model_name):
    """
    Recreate model architecture from scratch based on model name.
    This is a fallback when loading the saved model fails.
    Maps a checkpoint's filename stem (e.g. "cnn_2d_model_fixed") to the
    matching architecture-builder function in models/tensorflow_models.py,
    so Strategy 4 above can build a fresh, uncompiled model and then load
    just the raw weights onto it.
    """
    from models.tensorflow_models import (
        create_cnn_2d_model,
        create_lstm_model,
        create_transformer_model
    )
    
    creators = {
        'cnn_2d_model': create_cnn_2d_model,
        'cnn_2d_model_fixed': create_cnn_2d_model,
        'lstm_model': create_lstm_model,
        'lstm_model_fixed': create_lstm_model,
        'final_transformer_model': create_transformer_model,
        'transformer_model_fixed': create_transformer_model,
    }
    
    # Try exact match
    if model_name in creators:
        model = creators[model_name]()
        return model
    
    # Try partial match — handles filename variants that weren't in the
    # exact-match dict above (e.g. a versioned or renamed checkpoint file).
    for key, creator in creators.items():
        if key in model_name or model_name in key:
            model = creator()
            return model
    
    # No architecture recognised for this filename — caller treats this as
    # "recreation isn't possible", not an error, and moves on to Strategy 5.
    return None


class ModelWrapper:
    """
    Common interface around a loaded Keras or PyTorch model, so the rest
    of the app (e.g. pages/2_Live_Prediction.py) can call the same
    `.predict()` / `.predict_proba()` methods regardless of which of the
    six underlying frameworks/architectures actually produced the result.
    """

    def __init__(self, name, framework, model, extra=None):
        self.name = name
        self.framework = framework  # "keras" or "torch" — drives branching in predict_proba().
        self.model = model
        self.extra = extra or {}  # e.g. the raw PyTorch checkpoint dict, kept for reference.

    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape input based on model requirements.
        The 19 scaled features arrive as a flat (n_samples, 19) matrix,
        but each Keras architecture expects a different tensor shape
        (config.MODEL_REGISTRY[name]["input_shape"]): the 2D CNN wants a
        padded 2D "image", while LSTM/Transformer want a (timesteps,
        features) sequence. This reshapes/pads only for the models that
        need it and passes everything else through unchanged.
        """
        cfg = MODEL_REGISTRY[self.name]
        shape = cfg["input_shape"]
        n = X.shape[0]
        
        if self.name == "Two-Dimensional Convolutional Neural Network (2D CNN)":
            # Zero-pad the flat feature vector out to a perfect square-ish
            # grid (shape[0] x shape[1]) before adding the channel dimension.
            pad = int(np.prod(shape[:2])) - X.shape[1]
            Xp = np.pad(X, ((0, 0), (0, max(pad, 0))), mode="constant")
            return Xp.reshape((n, shape[0], shape[1], shape[2]))
        
        elif self.name == "Long Short-Term Memory (LSTM)":
            # Treat the 19 features as a short sequence rather than a flat vector.
            return X.reshape((n, shape[0], shape[1]))
        
        elif self.name == "Transformer":
            return X.reshape((n, shape[0], shape[1]))
        
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from the model.

        Keras:
            CNN -> (n, 5, 4, 1)
            LSTM -> (n, 1, 19)
            Transformer -> (n, 19, 1)

        PyTorch:
            All three models -> (n, 19)
        """
        X = np.asarray(X, dtype=np.float32)

        # ================================================================
        # KERAS MODELS
        # ================================================================
        if self.framework == "keras":

            # ------------------------------------------------------------
            # TRANSFORMER
            # ------------------------------------------------------------
            if self.name == "Transformer":
                try:
                    # The Transformer expects exactly:
                    # (batch_size, 19, 1)
                    Xr = X.reshape((-1, 19, 1))

                    # Convert to an explicit TensorFlow tensor.
                    Xr = tf.convert_to_tensor(Xr, dtype=tf.float32)

                    # Tell TensorFlow the exact expected shape.
                    Xr = tf.ensure_shape(Xr, [None, 19, 1])

                    # Run inference.
                    probs = self.model(Xr, training=False)

                    # Convert TensorFlow tensor to NumPy.
                    probs = np.asarray(probs.numpy(), dtype=np.float32)

                    # Make sure the model returned 10 class probabilities.
                    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
                        raise ValueError(
                            f"Transformer returned unexpected shape: {probs.shape}"
                        )

                    return probs

                except Exception as e:
                    print(
                        f"❌ Transformer inference failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise

            # ------------------------------------------------------------
            # CNN / LSTM
            # ------------------------------------------------------------
            try:
                Xr = self._reshape_for_model(X)

                Xr = tf.convert_to_tensor(Xr, dtype=tf.float32)

                probs = self.model(Xr, training=False)

                return np.asarray(probs.numpy(), dtype=np.float32)

            except Exception as e:
                print(
                    f"❌ Keras prediction error for {self.name}: "
                    f"{type(e).__name__}: {e}"
                )
                raise

        # ================================================================
        # PYTORCH MODELS
        # ================================================================
        import torch

        Xt = torch.tensor(X, dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():

            if self.name == "Feature-Based Contrastive Learning (FBCL)":
                logits = self.model(Xt, use_boosting=True)

            elif self.name == "Model-Agnostic Meta-Learning (MAML)":
                logits = self.model(Xt, mask=None)

            else:
                logits = self.model(Xt)

            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X: np.ndarray):
        """Convenience wrapper: returns (predicted_class_indices, full_probability_matrix)."""
        probs = self.predict_proba(X)
        return probs.argmax(axis=1), probs

    def timed_predict(self, X: np.ndarray):
        """Same as predict(), but also returns the per-sample inference time in ms."""
        t0 = time.perf_counter()
        preds, probs = self.predict(X)
        elapsed = (time.perf_counter() - t0) * 1000.0
        per_sample = elapsed / max(1, X.shape[0])
        return preds, probs, per_sample


# ============================================================
# MAIN MODEL LOADER WITH CACHING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_keras_model(path_str: str):
    """
    Load Keras model with compatibility handling.
    Wrapped in st.cache_resource so each Keras model file is only ever
    loaded from disk once per Streamlit server process, no matter how
    many times/pages call this — subsequent calls return the same
    in-memory model object instantly.
    """
    return load_keras_model_compatible(path_str)


@st.cache_resource(show_spinner=False)
def load_torch_checkpoint(model_name: str, path_str: str):
    """
    Load PyTorch model checkpoint.
    Also cached like load_keras_model above. Rebuilds the exact
    architecture the checkpoint was trained with (using the saved
    `arch`/`model_config` hyperparameters) and loads the trained weights
    onto it via `load_state_dict`.
    """
    import torch
    
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")
    
    try:
        ckpt = torch.load(path_str, map_location="cpu", weights_only=False)
    except Exception as e:
        # Some PyTorch versions/checkpoints only work with the stricter
        # weights_only=True loading path — try that before giving up.
        try:
            ckpt = torch.load(path_str, map_location="cpu", weights_only=True)
        except:
            raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")
    
    n_features = ckpt.get("n_features", N_FEATURES)
    n_classes = ckpt.get("n_classes", N_CLASSES)
    state_dict = ckpt["model_state_dict"]

    # Each of the three PyTorch architectures needs its own hyperparameters
    # (d_model, n_heads, n_layers, dropout) read back out of the checkpoint
    # so the freshly-built model exactly matches the shape of the saved weights.
    if model_name == "Meta Stochastic Gradient Descent (Meta-SGD)":
        arch = ckpt.get("arch", {})
        model = FaultTransformer(
            n_feat=n_features, n_classes=n_classes,
            d_model=arch.get("d_model", 64), n_heads=arch.get("n_heads", 4),
            n_layers=arch.get("n_layers", 3), dropout=arch.get("dropout", 0.15),
        )
        model.load_state_dict(state_dict)

    elif model_name == "Model-Agnostic Meta-Learning (MAML)":
        cfg = ckpt.get("model_config", {})
        model = OptimizedmamlTransformer(
            n_features=n_features, n_classes=n_classes,
            d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4), dropout=cfg.get("dropout", 0.3),
        )
        model.load_state_dict(state_dict)
        # MAML trains a meta-initialization, then fine-tunes on the target
        # task — switch_to_finetune() puts it into the fine-tuned mode
        # this dashboard actually does inference with.
        model.switch_to_finetune()

    elif model_name == "Feature-Based Contrastive Learning (FBCL)":
        cfg = ckpt.get("model_config", {})
        model = OptimizedFBCLTransformer(
            n_features=n_features, n_classes=n_classes,
            d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 16),
            n_layers=cfg.get("n_layers", 6), dropout=cfg.get("dropout", 0.15),
        )
        # FBCL was trained with an incrementally-grown ensemble of
        # "boosting heads" — recreate exactly as many heads as the saved
        # state_dict has before loading weights, or the shapes won't match.
        n_heads_saved = count_boosting_heads(state_dict)
        for _ in range(n_heads_saved):
            model.add_boosting_head()
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    model.eval()  # Inference mode from here on (no dropout/training behaviour).
    print(f"✅ Loaded {model_name} PyTorch model")
    return model, ckpt


@st.cache_resource(show_spinner=False)
def load_all_models() -> dict:
    """
    Loads every model in MODEL_REGISTRY once per session and caches them.
    Deliberately never raises on an individual model's failure — each
    model is loaded inside its own try/except so one bad file can't take
    down the other five; failures are collected into `errors` and surfaced
    later by get_load_errors() (e.g. shown in the Live Prediction sidebar).
    """
    wrappers = {}
    errors = {}
    
    print("\n" + "="*60)
    print("🚀 LOADING MODELS")
    print("="*60)
    
    for name, cfg in MODEL_REGISTRY.items():
        try:
            print(f"\n📦 Loading {name}...")
            
            if cfg["framework"] == "keras":
                # Use the cached version of the loader
                model = load_keras_model(str(cfg["file"]))
                wrappers[name] = ModelWrapper(name, "keras", model)
                print(f"✅ {name} loaded successfully")
                
            else:
                model, ckpt = load_torch_checkpoint(name, str(cfg["file"]))
                wrappers[name] = ModelWrapper(name, "torch", model, extra={"checkpoint": ckpt})
                print(f"✅ {name} loaded successfully")
                
        except Exception as exc:
            errors[name] = str(exc)
            print(f"❌ {name} failed to load: {exc}")
    
    print("\n" + "="*60)
    print(f"✅ Loaded {len(wrappers)} of {len(MODEL_REGISTRY)} models")
    if errors:
        print(f"⚠️ Failed: {list(errors.keys())}")
    print("="*60 + "\n")
    
    return {"models": wrappers, "errors": errors}


def get_model(name: str) -> ModelWrapper | None:
    """Returns the cached, loaded wrapper for `name`, or None if it failed to load."""
    registry = load_all_models()
    return registry["models"].get(name)


def get_load_errors() -> dict:
    """Returns {model_name: error_message} for every model that failed to load."""
    registry = load_all_models()
    return registry["errors"]


def get_available_models() -> list:
    """Returns the names of every model that loaded successfully."""
    registry = load_all_models()
    return list(registry["models"].keys())


def is_model_available(name: str) -> bool:
    """True if `name` loaded successfully and is ready for inference."""
    registry = load_all_models()
    return name in registry["models"]