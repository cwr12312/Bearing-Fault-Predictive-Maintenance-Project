"""
config.py
=========
Central configuration for the Industrial Predictive Maintenance Dashboard.

Every path, constant, class label, feature name and colour used across the
app is defined here so that the rest of the codebase never hard-codes
"magic values". This mirrors the exact preprocessing and dataset
specification described in the project's Proposal and Interim Report.
"""

from pathlib import Path  # Path gives us cross-platform (Windows/Linux/Mac) filesystem paths.

# --------------------------------------------------------------------------
# PROJECT ROOT / DIRECTORY LAYOUT
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent  # Absolute path to the folder this config.py lives in — the project root.
MODELS_DIR = ROOT_DIR / "models"            # Where the six trained model files (.keras/.pth) are stored.
DATA_DIR = ROOT_DIR / "data"                # Where the CSV datasets (metrics, train/val/test splits) live.
CSS_DIR = ROOT_DIR / "css"                  # Custom stylesheet(s) injected by utils/styling.inject_css().
ASSETS_DIR = ROOT_DIR / "assets"            # Static assets (images/icons) used by the UI, if any.
CHARTS_DIR = ROOT_DIR / "charts"            # Destination folder for any charts exported/saved to disk.
REPORTS_DIR = ROOT_DIR / "reports"          # Destination folder for generated maintenance/prediction reports.
OUTPUTS_DIR = ROOT_DIR / "outputs"          # General-purpose output folder (e.g. downloaded predictions).
LOGS_DIR = ROOT_DIR / "logs"                # Destination folder for any runtime log files.

for _d in (CHARTS_DIR, REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    # Make sure every writable output folder actually exists on disk before
    # anything tries to save into it — exist_ok=True means this is a no-op
    # if the folder is already there (e.g. shipped with a .gitkeep file).
    _d.mkdir(parents=True, exist_ok=True)

METRICS_CSV = DATA_DIR / "model_metrics.csv"  # Pre-computed accuracy/precision/recall/etc. per model, used across pages.

# --------------------------------------------------------------------------
# PROJECT METADATA  (About page)
# --------------------------------------------------------------------------
APP_NAME = "Industrial Predictive Maintenance"  # Short branding name shown in cards/headers.
PROJECT_TITLE = "Industrial Predictive Maintenance System for Bearing Fault Diagnosis Using Advanced Deep Learning Models"  # Full formal project title, e.g. for the About page.
PROJECT_SHORT_TITLE = "Industrial Predictive Maintenance"  # Used in browser tab titles (st.set_page_config's page_title).
DESIGNER_CREDIT = "CHEE WEI RONG"  # Author/designer credit shown on the About page.
PROJECT_DESCRIPTION = (
    # Multi-sentence blurb describing what the dashboard does, shown on the About page.
    "A dual-path, interpretable machine learning system for diagnosing rolling-element "
    "bearing faults from vibration-derived statistical features. This dashboard packages "
    "six advanced deep-learning and meta-learning models — Two-Dimensional Convolutional Neural Network (2D CNN), "
    "Long Short-Term Memory (LSTM), Transformer, Model-Agnostic Meta-Learning (MAML), Meta Stochastic Gradient "
    "Descent (Meta-SGD) and Feature-Based Contrastive Learning (FBCL) — trained on the Case Western Reserve University (CWRU) "
    "Bearing Data Center benchmark, into a single production-style monitoring, prediction "
    "and explainability console."
)
OBJECTIVES = [
    # Bullet-point project objectives, rendered as a list on the About page.
    "Classify ten industrial bearing health states (Normal, Ball, Inner-Race and "
    "Outer-Race faults at three severities) with \u226595% accuracy.",
    "Deliver sub-150ms inference suitable for near-real-time monitoring on commodity hardware.",
    "Compare feature-driven deep learning against meta-learning / continual-learning approaches.",
    "Provide physically interpretable, SHAP-based explanations tied to vibration-analysis theory.",
    "Package the full pipeline into an industrial-grade monitoring dashboard.",
]
METHODOLOGY = (
    # Paragraph summarising the data pipeline (features -> augmentation -> training), for the About page.
    "Raw CWRU vibration recordings were reduced to 9 time-domain statistical features "
    "(max, min, mean, sd, rms, skewness, kurtosis, crest, form). Ten additional engineered "
    "features (ratios and polynomial interactions of the base statistics) were derived, "
    "producing a 19-dimensional feature vector. The training split was augmented via SMOTE "
    "and bootstrapped Gaussian-noise sampling to ~100,000 balanced samples before training "
    "the six deep-learning / meta-learning architectures used in this dashboard."
)
SYSTEM_ARCHITECTURE = (
    # One-line description of the overall data flow, shown on the About page.
    "Streamlit front-end -> cached model registry (TensorFlow/Keras + PyTorch) -> shared "
    "preprocessing pipeline (StandardScaler + feature engineering) -> per-model inference "
    "adapters -> SHAP / permutation explainability layer -> Plotly visualisation layer."
)
DATASET_INFO = {
    # Key facts about the CWRU dataset used to train/evaluate the models, shown on the About page.
    "Source": "Case Western Reserve University (CWRU) Bearing Data Center",
    "Original samples": "2,300 (9 features + 1 target, perfectly balanced, 230 per class)",
    "Augmented training samples": "~100,000 (balanced via SMOTE + Gaussian bootstrapping)",
    "Validation / Test samples": "327 / 329 (held out, untouched by augmentation)",
    "Engineered feature count": "19 (9 base statistical + 10 derived)",
    "Number of classes": "10",
    "Operating condition": "1 HP load, 1772 RPM, drive-end accelerometer",
}
DEVELOPMENT_TOOLS = [
    # Tech stack list, shown on the About page.
    "Python 3.10+", "TensorFlow / Keras", "PyTorch", "Streamlit", "Plotly & Plotly Express",
    "Pandas / NumPy", "scikit-learn", "SHAP", "Custom CSS (Industrial Dark Theme)",
]

# --------------------------------------------------------------------------
# CLASS LABELS  (Table 5 & 6, Interim Report — fault_encoded mapping)
# --------------------------------------------------------------------------
CLASS_NAMES = [
    # The 10 raw class labels the models were trained to predict, in the
    # exact order that matches each model's output index (index 0 = "Ball_007", etc.).
    "Ball_007", "Ball_014", "Ball_021",   # Ball fault at 3 defect sizes (inches).
    "IR_007", "IR_014", "IR_021",         # Inner-race fault at 3 defect sizes.
    "Normal",                              # Healthy bearing, no fault.
    "OR_007_6", "OR_014_6", "OR_021_6",   # Outer-race fault at 3 defect sizes (6 o'clock position).
]
N_CLASSES = len(CLASS_NAMES)  # 10 — used anywhere the model output size / class count is needed.

CLASS_DISPLAY_NAMES = {
    # Human-readable label for each raw class name, shown in the UI instead
    # of the internal codes above (e.g. "IR_014" -> "Inner Race Fault (0.014in)").
    "Ball_007": "Ball Fault (0.007in)",
    "Ball_014": "Ball Fault (0.014in)",
    "Ball_021": "Ball Fault (0.021in)",
    "IR_007": "Inner Race Fault (0.007in)",
    "IR_014": "Inner Race Fault (0.014in)",
    "IR_021": "Inner Race Fault (0.021in)",
    "Normal": "Normal / Healthy",
    "OR_007_6": "Outer Race Fault (0.007in, 6 o'clock)",
    "OR_014_6": "Outer Race Fault (0.014in, 6 o'clock)",
    "OR_021_6": "Outer Race Fault (0.021in, 6 o'clock)",
}

FAULT_FAMILY = {  # maps a specific class to its broad family, used for maintenance recs
    # Groups the 10 fine-grained classes into their 4 broad "families" —
    # utils/maintenance.py keys its recommendation rules off these families
    # rather than the specific defect size, since the recommended action is
    # usually the same across defect sizes within one family.
    "Ball_007": "Ball Fault", "Ball_014": "Ball Fault", "Ball_021": "Ball Fault",
    "IR_007": "Inner Race Fault", "IR_014": "Inner Race Fault", "IR_021": "Inner Race Fault",
    "Normal": "Normal",
    "OR_007_6": "Outer Race Fault", "OR_014_6": "Outer Race Fault", "OR_021_6": "Outer Race Fault",
}

SEVERITY_MAP = {  # inches of induced defect -> qualitative severity
    # Translates the numeric defect-size code embedded in a class name
    # (e.g. the "014" in "IR_014") into a plain-English severity word.
    "007": "Mild", "014": "Moderate", "021": "Severe",
}

# --------------------------------------------------------------------------
# FEATURES  (Table 4, Interim Report + base statistics)
# --------------------------------------------------------------------------
BASE_FEATURES = ["max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form"]
# ^ The 9 raw time-domain statistical features computed directly from a
# vibration signal segment — these are what a technician types in manually
# on the Live Prediction page's "Manual Feature Entry" mode.

ENGINEERED_FEATURES = [
    # The 10 additional features derived from the 9 base features above
    # (ratios, squares, products) that the preprocessing pipeline computes
    # automatically — a user never has to supply these directly.
    "peak_to_peak", "rms_mean_ratio", "crest_form_ratio", "kurtosis_squared",
    "skewness_abs", "rms_sd_ratio", "max_min_ratio", "kurtosis_crest",
    "mean_squared", "rms_squared",
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES  # 19 total, fixed column order
# ^ Concatenates base + engineered features into the exact 19-column order
# every model expects its input matrix to be in — this order must never change.
N_FEATURES = len(ALL_FEATURES)  # 19 — used to size input tensors / validate uploaded files.

FEATURE_DESCRIPTIONS = {
    # Plain-English description of every one of the 19 features, used as
    # tooltips (st.number_input's `help=`) and in the XAI explanation table
    # on the Live Prediction page so a non-expert can understand each value.
    "max": "Maximum amplitude of the vibration signal segment",
    "min": "Minimum amplitude of the vibration signal segment",
    "mean": "Mean amplitude (DC offset) of the signal",
    "sd": "Standard deviation — overall signal dispersion",
    "rms": "Root Mean Square — total vibration energy; rises with defect severity",
    "skewness": "Distribution asymmetry of the signal amplitude",
    "kurtosis": "Impulsiveness indicator; >3 suggests early-stage pitting/impacts",
    "crest": "Crest factor (peak / RMS) — impulsiveness relative to energy",
    "form": "Form factor (RMS / mean absolute value) — waveform shape indicator",
    "peak_to_peak": "max - min, total signal swing",
    "rms_mean_ratio": "rms / mean — energy relative to central tendency",
    "crest_form_ratio": "crest / form — combined shape/impulsiveness ratio",
    "kurtosis_squared": "kurtosis^2 — emphasises strongly impulsive signals",
    "skewness_abs": "|skewness| — magnitude of asymmetry, direction-agnostic",
    "rms_sd_ratio": "rms / sd — energy relative to dispersion",
    "max_min_ratio": "max / min — amplitude range ratio",
    "kurtosis_crest": "kurtosis * crest — compound impulsiveness feature",
    "mean_squared": "mean^2 — nonlinear central-tendency term",
    "rms_squared": "rms^2 — proportional to vibration power",
}

TARGET_ACCURACY = 0.95          # Project's accuracy goal (95%) — compared against the best model's real accuracy on the Executive Dashboard.
TARGET_INFERENCE_MS = 150       # Project's latency goal (150ms/sample) for "near-real-time" monitoring.
TARGET_FALSE_POSITIVE_RATE = 0.05  # Project's target ceiling on false-positive fault alerts.

# --------------------------------------------------------------------------
# MODEL REGISTRY
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Single source of truth for every trained model: which framework
    # loads it, where its file lives, what input shape it expects, and a
    # human-readable description — utils/model_loader.py iterates this
    # dict to load all six models, and pages/2_Live_Prediction.py uses the
    # keys as the model-selector dropdown options.
    "Two-Dimensional Convolutional Neural Network (2D CNN)": {
        "id": "cnn2d",                                  # Short internal id (unused elsewhere currently, kept for reference).
        "framework": "keras",                            # Tells model_loader.py to use the Keras loading path.
        "file": MODELS_DIR / "cnn_2d_model.keras",        # Path to the saved model weights/architecture file.
        "input_shape": (5, 4, 1),                         # Expected (height, width, channels) after reshaping the 19 features.
        "description": "2D Convolutional Network. Features zero-padded from 19 to 20 and "
                        "reshaped into a 5x4 pseudo-image so spatial convolution filters can "
                        "learn local statistical-feature interactions.",
        "supports_shap": True,                            # Whether the XAI layer can compute SHAP values for this model.
        "shap_mode": "gradient",                          # Which SHAP explainer algorithm to use (gradient-based for Keras models).
    },
    "Long Short-Term Memory (LSTM)": {
        "id": "lstm",
        "framework": "keras",
        "file": MODELS_DIR / "lstm_model.keras",
        "input_shape": (1, 19),                           # Treated as a 1-timestep sequence of 19 features.
        "description": "Bidirectional-style stacked LSTM treating the 19-feature vector as a "
                        "single timestep sequence; two LSTM layers extract sequential feature "
                        "dependencies before dense classification heads.",
        "supports_shap": True,
        "shap_mode": "gradient",
    },
    "Transformer": {
        "id": "transformer",
        "framework": "keras",
        "file": MODELS_DIR / "final_transformer_model.keras",
        "input_shape": (19, 1),                           # Each of the 19 features is one "token" fed to self-attention.
        "description": "Self-attention Transformer classifier: each of the 19 features is "
                        "treated as a token, allowing multi-head attention to learn "
                        "inter-feature relationships directly.",
        "supports_shap": True,
        "shap_mode": "gradient",
    },
    "Model-Agnostic Meta-Learning (MAML)": {
        "id": "maml",
        "framework": "torch",                             # Tells model_loader.py to use the PyTorch loading path.
        "file": MODELS_DIR / "maml_model.pth",
        "input_shape": (19,),                             # Flat 19-feature vector, no reshaping needed.
        "description": "Model-Agnostic Meta-Learning transformer encoder, self-supervised "
                        "pretrained via masked-feature reconstruction, then fine-tuned for "
                        "fast adaptation across fault classes.",
        "supports_shap": True,
        "shap_mode": "kernel",                            # Model-agnostic Kernel SHAP is used for the PyTorch models instead of gradient SHAP.
    },
    "Meta Stochastic Gradient Descent (Meta-SGD)": {
        "id": "meta_sgd",
        "framework": "torch",
        "file": MODELS_DIR / "meta_sgd.pth",
        "input_shape": (19,),
        "description": "Meta-SGD transformer encoder that learns both initial weights and "
                        "per-parameter learning rates for rapid adaptation to new fault "
                        "conditions.",
        "supports_shap": True,
        "shap_mode": "kernel",
    },
    "Feature-Based Contrastive Learning (FBCL)": {
        "id": "fbcl",
        "framework": "torch",
        "file": MODELS_DIR / "fbcl_model.pth",
        "input_shape": (19,),
        "description": "Feature Boosting Continual Learning transformer: a shared backbone "
                        "with additive boosting heads trained sequentially across defect "
                        "severities via knowledge distillation, mitigating catastrophic "
                        "forgetting.",
        "supports_shap": True,
        "shap_mode": "kernel",
    },
}
MODEL_NAMES = list(MODEL_REGISTRY.keys())  # Ordered list of the 6 model display names, used to populate model-selector dropdowns.

# --------------------------------------------------------------------------
# UI THEME  — Industrial Dark / Blue
# --------------------------------------------------------------------------
COLORS = {
    # Central colour palette for the "Industrial Dark" theme — every page
    # references these instead of hard-coding hex values, so the whole
    # theme can be changed from this one dict.
    "bg_primary": "#0B1220",       # Main page background colour.
    "bg_secondary": "#111A2E",     # Secondary panel/section background.
    "bg_card": "#152238",          # Background colour for KPI/alert cards.
    "bg_card_hover": "#1B2C4A",    # Card background on hover.
    "border": "#22314D",          # Default border colour for cards/panels.
    "accent_blue": "#2E8FFF",      # Primary accent colour (charts, highlights).
    "accent_blue_light": "#5CB2FF",# Lighter variant of the primary accent.
    "accent_cyan": "#22D3EE",      # Secondary accent colour (charts).
    "accent_green": "#22C55E",     # Positive/success/"Low risk" colour.
    "accent_amber": "#F59E0B",     # Warning/"Medium risk" colour.
    "accent_red": "#EF4444",       # Danger/"High risk" colour.
    "text_primary": "#E8EEFC",     # Main text colour.
    "text_secondary": "#93A2C2",   # Secondary/muted text colour.
    "text_muted": "#5E6C8C",       # Even more muted text (captions, hints).
    "gold": "#FFD700",             # Rank-1 medal colour (e.g. leaderboards).
    "silver": "#C0C0C0",           # Rank-2 medal colour.
    "bronze": "#CD7F32",           # Rank-3 medal colour.
}

RISK_COLORS = {
    # Maps each maintenance risk level (utils/maintenance.py) to the colour
    # used to render its badge/alert card, reusing the palette above.
    "Low": COLORS["accent_green"],
    "Medium": COLORS["accent_amber"],
    "High": COLORS["accent_red"],
    "Critical": "#B91C1C",  # A darker red than accent_red, to visually distinguish "Critical" from "High".
}

PLOTLY_TEMPLATE = "plotly_dark"  # Built-in Plotly theme applied to every chart so charts match the dark UI theme.

# --------------------------------------------------------------------------
# LLM REASONING LAYER  (see utils/llm_assistant.py)
# --------------------------------------------------------------------------
# No API key is ever hard-coded here. This dashboard uses Google Gemini
# exclusively as its LLM provider. The dashboard reads the key (if present)
# from environment variables or .streamlit/secrets.toml at runtime:
#   GEMINI_API_KEY  -> the only supported provider key
# If it isn't configured, the LLM features fall back to clearly-labelled
# rule-based / template responses grounded in this dashboard's real data.
# Model name is overridable via the DASHBOARD_LLM_MODEL env var / secret.
LLM_DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"  # Default Gemini model id used when no override is configured.
