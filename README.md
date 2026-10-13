# Industrial Predictive Maintenance Dashboard

Industrial Predictive Maintenance System for Bearing Fault Diagnosis Using Advanced Deep Learning Models

A production-styled Streamlit dashboard for diagnosing rolling-element bearing faults using
six trained deep-learning / meta-learning / continual-learning models Two-Dimensional
Convolutional Neural Network (2D CNN), Long Short-Term Memory (LSTM), Transformer,
Model-Agnostic Meta-Learning (MAML), Meta Stochastic Gradient Descent (Meta-SGD) and
Feature-Based Contrastive Learning (FBCL) evaluated on the CWRU Bearing Data Center dataset.
An LLM reasoning layer sits on top of these six models, entirely within the Live Prediction
page, to explain predictions in plain language, answer natural-language questions, reason
about root causes, mine maintenance notes, generate reports, generate scenarios, and fuse
multiple modalities.


## 1. Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows users can instead double-click **`run_dashboard.bat`**, which creates a virtual
environment, installs dependencies and launches the app automatically.

The app opens at `http://localhost:8501`.

