"""
utils/history_store.py
=======================
Persistent record of every fault the dashboard has predicted.

WHY THIS EXISTS
---------------
The Maintenance History and Multimodal Fusion tools need to know which faults
the system has been *recording* and which of them keep repeating. The Live
Prediction page only keeps a short in-memory table (``st.session_state``,
last 50 rows) that disappears on refresh, so on its own it cannot show a
long-running pattern.

Every prediction logged to that table is now also appended to
``logs/prediction_log.csv`` (one row per prediction). The analysis functions in
``utils/llm_assistant.py`` read from here.

HONESTY NOTE
------------
These rows are *model predictions*, not confirmed work orders. Nothing in this
file invents data: if the log is empty, the tools say so.

If the file system is read-only (some hosting setups), logging silently falls
back to the session table, so the page never crashes.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

from config import CLASS_DISPLAY_NAMES, FAULT_FAMILY

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "prediction_log.csv"
COLUMNS = ["timestamp", "model", "fault_class", "prediction", "family", "confidence", "risk", "reliable"]

# Safety cap for one batch upload so a single large file cannot swamp the history.
MAX_ROWS_PER_BATCH = 200
# Only the most recent rows are loaded for analysis.
MAX_ROWS_LOADED = 5000


def record_prediction(model: str, fault_class: str, confidence: float, risk: str, reliable: bool = True) -> bool:
    """Append one prediction to the persistent log. Returns True on success."""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "fault_class": fault_class,
        "prediction": CLASS_DISPLAY_NAMES.get(fault_class, fault_class),
        "family": FAULT_FAMILY.get(fault_class, "Unknown"),
        "confidence": f"{float(confidence):.4f}",
        "risk": risk,
        "reliable": "yes" if reliable else "no",
    }
    return record_many([row])


def record_many(rows: list[dict]) -> bool:
    """Append several rows (already shaped like COLUMNS). Never raises."""
    if not rows:
        return True
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLUMNS})
        return True
    except OSError:
        return False


def record_batch(model: str, fault_classes, confidences, risks, reliables=None) -> bool:
    """Log a batch upload (capped at MAX_ROWS_PER_BATCH rows)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    reliables = list(reliables) if reliables is not None else [True] * len(list(fault_classes))
    for c, cf, rk, rl in list(zip(fault_classes, confidences, risks, reliables))[:MAX_ROWS_PER_BATCH]:
        rows.append({
            "timestamp": now, "model": model, "fault_class": c,
            "prediction": CLASS_DISPLAY_NAMES.get(c, c),
            "family": FAULT_FAMILY.get(c, "Unknown"),
            "confidence": f"{float(cf):.4f}", "risk": rk, "reliable": "yes" if rl else "no",
        })
    return record_many(rows)


def _read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    out = []
    for r in rows[-MAX_ROWS_LOADED:]:
        try:
            r["confidence"] = float(r.get("confidence") or 0.0)
        except ValueError:
            r["confidence"] = 0.0
        r["reliable"] = (r.get("reliable") or "yes").strip().lower() != "no"
        out.append(r)
    return out


def _from_session(session_history: list[dict] | None) -> list[dict]:
    """Convert the Live Prediction page's session table into log-shaped rows (oldest first)."""
    display_to_class = {v: k for k, v in CLASS_DISPLAY_NAMES.items()}
    out = []
    for h in reversed(session_history or []):  # session table is newest-first
        disp = h.get("Prediction", "Unknown")
        cls = display_to_class.get(disp, disp)
        try:
            conf = float(str(h.get("Confidence", "0")).replace("%", "")) / 100.0
        except ValueError:
            conf = 0.0
        out.append({
            "timestamp": "", "model": h.get("Model", ""), "fault_class": cls,
            "prediction": disp, "family": FAULT_FAMILY.get(cls, "Unknown"),
            "confidence": conf, "risk": h.get("Risk", ""), "reliable": True,
        })
    return out


def load_records(session_history: list[dict] | None = None) -> list[dict]:
    """
    Oldest-first list of recorded predictions. Uses the persistent log when it
    has data; otherwise falls back to this session's table.
    """
    rows = _read_log()
    return rows if rows else _from_session(session_history)


def clear_log() -> bool:
    """Delete the persistent log (used by the 'Clear recorded log' button)."""
    try:
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        return True
    except OSError:
        return False
