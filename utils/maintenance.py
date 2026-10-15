"""
utils/maintenance.py
=====================
Maps a predicted fault class (+ confidence) to maintenance recommendations,
severity and risk level, following the structure requested for Page 5.
"""

from config import FAULT_FAMILY, SEVERITY_MAP, CLASS_DISPLAY_NAMES
# ^ FAULT_FAMILY: raw class name (e.g. "IR_014") -> broad family (e.g. "Inner Race Fault").
# SEVERITY_MAP: defect-size code (e.g. "014") -> qualitative severity word (e.g. "Moderate").
# CLASS_DISPLAY_NAMES: raw class name -> human-readable label shown in the UI.

RECOMMENDATIONS = {
    # The single source of truth for maintenance guidance: one entry per
    # broad fault family, each with a fixed risk level and a list of
    # recommended actions. Both pages/2_Live_Prediction.py and
    # pages/4_Maintenance_Recommendations.py read from this same dict, so
    # the guidance is always consistent everywhere it's shown.
    "Normal": {
        "risk": "Low",  # A healthy bearing carries no elevated risk.
        "actions": ["Machine Healthy", "Continue Routine Monitoring", "No maintenance action required"],
    },
    "Ball Fault": {
        "risk": "Medium",  # Ball faults are treated as a moderate, non-urgent concern.
        "actions": ["Inspect Bearing", "Monitor Temperature", "Schedule Maintenance",
                    "Track vibration trend over the next 2-4 weeks"],
    },
    "Inner Race Fault": {
        "risk": "High",  # Inner-race faults typically progress faster and warrant prompt attention.
        "actions": ["Immediate Inspection", "Bearing Replacement Recommended",
                    "Reduce load / speed until inspected", "Log for root-cause analysis"],
    },
    "Outer Race Fault": {
        "risk": "Critical",  # Outer-race faults are treated as the most urgent family in this ruleset.
        "actions": ["High Risk Alert", "Shutdown Recommendation", "Lubrication Check",
                    "Alignment Inspection", "Notify maintenance supervisor immediately"],
    },
}


def get_recommendation(class_name: str, confidence: float = None) -> dict:
    """
    class_name: one of config.CLASS_NAMES (e.g. 'OR_014_6')
    Returns dict with family, severity, risk, actions, display_name.
    """
    # Step 1: resolve which broad family this specific class belongs to
    # (defaults to "Ball Fault" if the class name is somehow unrecognised,
    # so this never raises for a bad/unexpected input).
    family = FAULT_FAMILY.get(class_name, "Ball Fault")
    # Step 2: look up that family's fixed risk/actions ruleset, and copy()
    # it so any per-call mutation below (e.g. appending a low-confidence
    # warning) never leaks back into the shared RECOMMENDATIONS dict.
    rec = RECOMMENDATIONS.get(family, RECOMMENDATIONS["Ball Fault"]).copy()

    # Step 3: extract the defect-size code (007/014/021) embedded in the
    # class name, if any, so we can translate it into a severity word.
    sev_code = None
    for code in ("007", "014", "021"):
        if code in class_name:
            sev_code = code
            break
    # "Normal" has no defect size, so its severity is always "N/A" rather
    # than accidentally matching a code substring in the word "Normal".
    severity = SEVERITY_MAP.get(sev_code, "N/A") if family != "Normal" else "N/A"

    risk = rec["risk"]  # Pull the family's fixed risk level out for the return value below.
    # Escalate risk if confidence is low (model unsure) -> recommend manual verification
    # A prediction below 60% confidence is treated as "the model isn't
    # sure", so a caveat is appended to the action list (but the risk
    # level itself is left unchanged — only the guidance text changes).
    low_confidence = confidence is not None and confidence < 0.60
    actions = list(rec["actions"])  # Copy the actions list so appending below doesn't mutate the shared RECOMMENDATIONS dict.
    if low_confidence and family != "Normal":
        # Only add this caveat for actual fault predictions — a low-confidence
        # "Normal" reading doesn't need a manual-inspection warning appended.
        actions.append("⚠ Model confidence is low — confirm with manual vibration inspection")

    # Bundle everything the UI needs into one dict: which family this is,
    # its human-readable display name, severity, risk level, and the final
    # (possibly caveat-appended) list of recommended actions.
    return {
        "family": family,
        "display_name": CLASS_DISPLAY_NAMES.get(class_name, class_name),
        "severity": severity,
        "risk": risk,
        "actions": actions,
    }
