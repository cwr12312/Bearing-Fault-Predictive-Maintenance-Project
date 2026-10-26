"""Page 5 (nav) — Maintenance Recommendations: rule-based guidance per predicted fault."""

import streamlit as st

from config import PROJECT_SHORT_TITLE, CLASS_NAMES, CLASS_DISPLAY_NAMES, RISK_COLORS
# alert_card renders a coloured (green/amber/red/critical) card of recommended actions.
from utils.styling import inject_css, page_header, section_title, alert_card, footer, render_sidebar
# get_recommendation: looks up the rule-based action list + risk level for one fault class.
# RECOMMENDATIONS: the full dict of fault-family -> {risk, actions} used in the reference grid below.
from utils.maintenance import get_recommendation, RECOMMENDATIONS

st.set_page_config(page_title=f"Maintenance Recommendations · {PROJECT_SHORT_TITLE}", layout="wide")
inject_css()
render_sidebar(active="maintenance")
page_header("Maintenance Recommendations", "Actionable guidance mapped to each predicted fault condition")

# --------------------------------------------------------------------------
# INTERACTIVE "WHAT-IF" SELECTOR
# Lets the user pick any of the 10 fault classes + a simulated confidence
# level to preview exactly what recommendation the system would show for
# that combination, without needing to run a real model prediction.
# --------------------------------------------------------------------------
section_title("Select a Fault Class")
choice = st.selectbox(
    "Predicted class", CLASS_NAMES,
    format_func=lambda c: CLASS_DISPLAY_NAMES.get(c, c),
    index=CLASS_NAMES.index("Normal"),
)
confidence = st.slider("Simulated prediction confidence", 0.0, 1.0, 0.92, 0.01)

# Same rule-based lookup used by the Live Prediction page's real
# predictions — here it's driven by the manual selector above instead of
# an actual model output, so this page can be explored on its own.
rec = get_recommendation(choice, confidence)
st.write("")
# Renders a colour-coded (green/amber/red/critical) card of recommended
# maintenance actions for the currently selected fault + confidence combo.
alert_card(
    f"{rec['display_name']}  ·  Severity: {rec['severity']}  ·  Risk: {rec['risk']}",
    rec["actions"],
    risk=rec["risk"],
)

st.write("")
st.write("")
section_title("Full Recommendation Reference")

# Illustrative example scenario labels shown to give a quick sense of when
# each fault family's guidance applies (not wired into the cards below).
family_examples = {
    "Normal": ["Machine Healthy", "Continue Routine Monitoring"],
    "Ball Fault": ["Inspect Bearing", "Monitor Temperature", "Schedule Maintenance"],
    "Inner Race Fault": ["Immediate Inspection", "Bearing Replacement Recommended"],
    "Outer Race Fault": ["High Risk Alert", "Shutdown Recommendation", "Lubrication Check", "Alignment Inspection"],
}

# One reference card per fault family (Normal, Ball/Inner Race/Outer Race Fault),
# laid out side by side regardless of which class is selected above.
cols = st.columns(4)
for col, (family, rules) in zip(cols, RECOMMENDATIONS.items()):
    with col:
        alert_card(f"{family} — Risk: {rules['risk']}", rules["actions"], risk=rules["risk"])

footer()
