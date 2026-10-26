"""
utils/styling.py
=================
CSS injection and small reusable HTML component builders (KPI cards,
badges, section titles, alert cards) used across every page so the look
stays consistent without duplicating markup.
"""

import streamlit as st                                          # renders HTML/markdown components in the app
from config import CSS_DIR, COLORS, APP_NAME, DESIGNER_CREDIT    # shared style constants and app metadata


def inject_css():
    css_path = CSS_DIR / "style.css"                              # path to the shared stylesheet
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)  # inject the raw CSS into the page so all pages share the theme


def page_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.12em;
                        color:{COLORS['accent_blue_light']};text-transform:uppercase;">
                {icon} INDUSTRIAL PREDICTIVE MAINTENANCE
            </div>
            <h1 style="margin:2px 0 2px 0;">{title}</h1>
            <div style="color:{COLORS['text_secondary']};font-size:0.95rem;">{subtitle}</div>
        </div>
        <hr style="margin:10px 0 22px 0;">
        """,
        unsafe_allow_html=True,
    )                                                                # renders a consistent page banner: small eyebrow label, title, subtitle, divider


def section_title(text: str):
    st.markdown(
        f'<div class="section-title"><div class="bar"></div><h3>{text}</h3></div>',
        unsafe_allow_html=True,
    )                                                                # renders a small heading with a colored accent bar (styled via .section-title in CSS)


def kpi_card(label: str, value: str, sub: str = "", delta: str = None, delta_positive: bool = True):
    delta_html = ""                                                  # optional trend indicator, empty unless a delta value is supplied
    if delta:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"    # choose green "up" or red "down" styling class
        arrow = "▲" if delta_positive else "▼"                          # matching arrow glyph
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'          # build the delta markup
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )                                                                     # renders a single KPI card (label, big value, subtext, optional delta)


def badge(text: str, kind: str = "blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'               # small colored pill label, color controlled by "kind" (blue/green/amber/red)


def medal(rank: int) -> str:
    return {1: '<span class="rank-gold">🥇</span>', 2: '<span class="rank-silver">🥈</span>',
            3: '<span class="rank-bronze">🥉</span>'}.get(rank, f"#{rank}")   # gold/silver/bronze medal emoji for top 3 ranks, plain "#N" otherwise


def alert_card(title: str, items: list[str], risk: str = "low"):
    risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high",
                  "Critical": "risk-critical"}.get(risk, "risk-low")          # map risk level text to its CSS class, default to low risk
    items_html = "".join(f'<div class="alert-item">• {i}</div>' for i in items)  # bullet-list each item passed in
    st.markdown(
        f"""
        <div class="alert-card {risk_class}">
            <div class="alert-title">{title}</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )                                                                          # renders a risk-colored alert/recommendation card


def footer():
    """
    Renders the single, consistent footer for every page. Per project
    requirements, the "Designed by" credit appears ONLY here — nowhere
    else in the sidebar, navigation, or page headers.
    """
    st.markdown(
        f"""
        <div class="app-footer">
            Industrial Predictive Maintenance System for Bearing Fault Diagnosis &middot;
            Built with Streamlit, TensorFlow, PyTorch &amp; Plotly
            <br>
            <span style="opacity:0.85;">Designed by {DESIGNER_CREDIT}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active: str = ""):
    """
    Renders the shared sidebar: brand header, then the primary navigation
    (App, About, Model Comparison, Live Prediction, Maintenance
    Recommendations, Dataset Explorer, Model Performance), then the live
    model-load status at the bottom. Called identically from every page so
    the sidebar never changes shape as the user navigates. No separate
    LLM/AI page — that functionality lives inside the Live Prediction page
    itself.
    """
    from utils.model_loader import load_all_models  # local import avoids circular import

    st.session_state.setdefault("show_agent_panel", False)      # tracks whether the floating "Manage Agent" chat panel is open
    st.session_state.setdefault("agent_chat_history", [])         # persists the agent chat messages across reruns

    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;padding:10px 0 18px 0;">
                <div style="font-weight:800;font-size:1.05rem;color:{COLORS['text_primary']};">
                    {APP_NAME.upper()}
                </div>
                <div style="font-size:0.72rem;color:{COLORS['accent_blue_light']};
                            letter-spacing:0.1em;">BEARING FAULT DIAGNOSTICS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )                                                            # brand header block at the top of the sidebar
        st.markdown("---")
        st.caption("NAVIGATION")
        st.page_link("app.py", label="App")                           # link to the main app entry point
        st.page_link("pages/1_About.py", label="About")
        st.page_link("pages/1_Model_Comparison.py", label="Model Comparison")
        st.page_link("pages/2_Live_Prediction.py", label="Live Prediction")
        st.page_link("pages/4_Maintenance_Recommendations.py", label="Maintenance Recommendations")
        st.page_link("pages/5_Dataset_Explorer.py", label="Dataset Explorer")
        st.page_link("pages/6_Model_Performance.py", label="Model Performance")
        if st.button("Manage Agent", key="nav_manage_agent", use_container_width=False):
            st.session_state.show_agent_panel = not st.session_state.show_agent_panel   # toggle the floating agent panel open/closed
        st.markdown("---")
        st.caption("SYSTEM STATUS")
        registry = load_all_models()                                    # get the current model load status (cached)
        n_ok = len(registry["models"])                                    # count of successfully loaded models
        n_err = len(registry["errors"])                                    # count of models that failed to load
        if n_err == 0:
            st.success(f"**{n_ok}/6 Models Loaded**")                        # all models loaded fine -> green success message
        else:
            st.warning(f"**{n_ok}/6 Models Loaded** &middot; {n_err} failed")  # some failed -> amber warning message
            with st.expander("Load errors"):
                for name, msg in registry["errors"].items():
                    st.caption(f"**{name}**: {msg}")                          # list each failed model's error detail

    # "Manage Agent" floating panel — rendered outside the sidebar `with`
    # block (see docstring below) so it overlays the whole page, not just
    # the sidebar column.
    _render_agent_panel()                                                # draw the floating chat panel if it's currently open


def _render_agent_panel():
    """
    Renders the floating "Manage Agent" chat panel triggered by the
    "Manage Agent" sidebar button above. This is a pure CSS overlay
    (position: fixed, see the "Manage Agent" rules in css/style.css) around
    real Streamlit widgets — it never touches, resizes, or re-renders any
    existing page content; when closed, this function simply returns and
    nothing about the underlying page is different.

    The chat itself is answered by utils.llm_assistant.chat_with_agent(),
    which reuses the exact same call_llm() provider/template layer as every
    other LLM feature in this dashboard — no second agent/backend exists.
    """
    if not st.session_state.get("show_agent_panel", False):
        return

    from utils.llm_assistant import chat_with_agent

    # ==============================================================
    # FLOATING MANAGE AGENT PANEL
    # ==============================================================
    with st.container(key="manage_agent_panel"):

        # Header
        head_l, head_r = st.columns([8, 1])

        with head_l:
            st.markdown(
                '<div class="agent-panel-header">Manage Agent</div>',
                unsafe_allow_html=True,
            )

        with head_r:
            close_clicked = st.button(
                "✕",
                key="agent_panel_close_btn",
            )

        # ==========================================================
        # CHAT HISTORY
        # ==========================================================
        with st.container(
            key="agent_panel_body",
            height=400,
            border=False,
        ):
            history = st.session_state.get(
                "agent_chat_history",
                [],
            )

            if not history:
                st.markdown(
                    '<div class="agent-panel-empty">'
                    'Ask about model performance, a prediction, '
                    'or maintenance guidance.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            for msg in history:
                bubble_class = (
                    "agent-bubble-user"
                    if msg.get("role") == "user"
                    else "agent-bubble-assistant"
                )

                safe_text = (
                    str(msg.get("content", ""))
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

                st.markdown(
                    f'<div class="agent-bubble {bubble_class}">'
                    f'{safe_text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ==========================================================
        # MESSAGE INPUT
        # ==========================================================
        with st.form(
            key="agent_panel_form",
            clear_on_submit=True,
            border=False,
        ):
            in_col, send_col = st.columns([5, 1])

            with in_col:
                user_msg = st.text_input(
                    "Message",
                    placeholder="Type your message...",
                    label_visibility="collapsed",
                )

            with send_col:
                sent = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                )

    # ==============================================================
    # CLOSE PANEL
    # ==============================================================
    if close_clicked:
        st.session_state.show_agent_panel = False
        st.rerun()

    # ==============================================================
    # PROCESS MESSAGE
    # ==============================================================
    if sent and user_msg and user_msg.strip():

        clean_message = user_msg.strip()

        st.session_state.agent_chat_history.append(
            {
                "role": "user",
                "content": clean_message,
            }
        )

        # Maximum 20 stored messages
        st.session_state.agent_chat_history = (
            st.session_state.agent_chat_history[-20:]
        )

        try:
            with st.spinner("Thinking..."):
                reply, _mode = chat_with_agent(
                    st.session_state.agent_chat_history
                )

            if not reply:
                reply = (
                    "I couldn't generate a response right now. "
                    "Please try again."
                )

        except Exception as e:
            print("=" * 60)
            print("MANAGE AGENT ERROR")
            print("=" * 60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("=" * 60)

            reply = (
                "The Manage Agent could not process that message "
                "right now. The rest of the dashboard is still available."
            )

        st.session_state.agent_chat_history.append(
            {
                "role": "assistant",
                "content": str(reply),
            }
        )

        st.session_state.agent_chat_history = (
            st.session_state.agent_chat_history[-20:]
        )

        st.rerun()