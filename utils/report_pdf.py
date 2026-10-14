"""
utils/report_pdf.py
====================
Builds the "Predictive Maintenance Assessment Report" as a real, multi-page PDF
(ReportLab). Every number and sentence comes from dashboard state:

  * the current prediction (class, confidence, risk, probabilities),
  * the sensor features and the input-reliability check,
  * the Root Cause / Maintenance History / Multimodal Fusion outputs, but only
    if the user actually generated them (otherwise the section says so),
  * the recorded fault log, and the model leaderboard.

Nothing is invented. Sections that were not generated are marked as such.
"""

from __future__ import annotations

import datetime
import io
import re
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

NAVY = colors.HexColor("#14213D")
ACCENT = colors.HexColor("#E07A1F")
LIGHT = colors.HexColor("#F3F5F9")
GRID = colors.HexColor("#CBD2DE")
TEXT = colors.HexColor("#222B38")
RISK_COLORS = {"Low": "#2E9D5B", "Medium": "#D9A400", "High": "#E07A1F", "Critical": "#C62828"}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                      textColor=TEXT, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#5B6678"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=NAVY,
                    spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY,
                    spaceBefore=6, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=colors.white)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11, leading=15, textColor=colors.HexColor("#DDE3EE"))


# ----------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------
def _inline(text: str) -> str:
    """Escape for ReportLab Paragraph and convert **bold** / _italic_ markdown."""
    t = escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w])_(.+?)_(?![\w])", r"<i>\1</i>", t)
    return t.replace("  \n", "<br/>")


def _p(text: str, style=BODY) -> Paragraph:
    return Paragraph(_inline(text), style)


def markdown_to_flowables(md: str) -> list:
    """Small markdown renderer: paragraphs, bullets, numbered items, pipe tables."""
    out, lines, i = [], md.replace("\r", "").split("\n"), 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip() or ln.strip() == "---":
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            if rows:
                out.append(_table(rows))
                out.append(Spacer(1, 4))
            continue
        hm = re.match(r"^(#{1,6})\s+(.*)", ln)
        if hm:
            level = len(hm.group(1))
            out.append(Spacer(1, 3))
            out.append(Paragraph(_inline(hm.group(2)), H2 if level <= 3 else ParagraphStyle(
                "h3", parent=BODY, fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=NAVY,
                spaceBefore=4, spaceAfter=2)))
            i += 1
            continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)", ln)
        if m:
            out.append(Paragraph(_inline(m.group(2)), BULLET, bulletText=f"{m.group(1)}."))
        elif re.match(r"^\s*[-*]\s+", ln):
            out.append(Paragraph(_inline(re.sub(r"^\s*[-*]\s+", "", ln)), BULLET, bulletText="•"))
        elif ln.startswith("   _Evidence"):
            out.append(Paragraph(_inline(ln.strip()), ParagraphStyle("ev", parent=BULLET, leftIndent=24,
                                                                      textColor=colors.HexColor("#5B6678"))))
        else:
            out.append(_p(ln.strip()))
        i += 1
    return out


def _table(rows: list[list[str]], col_widths=None, header=True, zebra=True) -> Table:
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    data = []
    for ri, r in enumerate(rows):
        st = CELLH if (header and ri == 0) else CELL
        data.append([Paragraph(_inline(c), st) for c in r])
    if col_widths is None:
        col_widths = [CONTENT_W / n] * n
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for ri in range(1 if header else 0, len(data)):
            if ri % 2 == 0:
                style.append(("BACKGROUND", (0, ri), (-1, ri), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def _kv_table(pairs: list[tuple[str, str]], label_w=42 * mm) -> Table:
    data = [[Paragraph(_inline(k), CELLB), Paragraph(_inline(v), CELL)] for k, v in pairs]
    t = Table(data, colWidths=[label_w, CONTENT_W - label_w])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, GRID), ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _section(title: str, num: int) -> list:
    bar = Drawing(CONTENT_W, 3)
    bar.add(Rect(0, 0, CONTENT_W, 1.2, fillColor=ACCENT, strokeColor=None))
    return [Paragraph(f"{num}. {escape(title)}", H1), bar, Spacer(1, 4)]


def _bar_chart(items: list[tuple[str, float]], title: str, unit: str = "", color="#E07A1F",
               height_per_bar=15, max_value=None) -> Drawing:
    items = items[:10]
    n = max(len(items), 1)
    h = 38 + n * height_per_bar
    d = Drawing(CONTENT_W, h)
    d.add(String(0, h - 12, title, fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
    if not items:
        return d
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 150, 8, CONTENT_W - 190, n * height_per_bar
    ch.data = [[v for _, v in items][::-1]]
    ch.categoryAxis.categoryNames = [k for k, _ in items][::-1]
    ch.categoryAxis.labels.fontName = "Helvetica"
    ch.categoryAxis.labels.fontSize = 7.5
    ch.categoryAxis.labels.dx = -4
    ch.categoryAxis.visibleTicks = 0
    ch.valueAxis.valueMin = 0
    ch.valueAxis.valueMax = max_value or max(v for _, v in items) * 1.15 or 1
    ch.valueAxis.labels.fontSize = 7
    ch.valueAxis.labelTextFormat = "%.0f" + unit.replace("%", "%%")
    ch.bars[0].fillColor = colors.HexColor(color)
    ch.bars[0].strokeColor = None
    ch.barLabelFormat = "%.1f" if unit else "%d"
    ch.barLabels.nudge = 8
    ch.barLabels.fontSize = 7
    ch.bars.strokeColor = None
    d.add(ch)
    return d


# ----------------------------------------------------------------------------
# page furniture
# ----------------------------------------------------------------------------
def _make_doc(buf, title, generated):
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=22 * mm,
                          bottomMargin=18 * mm, title=title, author="Predictive Maintenance Dashboard",
                          subject="Bearing fault assessment")

    def _cover(canv, d):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_H - 95 * mm, PAGE_W, 95 * mm, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE_H - 98 * mm, PAGE_W, 3 * mm, stroke=0, fill=1)
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#5B6678"))
        canv.drawString(MARGIN, 11 * mm, "Decision-support document. Not a certified inspection. Verify before acting.")
        canv.drawRightString(PAGE_W - MARGIN, 11 * mm, f"Page {d.page}")
        canv.restoreState()

    def _inner(canv, d):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE_H - 12.8 * mm, PAGE_W, 0.8 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 8.5)
        canv.drawString(MARGIN, PAGE_H - 8 * mm, "PREDICTIVE MAINTENANCE ASSESSMENT REPORT")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 8 * mm, generated)
        canv.setStrokeColor(GRID)
        canv.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canv.setFillColor(colors.HexColor("#5B6678"))
        canv.setFont("Helvetica", 8)
        canv.drawString(MARGIN, 9.5 * mm, "Decision-support document. Not a certified inspection. Verify before acting.")
        canv.drawRightString(PAGE_W - MARGIN, 9.5 * mm, f"Page {d.page}")
        canv.restoreState()

    cover_frame = Frame(MARGIN, 18 * mm, CONTENT_W, PAGE_H - 40 * mm, id="cover", leftPadding=0, rightPadding=0)
    body_frame = Frame(MARGIN, 18 * mm, CONTENT_W, PAGE_H - 40 * mm, id="body", leftPadding=0, rightPadding=0)
    doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_frame], onPage=_cover),
                          PageTemplate(id="inner", frames=[body_frame], onPage=_inner)])
    return doc


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def build_report_pdf(ctx: dict) -> bytes:
    """
    ctx keys (all real dashboard state):
      asset, predicted_display, confidence, model_name, risk, family, actions, probabilities
      (list of (class display, percent)), features (list of (name, value)), reliability (dict|None),
      root_cause_md, history_md, fusion_md (str|None), history_records (list), best_model (name, acc),
      leaderboard (list of rows), notes (dict of user-typed text used for the outputs)
    """
    now = datetime.datetime.now()
    generated = now.strftime("%d %b %Y, %H:%M")
    ref_no = "PMR-" + now.strftime("%Y%m%d-%H%M%S")
    buf = io.BytesIO()
    doc = _make_doc(buf, f"Predictive Maintenance Assessment {ref_no}", generated)

    risk = ctx.get("risk", "Unknown")
    risk_hex = RISK_COLORS.get(risk, "#5B6678")
    pred = ctx["predicted_display"]
    conf = ctx["confidence"]
    rel = ctx.get("reliability")

    story: list = []

    # ---- cover -------------------------------------------------------------
    story += [Paragraph("Predictive Maintenance<br/>Assessment Report", TITLE), Spacer(1, 4),
              Paragraph("Bearing fault diagnosis and decision support", SUB), Spacer(1, 3),
              Paragraph(f"Reference {ref_no}  |  Generated {generated}", SUB)]
    story += [Spacer(1, 33 * mm)]  # clears the navy band drawn by the cover template
    status_tbl = Table([[Paragraph("<font color='white'><b>CURRENT CONDITION</b></font>", CELL),
                         Paragraph("<font color='white'><b>RISK LEVEL</b></font>", CELL),
                         Paragraph("<font color='white'><b>MODEL CONFIDENCE</b></font>", CELL)],
                        [Paragraph(f"<font size=12><b>{escape(pred)}</b></font>", CELL),
                         Paragraph(f"<font size=16 color='{risk_hex}'><b>{escape(risk)}</b></font>", CELL),
                         Paragraph(f"<font size=16><b>{conf*100:.1f}%</b></font>", CELL)]],
                       colWidths=[CONTENT_W * 0.5, CONTENT_W * 0.25, CONTENT_W * 0.25])
    status_tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                                    ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID), ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
                                    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [status_tbl, Spacer(1, 8)]
    if rel is not None and not rel.get("in_distribution", True):
        story += [Paragraph("<font color='#C62828'><b>Reliability warning:</b></font> the input for this prediction "
                            "lies far outside the data the models were trained on. Treat the predicted fault as "
                            "unverified (see section 3).", BODY)]
    story += [Spacer(1, 6), _kv_table([
        ("Asset", ctx.get("asset", "Live Prediction Sample")), ("Model used", ctx["model_name"]),
        ("Fault family", ctx.get("family", "n/a")), ("Report reference", ref_no),
        ("Prepared by", "Predictive Maintenance Dashboard (automated)"),
    ])]

    from reportlab.platypus import NextPageTemplate
    story += [NextPageTemplate("inner"), PageBreak()]

    # ---- 1. executive summary ---------------------------------------------
    n = 1
    story += _section("Executive Summary", n)
    summary = (f"The **{ctx['model_name']}** model classified the current vibration sample as **{pred}** with "
               f"**{conf*100:.1f}%** confidence. The associated maintenance risk is **{risk}**.")
    story.append(_p(summary))
    recs = ctx.get("history_records") or []
    reliable_recs = [r for r in recs if r.get("reliable") is not False]
    if reliable_recs:
        story.append(_p(f"The system has recorded **{len(reliable_recs)}** reliable prediction(s) in total; "
                        "see section 6 for the faults that repeat most often."))
    if rel is not None and not rel.get("in_distribution", True):
        story.append(_p("**Important:** this particular input is out of range for the trained models, so the "
                        "prediction should be verified before any maintenance decision is made."))
    story.append(_p("**Recommended priority actions:**"))
    for a in ctx.get("actions", []):
        story.append(Paragraph(_inline(a), BULLET, bulletText="•"))

    # ---- 2. prediction details --------------------------------------------
    n += 1
    story += _section("Prediction Details", n)
    probs = ctx.get("probabilities") or []
    if probs:
        story.append(_bar_chart(sorted(probs, key=lambda kv: -kv[1]), "Model probability by class (%)", unit="%",
                                max_value=100))
        top = sorted(probs, key=lambda kv: -kv[1])[:3]
        rows = [["Rank", "Class", "Probability"]] + [[str(i), c, f"{p:.1f}%"] for i, (c, p) in enumerate(top, 1)]
        story += [Spacer(1, 4), _table(rows, [18 * mm, CONTENT_W - 50 * mm, 32 * mm])]
    else:
        story.append(_p("Class probabilities were not available for this prediction."))

    # ---- 3. sensor features + reliability ---------------------------------
    n += 1
    story += _section("Sensor Features and Input Reliability", n)
    feats = ctx.get("features") or []
    if feats:
        rows = [["Feature", "Value (as entered)"]] + [[k, f"{v:.4g}"] for k, v in feats]
        story.append(_table(rows, [CONTENT_W * 0.5, CONTENT_W * 0.5]))
        story.append(Spacer(1, 6))
    if rel is None:
        story.append(_p("No reference data was available to compare this sample with the training data."))
    elif rel.get("in_distribution", True):
        story.append(_p("**Reliability check: passed.** The sample's feature pattern lies within the range of the "
                        "training data for the predicted class."))
        if not rel.get("nearest_agrees", True):
            story.append(_p(f"Note: its overall pattern is closest to **{rel.get('nearest_display', 'another class')}**, "
                            "so confirm the fault type on site."))
    else:
        story.append(_p("**Reliability check: FAILED.** The sample lies far outside the training data, so the "
                        "prediction may reflect the input values rather than the bearing."))
        for b in rel.get("out_of_range", []):
            story.append(Paragraph(_inline(f"{b['feature']} = {b['value']:.4g} (training range "
                                           f"{b['lo']:.3g} to {b['hi']:.3g})"), BULLET, bulletText="•"))

    # ---- 4. root cause -----------------------------------------------------
    n += 1
    story += _section("Root Cause Analysis", n)
    story += markdown_to_flowables(ctx["root_cause_md"]) if ctx.get("root_cause_md") else [
        _p("_Not generated in this session. Use the Root Cause Reasoning tool on the Live Prediction page._")]

    # ---- 5. multimodal fusion ---------------------------------------------
    n += 1
    story += _section("Multimodal Fusion Assessment", n)
    story += markdown_to_flowables(ctx["fusion_md"]) if ctx.get("fusion_md") else [
        _p("_Not generated in this session. Use the Multimodal Fusion tool on the Live Prediction page._")]

    # ---- 6. maintenance history -------------------------------------------
    n += 1
    story += _section("Maintenance History (recorded faults)", n)
    story += markdown_to_flowables(ctx["history_md"]) if ctx.get("history_md") else [
        _p("_Not generated in this session. Use the Mine Maintenance History tool on the Live Prediction page._")]

    # ---- 7. model context --------------------------------------------------
    n += 1
    board = ctx.get("leaderboard") or []
    story += _section("Model Context", n)
    best = ctx.get("best_model")
    if best:
        story.append(_p(f"Best-performing model overall: **{best[0]}** ({best[1]*100:.2f}% test accuracy)."))
    if board:
        rows = [["Model", "Accuracy", "F1"]] + [[r[0], f"{r[1]*100:.2f}%", f"{r[2]:.3f}" if r[2] is not None else "n/a"]
                                                  for r in board]
        story.append(_table(rows, [CONTENT_W * 0.6, CONTENT_W * 0.2, CONTENT_W * 0.2]))

    # ---- 8. method and limitations ----------------------------------------
    n += 1
    story += _section("Method, Limitations and Sign-off", n)
    for t in [
        "Predictions come from machine-learning models trained on bearing vibration features (CWRU data, 10 classes).",
        "Root-cause and fusion outputs are rule-based decision support; they do not replace a physical inspection.",
        "Maintenance history is built from the dashboard's own prediction log. These are model predictions, not "
        "confirmed work orders, and are always subject to human review.",
        "Sections marked 'not generated' were not run during this session; nothing has been filled in on their behalf.",
    ]:
        story.append(Paragraph(_inline(t), BULLET, bulletText="•"))
    story.append(Spacer(1, 10))
    sign = Table([[Paragraph("Reviewed by (name):", CELL), "", Paragraph("Date:", CELL), ""],
                  [Paragraph("Signature:", CELL), "", Paragraph("Decision:", CELL), ""]],
                 colWidths=[32 * mm, 62 * mm, 22 * mm, CONTENT_W - 116 * mm], rowHeights=[12 * mm, 12 * mm])
    sign.setStyle(TableStyle([("LINEBELOW", (1, 0), (1, 0), 0.5, TEXT), ("LINEBELOW", (3, 0), (3, 0), 0.5, TEXT),
                              ("LINEBELOW", (1, 1), (1, 1), 0.5, TEXT), ("LINEBELOW", (3, 1), (3, 1), 0.5, TEXT),
                              ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    story.append(KeepTogether([sign]))

    doc.build(story)
    return buf.getvalue()
