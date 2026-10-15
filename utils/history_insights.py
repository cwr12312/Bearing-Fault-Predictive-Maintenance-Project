"""
utils/history_insights.py
==========================
Maintenance History built from the dashboard's recorded predictions.

Output = one table (most trending fault, how many times it was predicted, which models predicted it
the most, its risk, latest-week count) followed by trending recommendations in point form.
The text box can also be used to ASK questions ("what is the most trending fault?",
"best model of the week", "how many outer race faults this week?").

Every number is counted straight from the recorded rows. No randomness, no guessing:
the same log and the same question always give the same answer.
No Streamlit import here, so it can be tested on its own.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections import Counter, defaultdict

from config import CLASS_DISPLAY_NAMES, CLASS_NAMES, FAULT_FAMILY, SEVERITY_MAP
from utils.maintenance import RECOMMENDATIONS

_SEV_ORDER = {"007": 0, "014": 1, "021": 2}
MIN_FOR_PATTERN = 3


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------

def _parse_ts(s):
    try:
        return _dt.datetime.strptime((s or "").strip()[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _monday(d: _dt.date) -> _dt.date:
    return d - _dt.timedelta(days=d.weekday())


def _fmt(d: _dt.date) -> str:
    return d.strftime("%d %b %Y").lstrip("0")


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "0.0%"


def _rd(n: int) -> str:
    return f"{n} reading" + ("" if n == 1 else "s")


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _sev(cls: str) -> str | None:
    return next((c for c in _SEV_ORDER if c in cls), None)


def _risk_of_class(cls: str) -> str:
    return RECOMMENDATIONS.get(FAULT_FAMILY.get(cls, ""), {}).get("risk", "Unknown")


def prepare(records: list[dict] | None, include_unreliable: bool = False):
    """Keep reliable rows only (unless include_unreliable); add parsed time. Returns (rows, excluded_count)."""
    rows, excluded = [], 0
    for r in records or []:
        if r.get("reliable") is False and not include_unreliable:
            excluded += 1
            continue
        cls = r.get("fault_class", "")
        try:
            conf = float(r.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        rows.append({
            "cls": cls, "model": (r.get("model") or "Unknown model").strip() or "Unknown model",
            "conf": conf, "ts": _parse_ts(r.get("timestamp", "")),
            "family": r.get("family") or FAULT_FAMILY.get(cls, "Unknown"),
        })
    return rows, excluded


def _latest_week(rows):
    """(monday, sunday) of the newest recorded week, or None when no row has a date."""
    stamps = [r["ts"] for r in rows if r["ts"]]
    if not stamps:
        return None
    m = _monday(max(stamps).date())
    return m, m + _dt.timedelta(days=6)


def _in_week(rows, wk):
    return [r for r in rows if r["ts"] and wk[0] <= r["ts"].date() <= wk[1]]


def _weekly_counts(rows, classes: set[str]):
    """[(week_monday, count_of_these_classes, all_rows)] for every calendar week from first to last record."""
    dated = [r for r in rows if r["ts"]]
    if not dated:
        return []
    first, last = _monday(min(r["ts"] for r in dated).date()), _monday(max(r["ts"] for r in dated).date())
    out, w = [], first
    while w <= last:
        wk = (w, w + _dt.timedelta(days=6))
        in_wk = _in_week(dated, wk)
        out.append((w, sum(1 for r in in_wk if r["cls"] in classes), len(in_wk)))
        w += _dt.timedelta(days=7)
    return out


def _fault_counter(rows):
    return Counter(r["cls"] for r in rows if r["cls"] != "Normal")


def _pick_top(rows, counter: Counter):
    """Most predicted fault. Ties: most recent first, then the larger defect. Returns (class, tied_classes)."""
    if not counter:
        return None, []
    best = max(counter.values())
    tied = [c for c, n in counter.items() if n == best]
    last_seen = {}
    for r in rows:
        if r["cls"] in tied:
            t = r["ts"] or _dt.datetime.min
            last_seen[r["cls"]] = max(last_seen.get(r["cls"], _dt.datetime.min), t)
    tied.sort(key=lambda c: (last_seen.get(c, _dt.datetime.min), _SEV_ORDER.get(_sev(c) or "", -1)), reverse=True)
    return tied[0], tied[1:]


def _model_phrase(counter: Counter) -> tuple[str, str]:
    """('CNN', '9 times') or ('A and B', '6 times each') for the model(s) with the highest count."""
    if not counter:
        return "n/a", ""
    top = max(counter.values())
    names = sorted(m for m, n in counter.items() if n == top)
    if len(names) == 1:
        return names[0], _times(top)
    joined = " and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + " and " + names[-1]
    return joined, ("once each" if top == 1 else f"{top} times each")


# --------------------------------------------------------------------------
# Main table + recommendations
# --------------------------------------------------------------------------

def trending_summary(rows) -> dict | None:
    counter = _fault_counter(rows)
    top, tied = _pick_top(rows, counter)
    if not top:
        return None
    total = len(rows)
    n = counter[top]
    models = Counter(r["model"] for r in rows if r["cls"] == top)
    wk = _latest_week(rows)
    weekly = _weekly_counts(rows, {top})
    latest_n = None
    if wk:
        latest_n = sum(1 for r in _in_week(rows, wk) if r["cls"] == top)
    direction = None
    if len(weekly) >= 2:
        prev, last = weekly[-2][1], weekly[-1][1]
        direction = ("increasing" if last > prev else "decreasing" if last < prev else "unchanged", prev, last)
    fam = FAULT_FAMILY.get(top, "Unknown")
    prog = None
    seq = [_sev(r["cls"]) for r in sorted([r for r in rows if r["family"] == fam and r["cls"] != "Normal"],
                                         key=lambda r: r["ts"] or _dt.datetime.min) if _sev(r["cls"])]
    if len(seq) >= 2 and any(r["ts"] for r in rows):
        a, b = seq[0], seq[-1]
        prog = ("worsening" if _SEV_ORDER[b] > _SEV_ORDER[a] else "easing" if _SEV_ORDER[b] < _SEV_ORDER[a] else "steady", a, b)
    return {"cls": top, "display": CLASS_DISPLAY_NAMES.get(top, top), "count": n, "total": total, "tied": tied,
            "models": models, "risk": _risk_of_class(top), "family": fam, "week": wk, "latest_n": latest_n,
            "direction": direction, "progression": prog, "fault_total": sum(counter.values())}


def _recommendations(t: dict) -> list[str]:
    rec = RECOMMENDATIONS.get(t["family"], {})
    out = [f"Treat **{t['display']}** as the priority: it is the most predicted fault ({_times(t['count'])} "
           f"in {_rd(t['total'])}, {t['risk']} risk)."]
    out += list(rec.get("actions", []))
    if t["count"] >= MIN_FOR_PATTERN:
        out.append("Repeat fault: open a root-cause analysis (check alignment, lubrication, mounting and load).")
    elif t["total"] < MIN_FOR_PATTERN:
        out.append(f"Only {t['total']} reading(s) recorded so far: record more predictions before relying on this trend.")
    if t["direction"] and t["direction"][0] == "increasing":
        out.append(f"Getting more frequent: {t['direction'][1]} in the previous week, {t['direction'][2]} in the latest "
                   "week. Shorten the inspection interval.")
    if t["progression"] and t["progression"][0] == "worsening":
        a, b = t["progression"][1], t["progression"][2]
        out.append(f"Severity is rising ({SEVERITY_MAP[a].lower()} 0.{a}in to {SEVERITY_MAP[b].lower()} 0.{b}in): "
                   "plan the replacement before it fails.")
    if len(t["models"]) > 1:
        out.append(f"{len(t['models'])} different models predicted this fault, so treat it as likely real.")
    if t["tied"]:
        names = ", ".join(CLASS_DISPLAY_NAMES.get(c, c) for c in t["tied"])
        out.append(f"Also predicted {_times(t['count'])}, the same count: {names}. Check them as well.")
    seen, final = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


def _table_md(t: dict) -> list[str]:
    top_model, top_n = _model_phrase(t["models"])
    models_all = ", ".join(f"{m} ({n})" for m, n in sorted(t["models"].items(), key=lambda kv: (-kv[1], kv[0])))
    L = ["| Item | Result |", "|---|---|",
         f"| Most trending fault (most predicted) | {t['display']} |",
         f"| Times predicted | {t['count']} out of {t['total']} recorded {'reading' if t['total'] == 1 else 'readings'} ({_pct(t['count'], t['total'])}) |",
         f"| Model that predicted it the most | {top_model} ({top_n}) |",
         f"| All models that predicted it | {models_all} |",
         f"| Risk of this fault | {t['risk']} |"]
    if t["week"] and t["latest_n"] is not None:
        L.append(f"| Predicted in the latest week ({_fmt(t['week'][0])} to {_fmt(t['week'][1])}) | {t['latest_n']} |")
    if t["direction"]:
        word, prev, last = t["direction"]
        L.append(f"| Weekly trend | {word.capitalize()} ({prev} in the previous week, {last} in the latest week) |")
    return L


# --------------------------------------------------------------------------
# Questions typed in the box
# --------------------------------------------------------------------------

_Q_STARTS = re.compile(r"^\s*(what|which|who|how|when|where|is|are|does|do|did|show|tell|list|give|can|compare|"
                       r"has|have|was|were|any)\b")


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    return "?" in t or bool(_Q_STARTS.match(t)) or len(t.split()) <= 8 and bool(
        re.search(r"\b(trend\w*|best model|top model|risk|weekly|per week|recommend\w*|most predicted|most common)\b", t))


def _faults_in_question(q: str) -> tuple[set[str], str]:
    """Which fault classes does the question talk about? Returns (classes, label). Empty = none named."""
    fam = None
    if re.search(r"\bouter\b", q):
        fam = "Outer Race Fault"
    elif re.search(r"\binner\b", q):
        fam = "Inner Race Fault"
    elif re.search(r"\bball\b|\brolling\b", q):
        fam = "Ball Fault"
    elif re.search(r"\bnormal\b|\bhealthy\b|\bno fault\b", q):
        fam = "Normal"
    if not fam:
        return set(), ""
    sizes = [code for code, pat in (("007", r"(?<![\d])0?\.?007\b"), ("014", r"(?<![\d])0?\.?014\b"),
                                    ("021", r"(?<![\d])0?\.?021\b")) if re.search(pat, q)]
    classes = {c for c in CLASS_NAMES if FAULT_FAMILY.get(c) == fam and (not sizes or _sev(c) in sizes)}
    if sizes and fam != "Normal":
        label = f"{fam} ({', '.join('0.' + s + 'in' for s in sizes)})"
    else:
        label = "healthy (Normal)" if fam == "Normal" else fam
    return classes, label


def _week_note(wk, today=None) -> str:
    today = today or _dt.date.today()
    base = f"Latest recorded week: {_fmt(wk[0])} to {_fmt(wk[1])}."
    if _monday(today) != wk[0]:
        base += " (No readings are dated in the current calendar week, so \"this week\" here means that latest recorded week.)"
    return base


def _model_table(rows) -> tuple[list[str], list]:
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    stats = []
    for m, rs in by.items():
        stats.append((m, len(rs), sum(1 for r in rs if r["cls"] != "Normal"), sum(r["conf"] for r in rs) / len(rs)))
    stats.sort(key=lambda s: (-s[3], -s[1], s[0]))
    L = ["| Model | Readings | Fault readings | Average confidence |", "|---|---|---|---|"]
    for m, n, f, c in stats:
        L.append(f"| {m} | {n} | {f} | {c * 100:.1f}% |")
    return L, stats


def answer_question(question: str, rows, excluded: int, best_overall: tuple | None = None, today=None) -> str | None:
    """Answers the typed question from the rows. Returns None if the text is not a question this tool understands."""
    q = " ".join(question.lower().split())
    if not q:
        return None
    classes, label = _faults_in_question(q)
    wk = _latest_week(rows)
    q_nopw = re.sub(r"\b(per week|weekly|each week|by week|week by week)\b", " ", q)   # these ask for a table, not one week
    wants_week = bool(re.search(r"\b(week|latest)\b", q_nopw))
    parts: list[str] = []
    matched = False

    def scope_rows():
        return _in_week(rows, wk) if (wants_week and wk) else rows

    scope_txt = "in the latest recorded week" if (wants_week and wk) else "in the whole history"

    # --- best / top model ------------------------------------------------------------------
    if re.search(r"\b(best|top|strongest|most (?:confident|reliable|accurate)|model of the week)\b.*\bmodel|"
                 r"\bmodel\b.*\b(best|top|most|predict\w*|confident)|\bwhich model\b|\bmodels?\b", q):
        matched = True
        sub = scope_rows()
        if classes:
            sub = [r for r in sub if r["cls"] in classes]
        if not sub:
            parts.append("**Models:** no readings match that question yet.")
        else:
            by_fault = bool(classes) or bool(re.search(r"\b(predict\w*|most|fault)\b", q) and not re.search(r"\bbest\b|confiden", q))
            if by_fault and not classes:
                faults = [r for r in sub if r["cls"] != "Normal"]
                cnt = Counter(r["model"] for r in faults)
                name, n = _model_phrase(cnt)
                parts.append(f"**Model that flagged the most faults {scope_txt}:** {name} ({n}).")
            elif by_fault and classes:
                cnt = Counter(r["model"] for r in sub)
                name, n = _model_phrase(cnt)
                parts.append(f"**Model that predicted {label} the most {scope_txt}:** {name} ({n}).")
            tbl, stats = _model_table(sub)
            top = stats[0]
            parts += ["", f"**Best model {scope_txt}** (highest average confidence): **{top[0]}**, "
                          f"{top[3] * 100:.1f}% over {top[1]} reading(s).", ""] + tbl
            note = ("Confidence is how sure a model was, not proof it was right: the log holds no confirmed answers.")
            if best_overall:
                note += f" On the test set, the most accurate model overall is {best_overall[0]} ({best_overall[1] * 100:.2f}%)."
            parts += ["", f"_{note}_"]
            if wants_week and wk:
                parts += ["", f"_{_week_note(wk, today)}_"]
            elif wants_week and not wk:
                parts.append("_These records have no dates, so the whole history was used._")

    # --- trending / most predicted fault -----------------------------------------------------
    if re.search(r"\b(trend\w*|most (?:common|frequent|predicted|repeated|repeating)|top fault|frequent|"
                 r"commonest|keeps? (?:coming|repeating)|repeat\w*|biggest problem)\b", q) and not (
            matched and re.search(r"\bmodel", q)):
        matched = True
        sub = scope_rows()
        counter = _fault_counter(sub)
        if not counter:
            parts.append(f"**Most trending fault:** no faults were recorded {scope_txt}; every reading was healthy.")
        else:
            top, tied = _pick_top(sub, counter)
            n = counter[top]
            tot = len(sub)
            line = (f"**Most trending fault {scope_txt}:** {CLASS_DISPLAY_NAMES.get(top, top)}, predicted "
                    f"{_times(n)} out of {_rd(tot)} ({_pct(n, tot)}). Risk: {_risk_of_class(top)}.")
            parts.append(line)
            if tied:
                parts.append("Same count: " + ", ".join(CLASS_DISPLAY_NAMES.get(c, c) for c in tied) + ".")
            parts += ["", "| Fault | Times predicted | Risk |", "|---|---|---|"]
            for c, k in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
                parts.append(f"| {CLASS_DISPLAY_NAMES.get(c, c)} | {k} | {_risk_of_class(c)} |")
            if wants_week and wk:
                parts += ["", f"_{_week_note(wk, today)}_"]

    # --- counts of a named fault / totals ------------------------------------------------------
    if classes and not matched or (classes and re.search(r"\b(how many|count|times|number)\b", q)):
        matched = True
        sub = scope_rows()
        k = sum(1 for r in sub if r["cls"] in classes)
        parts.append(f"**{label.capitalize() if label.startswith('healthy') else label}:** predicted "
                     f"{_times(k)} out of {_rd(len(sub))} {scope_txt} ({_pct(k, len(sub))}).")
        if k and not label.startswith("healthy"):
            mc = Counter(r["model"] for r in sub if r["cls"] in classes)
            nm, nn = _model_phrase(mc)
            parts.append(f"Most often predicted by {nm} ({nn}).")
        if wants_week and wk:
            parts.append(f"_{_week_note(wk, today)}_")
    elif re.search(r"\bhow many\b.*\b(reading|prediction|record|sample)s?\b|\btotal\b", q):
        matched = True
        parts.append(f"**Recorded readings:** {len(rows)} reliable ({sum(1 for r in rows if r['cls'] == 'Normal')} healthy, "
                     f"{sum(1 for r in rows if r['cls'] != 'Normal')} fault)"
                     + (f"; {excluded} left out as unreliable." if excluded else "."))

    # --- weekly table ------------------------------------------------------------------------
    if re.search(r"\b(per week|weekly|each week|by week|week by week)\b", q):
        matched = True
        cl = classes or ({_fault_counter(rows).most_common(1)[0][0]} if _fault_counter(rows) else set())
        weeks = _weekly_counts(rows, cl)
        if not weeks:
            parts.append("**Per week:** these records have no dates, so weekly counts are not possible.")
        else:
            nm = label or (CLASS_DISPLAY_NAMES.get(next(iter(cl)), "") if cl else "")
            parts += ["", f"**Per week: {nm}**", "", "| Week starting (Monday) | Times predicted | All readings that week |", "|---|---|---|"]
            for w, c, t in weeks[-8:]:
                parts.append(f"| {_fmt(w)} | {c} | {t} |")

    # --- risk ----------------------------------------------------------------------------------
    if re.search(r"\b(risk|danger|critical|severity|serious)\b", q) and not classes:
        matched = True
        counter = _fault_counter(rows)
        top, _ = _pick_top(rows, counter)
        rc = Counter(_risk_of_class(r["cls"]) for r in rows if r["cls"] != "Normal")
        if top:
            parts.append(f"**Risk of the most trending fault ({CLASS_DISPLAY_NAMES.get(top, top)}):** {_risk_of_class(top)}.")
            parts.append("Fault readings by risk level: " + ", ".join(f"{k} {v}" for k, v in
                         sorted(rc.items(), key=lambda kv: ["Critical", "High", "Medium", "Low"].index(kv[0]) if kv[0] in
                                ["Critical", "High", "Medium", "Low"] else 9)) + ".")
        else:
            parts.append("**Risk:** no faults recorded, so the risk level is Low.")
    elif classes and re.search(r"\b(risk|danger|critical|serious)\b", q):
        matched = True
        c0 = sorted(classes)[0]
        parts.append(f"**Risk of {label}:** {_risk_of_class(c0)}.")

    # --- recommendations -------------------------------------------------------------------------
    if re.search(r"\b(recommend\w*|what (?:should (?:i|we) do|to do|can (?:i|we) do)|what do (?:i|we) do|actions?|advice|next steps?|fix)\b", q):
        matched = True
        t = trending_summary(rows)
        if classes and not label.startswith("healthy"):
            fam = FAULT_FAMILY.get(sorted(classes)[0])
            acts = RECOMMENDATIONS.get(fam, {}).get("actions", [])
            parts += [f"**What to do for {label}:**", ""] + [f"- {a}" for a in acts]
        elif t:
            parts += [f"**What to do about the most trending fault ({t['display']}):**", ""] + [f"- {a}" for a in _recommendations(t)]
        else:
            parts.append("**What to do:** no faults recorded, so continue routine monitoring.")

    # --- getting worse ------------------------------------------------------------------------------
    if re.search(r"\b(worse|worsen\w*|getting worse|progress\w*|escalat\w*|deteriorat\w*)\b", q):
        matched = True
        t = trending_summary(rows)
        if classes and not label.startswith("healthy"):
            fam = FAULT_FAMILY.get(sorted(classes)[0])
            seq = [_sev(r["cls"]) for r in sorted([r for r in rows if r["family"] == fam and r["cls"] != "Normal"],
                                                 key=lambda r: r["ts"] or _dt.datetime.min) if _sev(r["cls"])]
            if len(seq) >= 2 and any(r["ts"] for r in rows):
                a, b = seq[0], seq[-1]
                kind = ("worsening" if _SEV_ORDER[b] > _SEV_ORDER[a] else "easing" if _SEV_ORDER[b] < _SEV_ORDER[a] else "steady")
                t = {"progression": (kind, a, b), "family": fam}
            else:
                t = {"progression": None, "family": fam}
        if t and t["progression"]:
            kind, a, b = t["progression"]
            msg = {"worsening": "is getting worse", "easing": "is easing", "steady": "is steady"}[kind]
            parts.append(f"**{t['family']}** {msg}: {SEVERITY_MAP[a].lower()} (0.{a}in) at first, "
                         f"{SEVERITY_MAP[b].lower()} (0.{b}in) most recently.")
        elif t:
            parts.append(f"**{t['family']}:** not enough dated readings of different sizes to say whether it is getting worse.")
        else:
            parts.append("No faults recorded, so nothing is getting worse.")

    return "\n".join(parts).strip() if matched and parts else None


_HELP = ("I can answer questions about the recorded history. Try: \"What is the most trending fault?\", "
         "\"Which model is best this week?\", \"How many outer race faults this week?\", "
         "\"What is the risk?\", \"Fault counts per week\" or \"What should I do?\".")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def history_markdown(records, question_or_notes: str = "", best_overall: tuple | None = None,
                     notes_summary=None, today=None) -> str:
    rows, excluded = prepare(records)
    L = ["### Maintenance History", ""]
    text = (question_or_notes or "").strip()

    all_unreliable = False
    if not rows and excluded:
        # Every recorded prediction was flagged as unreliable. Still show the history table (so the page is
        # never empty) and warn clearly that the counts come from inputs outside the training data.
        rows, excluded = prepare(records, include_unreliable=True)
        all_unreliable = True
        L += [f"> **Note:** all {len(rows)} recorded prediction(s) came from inputs far outside the data the models "
              "were trained on, so treat this history as indicative only. Run predictions on realistic inputs "
              "(for example rows from the test set) for a dependable trend.", ""]

    if not rows:
        return ("### Maintenance History\n\nNo predictions have been recorded yet, so there is nothing to count. "
                "Every prediction is saved automatically; the most trending fault will show here as the record grows.")

    answer = None
    notes_part = None
    if text:
        answer = answer_question(text, rows, excluded, best_overall, today)
        if answer is None:
            if _looks_like_question(text):
                answer = _HELP
            elif notes_summary is not None:
                notes_part = notes_summary(text)
    if answer:
        L += ["#### Your question", "", f"> {' '.join(text.split())}", "", answer, "", "---", ""]

    t = trending_summary(rows)
    if not t:
        L += ["| Item | Result |", "|---|---|",
              "| Most trending fault | None: every recorded reading was healthy |",
              f"| Healthy readings | {len(rows)} out of {len(rows)} ({_pct(len(rows), len(rows))}) |",
              "| Risk | Low |", "",
              "#### Most trending recommendations", "",
              "- Machine healthy: continue routine monitoring.",
              "- No maintenance action required."]
    else:
        L += _table_md(t)
        L += ["", "#### Most trending recommendations", ""] + [f"- {x}" for x in _recommendations(t)]

    dated = [r["ts"] for r in rows if r["ts"]]
    span = f" from {_fmt(min(dated).date())} to {_fmt(max(dated).date())}" if dated else ""
    foot = f"Counted from {len(rows)} recorded prediction(s){span}."
    if excluded:
        foot += f" {excluded} reading(s) left out because the input was unreliable."
    L += ["", f"_{foot}_"]
    if notes_part:
        L += ["", "---", "", notes_part]
    return "\n".join(L)
