"""
utils/history_analysis.py
==========================
Turns the recorded prediction log (see utils/history_store.py) into a real
maintenance-history picture: which faults the system has recorded, which
ones repeat the most, whether a fault is getting more severe, and how the
CURRENT predicted fault fits into that history.

Everything is computed from recorded rows. Nothing is invented, and there is
no randomness: the same log always gives the same result.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter

from config import CLASS_DISPLAY_NAMES, FAULT_FAMILY, SEVERITY_MAP
from utils.maintenance import RECOMMENDATIONS

# Fewer records than this is too little to call anything a "pattern".
MIN_RECORDS_FOR_PATTERN = 3

_SEV_ORDER = {"007": 0, "014": 1, "021": 2}


def _sev_code(fault_class: str) -> str | None:
    for code in _SEV_ORDER:
        if code in fault_class:
            return code
    return None


def _risk_of(family: str) -> str:
    return RECOMMENDATIONS.get(family, {}).get("risk", "Unknown")


def class_from_display(predicted_display: str) -> str | None:
    for cname, cdisp in CLASS_DISPLAY_NAMES.items():
        if cdisp == predicted_display:
            return cname
    return None


def analyze(records: list[dict], predicted_display: str | None = None) -> dict:
    """
    records: oldest-first rows with keys fault_class, family, risk, model, timestamp, confidence.
    Returns a plain dict of facts used by both Maintenance History and Multimodal Fusion.
    """
    excluded = sum(1 for r in records if r.get("reliable") is False)
    records = [r for r in records if r.get("reliable") is not False]
    total = len(records)
    cur_class = class_from_display(predicted_display) if predicted_display else None
    cur_family = FAULT_FAMILY.get(cur_class) if cur_class else None

    res: dict = {
        "excluded": excluded, "total": total, "enough": total >= MIN_RECORDS_FOR_PATTERN,
        "cur_class": cur_class, "cur_family": cur_family,
        "fault_total": 0, "normal_total": 0,
        "class_counts": [], "family_counts": [], "repeating": [],
        "top_class": None, "top_family": None,
        "cur_count": 0, "cur_family_count": 0, "cur_rank": None, "cur_streak": 0,
        "cur_first": None, "cur_last": None, "cur_models": {},
        "progression": None, "trend": None, "risk_counts": {},
        "first_ts": None, "last_ts": None,
    }
    if not total:
        return res

    classes = [r.get("fault_class", "") for r in records]
    families = [r.get("family") or FAULT_FAMILY.get(r.get("fault_class", ""), "Unknown") for r in records]
    stamps = [r.get("timestamp") or "" for r in records]
    stamped = [s for s in stamps if s]
    res["first_ts"], res["last_ts"] = (stamped[0], stamped[-1]) if stamped else (None, None)

    res["normal_total"] = sum(1 for c in classes if c == "Normal")
    res["fault_total"] = total - res["normal_total"]
    res["risk_counts"] = dict(Counter(r.get("risk", "Unknown") for r in records))

    # Fault-only rankings (healthy readings are not "faults the system keeps recording").
    cc = Counter(c for c in classes if c != "Normal")
    fc = Counter(f for f, c in zip(families, classes) if c != "Normal")
    last_seen: dict[str, str] = {}
    for c, s in zip(classes, stamps):
        last_seen[c] = s
    res["class_counts"] = [
        {"class": c, "display": CLASS_DISPLAY_NAMES.get(c, c), "count": n,
         "share": n / total, "family": FAULT_FAMILY.get(c, "Unknown"),
         "risk": _risk_of(FAULT_FAMILY.get(c, "")), "last_seen": last_seen.get(c, "")}
        for c, n in cc.most_common()
    ]
    res["family_counts"] = [{"family": f, "count": n, "share": n / total, "risk": _risk_of(f)}
                            for f, n in fc.most_common()]
    res["repeating"] = [x for x in res["class_counts"] if x["count"] >= 2]
    if res["class_counts"]:
        res["top_class"] = res["class_counts"][0]
    if res["family_counts"]:
        res["top_family"] = res["family_counts"][0]

    if cur_class:
        res["cur_count"] = classes.count(cur_class)
        res["cur_family_count"] = families.count(cur_family) if cur_family != "Normal" else classes.count("Normal")
        ranked = [x["class"] for x in res["class_counts"]]
        if cur_class in ranked:
            res["cur_rank"] = (ranked.index(cur_class) + 1, len(ranked))
        streak = 0
        for c in reversed(classes):
            if c == cur_class:
                streak += 1
            else:
                break
        res["cur_streak"] = streak
        idx = [i for i, c in enumerate(classes) if c == cur_class]
        if idx:
            res["cur_first"], res["cur_last"] = stamps[idx[0]] or None, stamps[idx[-1]] or None
        res["cur_models"] = dict(Counter(r.get("model", "") for r in records
                                         if r.get("fault_class") == cur_class))

        # Severity progression inside the same fault family (e.g. 0.007in -> 0.014in).
        if cur_family and cur_family != "Normal":
            sev_seq = [_sev_code(c) for c, f in zip(classes, families) if f == cur_family and _sev_code(c)]
            if len(sev_seq) >= 2:
                first, last = sev_seq[0], sev_seq[-1]
                if _SEV_ORDER[last] > _SEV_ORDER[first]:
                    res["progression"] = ("worsening", first, last)
                elif _SEV_ORDER[last] < _SEV_ORDER[first]:
                    res["progression"] = ("easing", first, last)
                else:
                    res["progression"] = ("steady", first, last)

    # Trend: share of fault readings in the older half vs the newer half of the log.
    if total >= 6:
        half = total // 2
        old = classes[:half]
        new = classes[half:]
        old_rate = sum(1 for c in old if c != "Normal") / len(old)
        new_rate = sum(1 for c in new if c != "Normal") / len(new)
        res["trend"] = (old_rate, new_rate)

    res["weekly"] = _weekly_stats(classes, stamps, cur_class)
    res["cur_trend"] = _current_fault_trend(classes, res["weekly"], cur_class)
    return res


def _parse_ts(s: str):
    try:
        return _dt.datetime.strptime((s or "").strip()[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _weekly_stats(classes: list[str], stamps: list[str], cur_class: str | None) -> dict | None:
    """
    How often the current fault was recorded in each calendar week (Monday to Sunday).
    Anchored on the LATEST recorded timestamp, never on the clock, so the numbers
    never change just because time passes or a button is pressed again.
    Returns None when no record carries a usable timestamp.
    """
    parsed = [(i, _parse_ts(s)) for i, s in enumerate(stamps)]
    parsed = [(i, d) for i, d in parsed if d is not None]
    if not parsed or not cur_class:
        return None
    first = min(d for _, d in parsed).date()
    last = max(d for _, d in parsed).date()
    span_days = (last - first).days + 1
    monday = lambda d: d - _dt.timedelta(days=d.weekday())
    weeks, w = [], monday(first)
    while w <= monday(last):
        weeks.append(w)
        w += _dt.timedelta(days=7)
    tot = {w: 0 for w in weeks}
    cur = {w: 0 for w in weeks}
    for i, d in parsed:
        wk = monday(d.date())
        tot[wk] += 1
        if classes[i] == cur_class:
            cur[wk] += 1
    cur_total = sum(cur.values())
    same_second = Counter(stamps[i] for i, _ in parsed)
    return {
        "weeks": [(w, cur[w], tot[w]) for w in weeks],
        "span_days": span_days, "first": first, "last": last,
        "cur_total": cur_total, "dated": len(parsed), "undated": len(stamps) - len(parsed),
        "avg_per_week": (cur_total / (span_days / 7)) if span_days >= 7 else None,
        "avg_per_day": cur_total / span_days,
        "batch_max": max(same_second.values()) if same_second else 0,
    }


def _current_fault_trend(classes: list[str], weekly: dict | None, cur_class: str | None) -> dict | None:
    """Is the current fault becoming more or less common? Weeks if there are 2+, else older vs newer half."""
    if not cur_class or cur_class == "Normal":
        return None
    if weekly:
        used = [(c, t) for _, c, t in weekly["weeks"] if t > 0][-2:]
        if len(used) == 2:
            (c0, t0), (c1, t1) = used
            return {"basis": "weeks", "old": c0 / t0, "new": c1 / t1, "old_n": c0, "new_n": c1}
    if len(classes) >= 6:
        half = len(classes) // 2
        o, n = classes[:half], classes[half:]
        return {"basis": "halves", "old": o.count(cur_class) / len(o), "new": n.count(cur_class) / len(n),
                "old_n": o.count(cur_class), "new_n": n.count(cur_class)}
    return None


def _fmt_date(d) -> str:
    return d.strftime("%d %b %Y").lstrip("0")


def _nice_ts(ts: str | None) -> str:
    d = _parse_ts(ts or "")
    return d.strftime("%d %b %Y, %H:%M").lstrip("0") if d else (ts or "this session")


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _range_text(a: dict) -> str:
    d1, d2 = _parse_ts(a["first_ts"] or ""), _parse_ts(a["last_ts"] or "")
    if d1 and d2:
        return _fmt_date(d1) if d1.date() == d2.date() else f"{_fmt_date(d1)} to {_fmt_date(d2)}"
    return "this session"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def history_markdown(records: list[dict], predicted_display: str) -> str:
    """
    Plain-language Maintenance History for the predicted fault, laid out as tables.
    Built only from recorded rows; same log in, same text out.
    """
    a = analyze(records, predicted_display)
    if not a["total"] and a["excluded"]:
        return ("### Maintenance History\n\nAll " + str(a["excluded"]) + " recorded prediction(s) came from inputs "
                "that are far outside the data the models were trained on, so they are left out as unreliable. "
                "There is no trustworthy history yet. Run predictions on realistic inputs (for example rows "
                "from the test set).")
    if not a["total"]:
        return ("### Maintenance History\n\nNo predictions have been recorded yet, so there is nothing to "
                "analyse. Every prediction is saved automatically; repeating faults will show up here as the "
                "record grows.")

    cur_class, cur_family = a["cur_class"], a["cur_family"]
    total, cnt = a["total"], a["cur_count"]
    L: list[str] = [f"### Maintenance History: {predicted_display}", ""]
    L.append(f"_Based on {total} recorded prediction(s) from {_range_text(a)}. "
             "This summary stays the same for this fault until the recorded log is cleared._")

    # ---- In one line ---------------------------------------------------------
    if cur_class == "Normal":
        one = (f"The bearing was predicted **healthy** in {cnt} of {total} recorded readings "
               f"({_pct(cnt / total)}).")
    elif cnt <= 1:
        one = (f"**{predicted_display}** has been predicted **once** in {total} recorded readings. "
               "This is the first time, so there is no repeat pattern yet.")
    else:
        one = (f"**{predicted_display}** has been predicted **{cnt} times** in {total} recorded readings "
               f"({_pct(cnt / total)}). It is a **repeating fault**.")
        if a["cur_streak"] >= 2:
            one += f" It showed up in the last **{a['cur_streak']} readings in a row**, so it is persisting."
    L += ["", "**In one line:** " + one]

    # ---- 1. Summary table ----------------------------------------------------
    rank = a["cur_rank"]
    rec = RECOMMENDATIONS.get(cur_family or "", {})
    models = ", ".join(f"{k} ({v})" for k, v in sorted(a["cur_models"].items(), key=lambda kv: -kv[1])) or "n/a"
    if cur_class == "Normal":
        repeat = "Not applicable (healthy reading)"
    elif cnt >= 2:
        repeat = "Yes"
    else:
        repeat = "No, first time"
    L += ["", "#### 1. This fault at a glance", "", "| Question | Answer |", "|---|---|",
          f"| Predicted fault | {predicted_display} |",
          f"| Times predicted | {cnt} out of {total} readings ({_pct(cnt / total)}) |",
          f"| Is it repeating? | {repeat} |"]
    if cur_class != "Normal":
        L.append(f"| Readings in a row (latest) | {a['cur_streak']} |")
        if rank:
            L.append(f"| Rank among recorded faults | {rank[0]} of {rank[1]} |")
    L.append(f"| First recorded | {_nice_ts(a['cur_first'])} |")
    L.append(f"| Last recorded | {_nice_ts(a['cur_last'])} |")
    L.append(f"| Risk level | {rec.get('risk', 'n/a')} |")
    L.append(f"| Predicted by | {models} |")
    if cur_family and cur_family != "Normal" and a["cur_family_count"] > cnt:
        L.append(f"| Whole {cur_family} family | {a['cur_family_count']} readings (all defect sizes) |")

    # ---- 2. Weekly frequency ---------------------------------------------------
    w = a["weekly"]
    L += ["", "#### 2. How often per week"]
    if cur_class == "Normal":
        L += ["", "Not needed for a healthy reading."]
    elif not w:
        L += ["", "Weekly counts are not available because these records have no dates (this happens when the "
                  "saved log cannot be read and only this session's table is used)."]
    else:
        L += ["", "| Week starting (Monday) | Times this fault was predicted | All readings that week | Share |",
              "|---|---|---|---|"]
        weeks = w["weeks"]
        shown = weeks[-8:]
        for ws, c, t in shown:
            tag = " (latest week)" if ws == weeks[-1][0] else ""
            share = _pct(c / t) if t else "-"
            L.append(f"| {_fmt_date(ws)}{tag} | {c} | {t} | {share} |")
        if len(weeks) > len(shown):
            L.append(f"| Earlier weeks ({len(weeks) - len(shown)}) | "
                     f"{sum(c for _, c, _ in weeks[:-8])} | {sum(t for _, _, t in weeks[:-8])} | |")
        L.append("")
        days = w["span_days"]
        if w["avg_per_week"] is not None:
            L.append(f"**Average:** about **{w['avg_per_week']:.1f} times per week** "
                     f"({w['cur_total']} times over {days} days, {_fmt_date(w['first'])} to {_fmt_date(w['last'])}).")
        else:
            day_word = "day" if days == 1 else "days"
            L.append(f"**So far:** {_times(w['cur_total'])} in {days} {day_word} "
                     f"(about {w['avg_per_day']:.1f} per day). There is less than one week of records, "
                     "so a true weekly rate cannot be given yet; the number above is a count, not a forecast.")
        if w["batch_max"] >= 5:
            L.append(f"_Note: {w['batch_max']} readings were saved in the same second (a batch upload), "
                     "so the counts include those uploaded rows._")
        if w["undated"]:
            L.append(f"_{w['undated']} reading(s) had no date and are not counted in the weekly table._")

    # ---- 3. Other recorded faults -------------------------------------------------
    L += ["", "#### 3. All faults recorded so far"]
    if not a["class_counts"]:
        L += ["", "No faults have been recorded. Every reading so far was healthy."]
    else:
        L += ["", "| Rank | Fault | Times recorded | Share of all readings | Risk | Last seen |",
              "|---|---|---|---|---|---|"]
        for i, x in enumerate(a["class_counts"][:6], 1):
            mark = " (current)" if x["class"] == cur_class else ""
            L.append(f"| {i} | {x['display']}{mark} | {x['count']} | {_pct(x['share'])} | "
                     f"{x['risk']} | {_nice_ts(x['last_seen'])} |")
        if len(a["class_counts"]) > 6:
            L.append(f"\n_{len(a['class_counts']) - 6} less frequent fault type(s) not shown._")
        L.append(f"\nHealthy readings: {a['normal_total']}. Fault readings: {a['fault_total']}."
                 + (f" {a['excluded']} reading(s) left out because the input was unreliable." if a["excluded"] else ""))

    # ---- 4. What the pattern means ---------------------------------------------------
    F: list[str] = []
    if not a["enough"]:
        F.append(f"Only {total} record(s) so far. At least {MIN_RECORDS_FOR_PATTERN} are needed before a "
                 "pattern means much.")
    else:
        t = a["cur_trend"]
        if t:
            basis = ("comparing the last two weeks" if t["basis"] == "weeks"
                     else "comparing the older half of the record with the newer half")
            if t["new"] > t["old"] + 0.10:
                F.append(f"**Getting more frequent:** {basis}, this fault went from {_pct(t['old'])} to "
                         f"{_pct(t['new'])} of readings.")
            elif t["new"] < t["old"] - 0.10:
                F.append(f"**Becoming less frequent:** {basis}, this fault went from {_pct(t['old'])} to "
                         f"{_pct(t['new'])} of readings.")
            else:
                F.append(f"**Steady:** {basis}, this fault stayed at about {_pct(t['new'])} of readings.")
        elif cur_class != "Normal":
            F.append("Not enough spread over time yet to say whether this fault is getting more or less common.")
        if a["progression"]:
            kind, first, last = a["progression"]
            if kind == "worsening":
                F.append(f"**Getting worse:** {cur_family} readings moved from {SEVERITY_MAP[first].lower()} "
                         f"(0.{first}in) to {SEVERITY_MAP[last].lower()} (0.{last}in), so the damage looks "
                         "like it is progressing.")
            elif kind == "easing":
                F.append(f"{cur_family} readings moved from {SEVERITY_MAP[first].lower()} to "
                         f"{SEVERITY_MAP[last].lower()}; severity is not rising in this record.")
        if len(a["cur_models"]) > 1:
            F.append("More than one model predicted this fault, which makes it more likely to be real.")
        crit, high = a["risk_counts"].get("Critical", 0), a["risk_counts"].get("High", 0)
        if crit or high:
            bits = [f"{n} {lbl}-risk" for n, lbl in ((crit, "Critical"), (high, "High")) if n]
            F.append("The record also holds " + " and ".join(bits) + " reading(s) across all faults.")
    if F:
        L += ["", "#### 4. What the pattern means", ""] + [f"- {f}" for f in F]

    # ---- 5. Actions ------------------------------------------------------------------
    actions = list(rec.get("actions", []))
    if cur_class and cur_class != "Normal" and cnt >= 3 and not any("root-cause" in x.lower() for x in actions):
        actions.append("Repeat fault: open a root-cause analysis (check alignment, lubrication, mounting and load)")
    if actions:
        L += ["", f"#### 5. Recommended actions ({rec.get('risk', 'n/a')} risk)", ""] + [f"- {x}" for x in actions]

    L += ["", "_Built from the dashboard's own prediction log. These are model predictions, not confirmed work "
              "orders, so a person should always review them before acting._"]
    return "\n".join(L)


def chart_frame(records: list[dict]):
    """Rows for a 'times recorded' bar chart: (display name, count). Faults only."""
    a = analyze(records)
    return [(x["display"], x["count"]) for x in a["class_counts"]]
