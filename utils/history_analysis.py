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
    return res


def _fmt_range(a: dict) -> str:
    if a["first_ts"] and a["last_ts"]:
        d1, d2 = a["first_ts"][:10], a["last_ts"][:10]
        return d1 if d1 == d2 else f"{d1} to {d2}"
    return "this session"


def history_markdown(records: list[dict], predicted_display: str) -> str:
    """The 'Maintenance History' write-up (without the optional pasted-notes section)."""
    a = analyze(records, predicted_display)
    if not a["total"] and a["excluded"]:
        return ("**Maintenance History**\n\nAll " + str(a["excluded"]) + " recorded prediction(s) came from inputs far outside "
                "the range of the training data, so they are excluded as unreliable and there is no trustworthy "
                "history yet. Run predictions on realistic, in-range inputs (for example rows from the test set).")
    if not a["total"]:
        return ("**Maintenance History**\n\nNo predictions have been recorded yet, so there is no "
                "history to analyse. Run a prediction first; every prediction is recorded here "
                "automatically, and repeating faults will show up as the log grows.")

    cur_class, cur_family = a["cur_class"], a["cur_family"]
    lines = [f"**Maintenance History - recorded faults ({_fmt_range(a)})**", ""]
    lines.append(f"The system has recorded **{a['total']}** prediction(s): "
                 f"**{a['fault_total']}** fault reading(s) and **{a['normal_total']}** normal reading(s).")

    if a["excluded"]:
        lines.append(f"_{a['excluded']} further prediction(s) were left out because the input was far outside "
                     "the range of the training data (unreliable)._")
    if not a["class_counts"]:
        lines += ["", "No faults have been recorded so far. Every recorded reading was Normal / Healthy."]
    else:
        lines += ["", "**Most repeating faults (ranked by number of times recorded):**", "",
                  "| Rank | Fault | Times recorded | Share of all records | Risk | Last seen |",
                  "|---|---|---|---|---|---|"]
        for i, x in enumerate(a["class_counts"][:6], 1):
            lines.append(f"| {i} | {x['display']} | {x['count']} | {x['share']*100:.0f}% | "
                         f"{x['risk']} | {x['last_seen'] or 'this session'} |")
        if len(a["class_counts"]) > 6:
            lines.append(f"\n_{len(a['class_counts']) - 6} less frequent fault type(s) not shown._")

        if len(a["family_counts"]) > 1 or a["family_counts"]:
            fam = ", ".join(f"{x['family']} ({x['count']})" for x in a["family_counts"])
            lines += ["", f"**By fault family:** {fam}."]

    # --- Current predicted fault in context ---------------------------------
    lines += ["", f"**Current predicted fault: {predicted_display}**"]
    if cur_class == "Normal":
        lines.append(f"- Recorded as Normal {a['cur_count']} time(s) out of {a['total']}.")
    elif cur_class:
        if a["cur_count"] <= 1:
            lines.append("- This is the **first time** this fault has been recorded; no repeat pattern yet.")
        else:
            rank = a["cur_rank"]
            rk = f" (rank {rank[0]} of {rank[1]} fault types)" if rank else ""
            lines.append(f"- Recorded **{a['cur_count']} times** out of {a['total']} "
                         f"({a['cur_count']/a['total']*100:.0f}%){rk} - a **repeating fault**.")
        if a["cur_streak"] >= 2:
            lines.append(f"- It appeared in the last **{a['cur_streak']} records in a row**, "
                         "so the condition is persisting rather than a one-off reading.")
        if cur_family != "Normal" and a["cur_family_count"] > a["cur_count"]:
            lines.append(f"- The wider **{cur_family}** family has been recorded "
                         f"{a['cur_family_count']} times in total (all defect sizes).")
        if len(a["cur_models"]) > 1:
            m = ", ".join(f"{k} x{v}" for k, v in sorted(a["cur_models"].items(), key=lambda kv: -kv[1]))
            lines.append(f"- Recorded by more than one model ({m}), which strengthens confidence that it is real.")

    # --- Pattern findings -----------------------------------------------------
    findings = []
    if not a["enough"]:
        findings.append(f"Only {a['total']} record(s) so far; at least {3} are needed before "
                        "patterns are meaningful.")
    else:
        if a["repeating"]:
            top = a["repeating"][0]
            findings.append(f"**{top['display']}** is the most repeated fault ({top['count']} of "
                            f"{a['fault_total']} fault readings).")
        elif a["fault_total"]:
            findings.append("No fault has repeated yet; each recorded fault is a single occurrence.")
        if a["progression"]:
            kind, first, last = a["progression"]
            fam = cur_family
            if kind == "worsening":
                findings.append(f"**Escalation:** {fam} readings moved from {SEVERITY_MAP[first].lower()} "
                                f"(0.{first}in) to {SEVERITY_MAP[last].lower()} "
                                f"(0.{last}in) over time, so the defect appears to be progressing.")
            elif kind == "easing":
                findings.append(f"{fam} readings moved from {SEVERITY_MAP[first].lower()} to "
                                f"{SEVERITY_MAP[last].lower()}; severity is not rising in this record.")
        if a["trend"]:
            old, new = a["trend"]
            if new > old + 0.10:
                findings.append(f"**Trend:** fault readings are rising ({old*100:.0f}% of the older half "
                                f"of the log vs {new*100:.0f}% of the newer half).")
            elif new < old - 0.10:
                findings.append(f"Trend: fault readings are falling ({old*100:.0f}% older half, "
                                f"{new*100:.0f}% newer half).")
            else:
                findings.append(f"Trend: fault rate is roughly stable ({old*100:.0f}% older half, "
                                f"{new*100:.0f}% newer half).")
        crit = a["risk_counts"].get("Critical", 0)
        high = a["risk_counts"].get("High", 0)
        if crit or high:
            parts = [f"{n} {lbl}" for n, lbl in ((crit, "Critical"), (high, "High")) if n]
            findings.append("Risk breakdown includes " + " and ".join(parts) + " reading(s).")
    if findings:
        lines += ["", "**Patterns found:**"] + [f"- {f}" for f in findings]

    # --- Recommended attention ---------------------------------------------
    rec = RECOMMENDATIONS.get(cur_family or "", {})
    actions = list(rec.get("actions", []))
    if cur_class and cur_class != "Normal" and a["cur_count"] >= 3 and not any("root-cause" in x.lower() for x in actions):
        actions.append("Repeat fault: open a root-cause analysis (check alignment, lubrication, mounting, load)")
    if actions:
        lines += ["", f"**Recommended maintenance actions ({rec.get('risk', 'n/a')} risk):**"]
        lines += [f"- {x}" for x in actions]

    lines += ["", "_Built from the dashboard's own prediction log. These are model predictions, not "
                  "confirmed work orders; useful as a starting point for retroactive labelling and "
                  "always subject to human review._"]
    return "\n".join(lines)


def chart_frame(records: list[dict]):
    """Rows for a 'times recorded' bar chart: (display name, count). Faults only."""
    a = analyze(records)
    return [(x["display"], x["count"]) for x in a["class_counts"]]
