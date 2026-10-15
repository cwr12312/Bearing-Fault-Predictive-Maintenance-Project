"""
utils/root_cause_engine.py
===========================
Root Cause Reasoning: a short table of possible causes plus a "What to do next" list.

How the result reacts to what is typed in the notes box
-------------------------------------------------------
* A cause the notes describe as a PROBLEM is moved up (and, if it is not one of the
  usual causes for this fault, it is ADDED to the table).
* A cause the notes say is FINE or ruled out ("grease is fine", "no dirt", "housing is tight")
  is REMOVED, together with its follow-up action.
* Causes that are equally supported are SHUFFLED every time the button is pressed
  (only when notes were typed). With no notes the usual order is kept.
* Notes that say the bearing was replaced recently add "wrong bearing or poor fitting".

Nothing is invented: every row comes from the fixed cause library below, and the
text of each row never changes, only which rows are shown and in what order.
No Streamlit import here, so it can be tested on its own.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from utils import note_parser as npz


@dataclass(frozen=True)
class Cause:
    key: str
    title: str
    why: str
    action: str
    keywords: tuple = ()          # word starts searched for in the notes
    signals: frozenset = frozenset()   # vibration signals that make it more likely
    lack_type: bool = False       # True when a shortage is the problem (grease)
    problem: str | None = None    # extra regex that always means "this is a problem"
    presence_keywords: tuple = () # second keyword group where the PRESENCE is the problem (e.g. dirt in grease)

    @property
    def pattern(self) -> re.Pattern | None:
        return re.compile(r"\b(?:" + "|".join(self.keywords) + r")") if self.keywords else None

    @property
    def presence_pattern(self) -> re.Pattern | None:
        return re.compile(r"\b(?:" + "|".join(self.presence_keywords) + r")") if self.presence_keywords else None

    @property
    def problem_re(self) -> re.Pattern | None:
        return re.compile(self.problem) if self.problem else None


_GREASE_ACTION = "Check the type and amount of grease."

# ---- usual causes per fault family (these are the rows shown when no notes are typed) ---------
_CORE: dict[str, list[Cause]] = {
    "Ball Fault": [
        Cause("ball_damage", "Damaged or chipped rolling balls",
              "A damaged ball hits the track as it spins, which makes sharp, knock-like vibration.",
              "Inspect the balls and the track for chips, pits or discolouration.",
              (), frozenset({"impulsive", "peaks"})),
        Cause("grease", "Dirty, dry or worn-out grease",
              "Dirt or old grease wears the balls and tracks faster and makes the bearing shake more.",
              _GREASE_ACTION, ("lubric", "grease", "greas", "dry", r"oil\b"), frozenset({"energy"}),
              lack_type=True, presence_keywords=("contamin", "dirt", "debris", "dust", "grit")),
        Cause("fitting_tight", "Poor fitting or too much tightening",
              "Damage during fitting, or tightening the bearing too much, can harm the balls early on.",
              "Check how the bearing was fitted and that it is not over-tightened.",
              ("preload", "over-?tight", "too tight", "overtighten", "tightened too", "fitting", "mounting damage"),
              frozenset()),
    ],
    "Inner Race Fault": [
        Cause("inner_wear", "Wear or pitting on the inner ring",
              "The inner ring turns with the shaft, so a damaged spot hits the balls again and again.",
              "Inspect the inner ring surface for pitting or flaking.",
              (), frozenset({"impulsive", "peaks"})),
        Cause("shaft_align", "Shaft out of line or out of balance",
              "An uneven shaft puts extra load on the inner ring and makes it tire out sooner.",
              "Check the shaft alignment and balance.",
              ("misalign", "unbalanc", "imbalanc", "coupling", "out of line", "bent shaft"), frozenset({"energy"})),
        Cause("shaft_fit", "Loose or over-tight fit on the shaft",
              "If the ring does not fit the shaft properly, it can rub and wear the surface early.",
              "Check the fit of the bearing on the shaft.",
              ("fretting", "creep", "looseness", "loose", "slipp", "spinning on the shaft"), frozenset(),
              problem=r"\b(not|isn't|wasn't)\s+(properly\s+|firmly\s+|tight(?:ly)?\s+)?(seated|secure|secured|fitted)\b"),
    ],
    "Outer Race Fault": [
        Cause("outer_wear", "Wear or pitting on the outer ring",
              "A damaged spot on the outer ring is hit every time a ball rolls over it.",
              "Inspect the outer ring for pitting or flaking at the load zone.",
              (), frozenset({"impulsive", "peaks"})),
        Cause("dirt", "Dirt or foreign particles inside the bearing",
              "Grit caught under the load leaves dents in the track and causes knocks.",
              "Inspect the seals and clean out or replace contaminated grease.",
              ("contamin", "debris", "dirt", "dust", "foreign", "grit", "particle", "swarf", "sand"),
              frozenset({"impulsive"})),
        Cause("grease", "Not enough grease, or the wrong grease",
              "Without a good grease film, metal rubs on metal and the track wears quickly.",
              _GREASE_ACTION, ("lubric", "grease", "greas", "dry", r"oil\b"), frozenset({"energy"}),
              lack_type=True),
        Cause("housing", "Housing out of line or loose",
              "An uneven or loose housing puts too much stress on one part of the outer ring.",
              "Check the housing bolts, fit and alignment.",
              ("housing", "loose", "misalign", "mount", "looseness", "out of line", "bolts"), frozenset(),
              problem=r"\b(not|isn't|wasn't)\s+(properly\s+|firmly\s+|tight(?:ly)?\s+)?(tight|secure|secured|seated|aligned)\b"),
    ],
}

# ---- extra causes: only added when the notes describe them as a problem -----------------------
_EXTRA: list[Cause] = [
    Cause("wrong_fit", "Wrong bearing, or the new bearing was fitted badly",
          "A bearing of the wrong size or type, or one knocked or pressed on unevenly, can wear and shake from the start.",
          "Check the part number, size and clearance of the fitted bearing, and how it was fitted.",
          ("wrong bearing", "wrong size", "wrong type", "incorrect bearing", "wrong part", "hammer", "cocked",
           "badly fitted", "poorly fitted", "improper(?:ly)? (?:install|fit|mount)", "incorrect(?:ly)? (?:install|fit|mount)",
           "installation (?:damage|error|problem)", "fitted wrong", "installed wrong", "mis-?fit")),
    Cause("overload", "Too much load or speed",
          "Running above the load or speed the bearing is built for tires out the metal and wears the track early.",
          "Check the load and speed against the bearing's rating.",
          ("overload", "over-?load", "heavy load", "high load", "excess(?:ive)? load", "over-?speed", "too fast",
           "high speed", "over capacity")),
    Cause("water", "Water, damp or rust inside the bearing",
          "Water washes away the grease and rust roughens the track, which causes knocks.",
          "Look for water or rust, and check the seals and the drainage around the bearing.",
          ("water", "moisture", "rust", "corros", "wet", "humid", "damp", "washdown", "steam", "condensation")),
    Cause("overheat", "Running too hot",
          "Heat thins and burns off the grease and softens the metal, so the bearing wears faster.",
          "Find out why the bearing runs hot: grease, load, cooling and ventilation.",
          ("overheat", "over-?heat", "running hot", "runs hot", "too hot", "high temp", "burning smell", "discolou?r")),
    Cause("electrical", "Electrical current passing through the bearing",
          "Stray current burns tiny pits in the track (fluting), which later shows up as vibration.",
          "Check the earthing and, if a drive is used, the shaft grounding.",
          ("electric", "fluting", "arcing", "vfd", "stray current", "inverter", "welding", "earthing", "grounding")),
    Cause("seal", "Worn or damaged seals letting dirt in",
          "A damaged seal lets grit and moisture in and lets grease out.",
          "Inspect the seals and replace any that are worn or damaged.",
          ("seal", "leak", "leaking", "grease coming out", "oil leak")),
    Cause("neighbour", "Vibration coming from nearby equipment, or resonance",
          "Machines nearby, or a part that rings at the same speed, can add vibration that is not caused by this bearing.",
          "Check whether nearby machines or a resonance are adding vibration, and measure with them stopped if possible.",
          ("nearby", "neighbou?ring", "adjacent", "next to", "resonan", "pump nearby", "other machine")),
    Cause("old_age", "Normal wear because the bearing is old",
          "Every bearing wears out after a long time in service, even when it is looked after well.",
          "Check how long the bearing has been in service against its expected life.",
          ("old bearing", "end of (?:its )?life", "years? old", "worn out", "never been replaced",
           "not been replaced", "original bearing", "age of the bearing", "long service")),
    Cause("align_extra", "Shaft out of line or out of balance",
          "An uneven shaft puts extra load on the bearing and makes it wear faster.",
          "Check the shaft alignment and balance.",
          ("misalign", "unbalanc", "imbalanc", "out of line", "bent shaft", "coupling")),
]


def _causes_for(family: str) -> list[Cause] | None:
    return _CORE.get(family)


def _evaluate(cause: Cause, text_clauses: list[str]) -> tuple[int, int]:
    """Returns (problem_clauses, ruled_out_clauses) for this cause."""
    pat, pres = cause.pattern, cause.presence_pattern
    if pat is None and pres is None:
        return 0, 0
    prob = ruled = 0
    for c in text_clauses:
        pols = []
        if pat is not None:
            pols.append(npz.topic_polarity(c, pat, cause.lack_type, cause.problem_re))
        if pres is not None:
            pols.append(npz.topic_polarity(c, pres, False, None))
        pols = [p for p in pols if p]
        if not pols:
            continue
        if any(p > 0 for p in pols):
            prob += 1
        else:
            ruled += 1
    return prob, ruled


def match_notes(family: str, text: str) -> dict:
    """
    Reads the notes and returns which causes they support or rule out for this fault family.
    {"supported": [titles], "ruled_out": [titles]}   (used by the Fusion engine too)
    """
    causes = list(_causes_for(family) or []) + [x for x in _EXTRA if x.key not in
                                                {c.key for c in (_causes_for(family) or [])}]
    cl = npz.clauses(text or "")
    sup, out = [], []
    for c in causes:
        pr, ru = _evaluate(c, cl)
        if pr and not ru:
            sup.append(c.title)
        elif ru and not pr:
            out.append(c.title)
    return {"supported": sup, "ruled_out": out}


def _sev_step(sev_code: str | None) -> str:
    return {
        "007": "Early stage: inspect at the next planned stop and check the vibration more often.",
        "014": "Middle stage: plan an inspection soon and have a replacement bearing ready.",
        "021": "Advanced stage: inspect quickly, plan a replacement and avoid long runs under heavy load.",
    }.get(sev_code, "Inspect the bearing by eye.")


def _signals_from_features(scaled_features, cname: str) -> set[str]:
    """Which of the three key readings are above the training average for this fault (invisible helper)."""
    if scaled_features is None:
        return set()
    try:
        from utils import reference_profiles as rp
        a = rp.assess(scaled_features, cname)
    except Exception:
        return set()
    if not a or not a.get("in_distribution"):
        return set()
    sig = set()
    for f, name in (("rms", "energy"), ("kurtosis", "impulsive"), ("crest", "peaks")):
        try:
            if a["features"][f]["value"] > 0.5:
                sig.add(name)
        except Exception:
            pass
    return sig


MAX_ROWS = 6


def build_root_cause(predicted_display: str, family: str, cname: str, notes: str = "",
                     scaled_features=None, seed: int | None = None) -> str:
    causes = _causes_for(family)
    if not causes:
        return (f"### Root Cause Analysis: {predicted_display}\n\n"
                "No fault is predicted, so there are no root causes to look for. Keep up routine monitoring.")

    rng = random.Random(seed)
    sev_code = next((c for c in ("007", "014", "021") if c in (cname or "")), None)
    notes = (notes or "").strip()
    has_notes = bool(notes)
    cl = npz.clauses(notes) if has_notes else []
    signals = _signals_from_features(scaled_features, cname)

    core_keys = {c.key for c in causes}
    pool = list(causes) + [x for x in _EXTRA if x.key not in core_keys]

    # A replacement mentioned in the notes counts as a "wrong bearing / badly fitted" clue.
    rep = npz.parse_replacement(notes) if has_notes else None
    tier = npz.replacement_tier(rep)

    scored = []   # (score, order, cause)
    ruled_out_keys = set()
    for order, c in enumerate(pool):
        is_core = c.key in core_keys
        pr, ru = _evaluate(c, cl) if has_notes else (0, 0)
        if c.key == "wrong_fit" and tier in ("recent", "early", "unknown") and not ru:
            pr += 2 if tier in ("recent", "unknown") else 1
        if ru and not pr:
            ruled_out_keys.add(c.key)
            continue                      # REMOVE: the notes say this is fine / ruled out
        if not is_core and not pr:
            continue                      # extras only appear when the notes describe them
        note_score = 3 * min(pr, 3)
        sig_score = 2 * len(c.signals & signals)
        scored.append((note_score + sig_score, order, c, is_core))

    # Rank: higher score first; equal scores keep the usual order with no notes, shuffle with notes.
    if has_notes:
        keyed = [(-s, rng.random(), o, c, core) for s, o, c, core in scored]
    else:
        keyed = [(-s, o, o, c, core) for s, o, c, core in scored]
    keyed.sort(key=lambda t: (t[0], t[1]))
    shown = [t[3] for t in keyed][:MAX_ROWS]
    score_of = {t[3].key: -t[0] for t in keyed}

    L = [f"### Root Cause Analysis: {predicted_display}", "",
         "#### Possible causes, most supported first", ""]
    if shown:
        L += ["| Possible cause | Why it could happen |", "|---|---|"]
        for c in shown:
            L.append(f"| {c.title} | {c.why} |")
    else:
        L.append("Your notes rule out all the usual causes for this fault, so none can be suggested from the "
                 "information given. Re-check the notes and inspect the bearing on site.")

    # ---- What to do next ------------------------------------------------------------
    steps = [_sev_step(sev_code)]
    for c in shown:
        if (score_of.get(c.key, 0) >= 3 or c.key not in core_keys) and c.action not in steps and c.action != _GREASE_ACTION:
            steps.append(c.action)
    grease_ruled_out = "grease" in ruled_out_keys
    grease_shown = any(c.key == "grease" for c in shown)
    if not grease_ruled_out and (grease_shown or not has_notes):
        steps.append(_GREASE_ACTION)
    steps.append("Measure the vibration again while the machine is running.")
    # keep the list short and free of repeats
    seen, final = set(), []
    for s in steps:
        if s not in seen:
            seen.add(s)
            final.append(s)
    L += ["", "#### What to do next", ""] + [f"- {s}" for s in final[:7]]
    return "\n".join(L)
