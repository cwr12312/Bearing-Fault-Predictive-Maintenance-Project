"""
utils/fusion_engine.py
=======================
Multimodal Fusion result:

    1. a three-row table  (Predicted fault / Model confidence / Base risk)
    2. "Reasoning" in point form: the typed temperature and the typed notes are COMPARED with what the
       predicted fault should look like
    3. "What to do next" in point form

Randomised, but never wrong
---------------------------
Each press of the button gives a fresh wording and a fresh order of the points. What is shuffled:
  * equivalent wordings of the same fact (every variant says exactly the same thing),
  * the order of points inside a group and the order of the follow-up actions.
What is never shuffled or changed: the numbers, the agree / do-not-agree verdicts, the table,
and the first (most urgent) action. Verdicts come only from the typed values and recorded rows.
No Streamlit import here, so it can be tested on its own.
"""

from __future__ import annotations

import random
import re

from utils import note_parser as npz
from utils import root_cause_engine as rce

# --------------------------------------------------------------------------
# Fixed facts per fault class (same figures the dashboard has always used)
# --------------------------------------------------------------------------

# Expected temperature rise above the normal value (deg C): (minimum, soft maximum). None = no limit.
THERMAL_RANGE = {
    "Normal": (None, 5), "Ball_007": (None, 8), "Ball_014": (5, 15), "Ball_021": (10, None),
    "IR_007": (3, 10), "IR_014": (8, 20), "IR_021": (15, None),
    "OR_007_6": (3, 10), "OR_014_6": (10, 20), "OR_021_6": (15, None),
}
THERMAL_TEXT = {
    "Normal": "no more than about 5°C above normal",
    "Ball_007": "little or no rise (under about 5°C)",
    "Ball_014": "a slight rise of roughly 5 to 10°C",
    "Ball_021": "a clear rise of 10°C or more",
    "IR_007": "a small rise of about 5°C",
    "IR_014": "a noticeable rise of about 10°C",
    "IR_021": "a strong rise of 15°C or more",
    "OR_007_6": "a mild rise of about 5°C",
    "OR_014_6": "a clear rise of 10°C or more",
    "OR_021_6": "a large rise of 15°C or more",
}
ACOUSTIC = {
    "Normal": ("a smooth, steady hum", ()),
    "Ball_007": ("faint, irregular clicking", ("click", "tick", "irregular", "faint")),
    "Ball_014": ("intermittent clicking or rattling", ("click", "rattl", "tick", "intermittent")),
    "Ball_021": ("persistent rattling or rumbling", ("rattl", "rumbl", "rough", "grind")),
    "IR_007": ("a light, regular tapping or buzzing", ("tap", "buzz", "whin", "regular")),
    "IR_014": ("a steady, pulsing growl or whine", ("growl", "whine", "whin", "pulse", "tap", "buzz")),
    "IR_021": ("loud, harsh grinding or howling", ("grind", "howl", "loud", "harsh", "screech")),
    "OR_007_6": ("a faint rhythmic knocking or rumbling", ("knock", "rumbl", "rhythm", "thump")),
    "OR_014_6": ("a distinct, repeating knock or thump", ("knock", "thump", "rumbl", "rhythm", "repeat")),
    "OR_021_6": ("a heavy, loud pounding or roaring", ("pound", "roar", "loud", "knock", "thump", "heavy")),
}
FIRST_ACTION = {
    "Normal": "All evidence points to a healthy bearing: continue routine monitoring; no maintenance action is needed.",
    "Ball_007": "Early-stage ball defect (mild): keep running, trend the vibration weekly and plan an inspection at the next scheduled stop.",
    "Ball_014": "Moderate ball defect: schedule maintenance within the next planned window and monitor the temperature closely.",
    "Ball_021": "Severe ball defect: plan a bearing replacement soon and limit operation until it is inspected.",
    "IR_007": "Early inner-race defect (mild): inspect promptly, because inner-race faults progress faster than ball faults.",
    "IR_014": "Moderate inner-race defect: reduce load or speed and replace the bearing at the earliest opportunity.",
    "IR_021": "Severe inner-race defect: stop or heavily derate the machine and replace the bearing immediately.",
    "OR_007_6": "Early outer-race defect (mild) at the 6 o'clock load zone: check lubrication and alignment now and re-test soon.",
    "OR_014_6": "Moderate outer-race defect: notify the maintenance supervisor and prepare for a shutdown and replacement.",
    "OR_021_6": "Severe outer-race defect: shut the machine down and replace the bearing before restarting.",
}

_NEGATION = re.compile(r"\b(no|not|without|never|none|nor|free of|absence of|isn't|aren't|doesn't|don't|didn't|"
                       r"hasn't|haven't|wasn't|nothing)\b")
_ANOMALY = re.compile(r"\b(knock|grind|click|rattl|howl|whin|whine|pound|roar|buzz|thump|rumbl|growl|screech|"
                      r"squeal|squeak|hiss|tick|tap|noisy|noise|clunk|bang|scrap|vibrat)")
_LOUD = re.compile(r"\b(loud|heavy|harsh|severe|violent|screech|howl|pound|roar|deafening)")
_SMOOTH = re.compile(r"\b(smooth|quiet|silent|steady hum|smooth hum|normal hum|normal sound|sounds? normal|sounds? fine|"
                     r"sounds? ok|sounds? okay|sounds? good|nothing unusual)")
_FAMILY_WORDS = {
    "Inner Race Fault": re.compile(r"inner[\s-]*(race|ring)"),
    "Outer Race Fault": re.compile(r"outer[\s-]*(race|ring)"),
    "Ball Fault": re.compile(r"\bball\b|rolling[\s-]*element|\broller\b"),
}
_OVERDUE = re.compile(r"overdue|(not|never|hasn't been|haven't been)\s+(been\s+)?serviced|no service|missed (?:the )?service")
_RECUR = re.compile(r"\b(again|recurr\w*|repeat\w*|keeps?|same (issue|problem|fault)|third time|second time|twice|"
                    r"multiple times|history of)\b")
_HEALTHY_NOTE = re.compile(r"\b(?:running|working|operating|performing)\s+(?:well|fine|smoothly|normally|perfectly)\b|"
                           r"\bin good (?:condition|shape)\b")


def _affirmed(text: str) -> str:
    """Keeps only clauses that are not negated, so 'no knocking, smooth hum' does not count as knocking."""
    return " ".join(c for c in npz.clauses(text) if not _NEGATION.search(c))


def _pick(rng: random.Random, variants: list[str]) -> str:
    return variants[rng.randrange(len(variants))]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def build_fusion(predicted_display: str, confidence: float | None, risk: str, cname: str, family: str,
                 thermal_temp_c: float | None = None, thermal_baseline_c: float | None = None,
                 acoustic_note: str = "", maintenance_note: str = "",
                 history_summary: dict | None = None, seed: int | None = None) -> str:
    rng = random.Random(seed)
    is_normal = cname == "Normal"
    sev_code = next((c for c in ("007", "014", "021") if c in cname), None)
    sev_word = {"007": "mild", "014": "moderate", "021": "severe"}.get(sev_code, "")
    cur_family = family

    # tallies of how the extra evidence compares with the prediction
    agree = disagree = 0
    temp_pts: list[str] = []
    sound_pts: list[str] = []
    note_pts: list[str] = []
    hist_pts: list[str] = []
    actions_extra: list[str] = []
    flags: list[str] = []

    # ---------------- Temperature ----------------
    if thermal_temp_c is not None:
        if thermal_baseline_c is None:
            temp_pts.append(f"Temperature: {thermal_temp_c:.1f}°C was entered but no normal value was given, so it "
                            "cannot be judged as high or low. Enter the normal temperature to compare.")
        else:
            d = thermal_temp_c - thermal_baseline_c
            lo, hi = THERMAL_RANGE[cname]
            exp = THERMAL_TEXT[cname]
            head = f"Temperature: {thermal_temp_c:.1f}°C measured against a normal {thermal_baseline_c:.1f}°C"
            tol = 2.0   # ordinary sensor / reading noise
            sensor_checks = [
                "the sensor may not be touching the bearing housing",
                "the reading may have been taken before the machine warmed up",
                "the normal value entered may be wrong",
            ]
            rng.shuffle(sensor_checks)
            if is_normal:
                if d > hi:
                    temp_pts.append(f"{head} ({d:.1f}°C warmer). A healthy bearing should stay within about {hi}°C of "
                                    "normal, so this is hotter than the Normal prediction allows: it does not agree.")
                    disagree += 1
                    actions_extra.append("The bearing is predicted healthy but runs hot: check lubrication, load and "
                                         "cooling before trusting the healthy result.")
                elif d < -5:
                    temp_pts.append(f"{head} ({abs(d):.1f}°C cooler). That is much cooler than normal, which is unusual "
                                    f"even for a healthy bearing, so check {sensor_checks[0]} or {sensor_checks[1]}.")
                else:
                    temp_pts.append(f"{head} ({'same as normal' if abs(d) < 0.05 else f'{abs(d):.1f}°C ' + ('warmer' if d > 0 else 'cooler')}). "
                                    "A healthy bearing should stay close to normal, so this agrees with the Normal prediction.")
                    agree += 1
            elif d < -tol:
                temp_pts.append(_pick(rng, [
                    f"{head}, which is {abs(d):.1f}°C COOLER than normal. A bearing with a fault should run warmer, "
                    "not cooler, than normal, so the temperature does not support the predicted fault.",
                    f"{head}: {abs(d):.1f}°C below normal. Damage makes a bearing hotter, never cooler, so this reading "
                    "does not agree with the predicted fault.",
                ]) + f" Possible reasons: {sensor_checks[0]}; {sensor_checks[1]}; {sensor_checks[2]}.")
                disagree += 1
                actions_extra.append("Measure the temperature again at the bearing housing after the machine has run for "
                                     "at least 30 minutes, and confirm the normal value.")
            elif lo is not None and d < lo - tol:
                temp_pts.append(_pick(rng, [
                    f"{head} ({d:.1f}°C above normal). This fault ({sev_word}) should show {exp}, so the temperature is "
                    "LOWER than expected and does not fully support the predicted severity.",
                    f"{head}: only {d:.1f}°C above normal, while {exp} is expected for this fault. The reading is lower "
                    "than the fault should give, so it does not agree.",
                ]) + f" Possible reasons: the damage is less advanced than predicted; {sensor_checks[0]}; {sensor_checks[1]}.")
                disagree += 1
                actions_extra.append("Measure the temperature again at the bearing housing after the machine has run for "
                                     "at least 30 minutes, and confirm the normal value.")
            elif hi is not None and d > hi + tol:
                temp_pts.append(_pick(rng, [
                    f"{head} ({d:.1f}°C above normal). That is hotter than expected for this stage ({exp}), so it "
                    "agrees that something is wrong, and suggests the damage may be further along than predicted.",
                    f"{head}: {d:.1f}°C above normal, more than expected here ({exp}). It agrees with a fault, and "
                    "warns the damage may be more advanced.",
                ]))
                agree += 1
                flags.append("temperature")
                actions_extra.append("Check lubrication and load, and watch the temperature closely; stop the machine "
                                     "if it keeps climbing.")
            elif (lo is not None and d < lo) or (hi is not None and d > hi):
                temp_pts.append(f"{head} ({d:.1f}°C above normal). This is close to the expected range ({exp}) for this "
                                "fault, so it neither clearly supports nor contradicts the prediction.")
            else:
                temp_pts.append(_pick(rng, [
                    f"{head} ({d:.1f}°C above normal). This matches what this fault should show ({exp}), so the "
                    "temperature supports the prediction.",
                    f"{head}: {d:.1f}°C above normal, which fits what this fault should show ({exp}). It agrees.",
                ]))
                agree += 1

    # ---------------- Sound ----------------
    if (acoustic_note or "").strip():
        aff = _affirmed(acoustic_note)
        anomaly = bool(_ANOMALY.search(aff) or _LOUD.search(aff))
        # an explicit "no knocking" / "no unusual noise" is a statement that nothing abnormal is heard
        explicit_none = any(_NEGATION.search(c) and (_ANOMALY.search(c) or _LOUD.search(c) or re.search(r"\b(sound|noise)", c))
                            for c in npz.clauses(acoustic_note))
        smooth = (bool(_SMOOTH.search(npz.normalise(acoustic_note))) or explicit_none) and not anomaly
        typical, kws = ACOUSTIC[cname]
        hit = any(k in aff for k in kws)
        quote = '"' + _short(acoustic_note) + '"'
        if is_normal:
            if anomaly:
                sound_pts.append(f"Sound ({quote}): an abnormal sound was reported, but the model predicts a healthy "
                                 "bearing, so the sound does not agree.")
                disagree += 1
                actions_extra.append("An abnormal sound was reported on a bearing predicted healthy: listen at the "
                                     "housing and measure the vibration again.")
            else:
                sound_pts.append(f"Sound ({quote}): no abnormal sound, which agrees with a healthy bearing "
                                 f"(expected: {typical}).")
                agree += 1
        elif hit:
            sound_pts.append(_pick(rng, [
                f"Sound ({quote}): matches the usual sound of this fault ({typical}), so it supports the prediction.",
                f"Sound ({quote}): this fits what this fault normally sounds like ({typical}). It agrees.",
            ]))
            agree += 1
            if _LOUD.search(aff) and sev_code == "007":
                sound_pts.append("The sound is described as loud or heavy, which is a lot for a mild-stage fault: "
                                 "the damage may be further along than predicted.")
                flags.append("sound")
        elif anomaly:
            sound_pts.append(f"Sound ({quote}): an abnormal sound was reported, which fits a damaged bearing, but it is "
                             f"not the usual sound for this fault (usually {typical}). It supports that something is "
                             "wrong, not the exact fault type.")
        elif smooth:
            if sev_code in ("014", "021"):
                sound_pts.append(f"Sound ({quote}): no abnormal sound was described, but a {sev_word} fault is normally "
                                 f"audible ({typical}), so this does not agree with the prediction.")
                disagree += 1
                actions_extra.append("Listen at the bearing housing with a stethoscope or vibration pen and compare "
                                     "with the prediction.")
            else:
                sound_pts.append(f"Sound ({quote}): no abnormal sound. A mild fault is often too quiet to hear "
                                 f"({typical} at most), so this does not contradict the prediction, but it does not "
                                 "confirm it either.")
        else:
            sound_pts.append(f"Sound ({quote}): nothing about the sound of the bearing could be found in this note, "
                             "so it was not compared.")

    # ---------------- Maintenance note ----------------
    mt = (maintenance_note or "").strip()
    if mt:
        low = npz.normalise(mt)
        aff = _affirmed(mt)
        rep = npz.parse_replacement(mt)
        tier = npz.replacement_tier(rep)
        had_content = False
        if rep:
            had_content = True
            age = npz.describe_age(rep)
            if is_normal:
                note_pts.append(f"Notes: the bearing was replaced {age}. A healthy reading is what a new bearing should "
                                "give, so this agrees with the Normal prediction.")
                agree += 1
            elif tier == "recent":
                note_pts.append(_pick(rng, [
                    f"Notes: the bearing was replaced {age}, yet the model predicts a {sev_word} fault. A new bearing "
                    "should not show wear this soon, so the notes do not agree with the prediction.",
                    f"Notes: the bearing was replaced {age}. A {sev_word} fault on a bearing that new is not expected, "
                    "so this does not agree with the prediction.",
                ]))
                disagree += 1
                checks = [
                    "the right bearing was fitted (part number, size and clearance match the original)",
                    "it was fitted correctly (not hammered or cocked, shaft and housing fit, alignment, preload)",
                    "the right type and amount of grease was used",
                    "the sensor is measuring the new bearing and not a different one",
                ]
                rng.shuffle(checks)
                note_pts.append("Check whether " + "; ".join(checks[:3]) + ".")
                note_pts.append("If all of that is correct, a faulty new bearing or a wrong reading are the next things to rule out.")
                actions_extra.append("Check that the replacement bearing is the correct one: part number, size, clearance "
                                     "class and seals match the original specification.")
                actions_extra.append("Inspect how the new bearing was fitted: press-fit, shaft and housing fit, alignment "
                                     "and preload.")
                actions_extra.append("Compare today's vibration with a reading taken just after the replacement, if one exists.")
            elif tier == "early":
                note_pts.append(f"Notes: the bearing was replaced {age}. A fault this early in the life of a bearing "
                                "points more to fitting, wrong bearing or lubrication problems than to normal wear.")
                actions_extra.append("Check the fitting and the part number of the replacement bearing.")
            elif tier == "old":
                note_pts.append(f"Notes: the bearing was replaced {age}, so normal wear is a reasonable explanation for "
                                "the predicted fault. This does not contradict it.")
                agree += 1
            else:
                note_pts.append("Notes: a bearing replacement is mentioned but not when it happened. If it was recent, "
                                "a fault this soon should make you check that the right bearing was fitted correctly.")
                actions_extra.append("Confirm when the bearing was replaced; if it was recent, check the part number and fitting.")
        mentioned = [f for f, rx in _FAMILY_WORDS.items() if rx.search(aff)]
        if mentioned:
            had_content = True
            if not is_normal and cur_family in mentioned:
                note_pts.append(f"Notes: they name the same type of fault ({cur_family}) that the model predicts, so they agree.")
                agree += 1
            elif not is_normal:
                note_pts.append(f"Notes: they talk about {', '.join(mentioned)}, but the model predicts {cur_family}. "
                                "They do not agree on the fault type.")
                disagree += 1
                actions_extra.append("The notes and the model name different fault types: inspect the bearing to find out which is right.")
            else:
                note_pts.append(f"Notes: they mention {', '.join(mentioned)}, but the model predicts a healthy bearing. "
                                "They do not agree.")
                disagree += 1
        if not is_normal:
            m = rce.match_notes(cur_family, mt)
            if m["supported"]:
                had_content = True
                note_pts.append("Notes point to: " + "; ".join(m["supported"][:4]) + ". These are known causes of this kind of fault.")
            if m["ruled_out"]:
                had_content = True
                note_pts.append("Notes say these are fine, so they are not likely causes: " + "; ".join(m["ruled_out"][:4]) + ".")
        if _OVERDUE.search(low):
            had_content = True
            if is_normal:
                note_pts.append("Notes: service is overdue. Plan it even though the reading is healthy.")
            else:
                note_pts.append("Notes: service is overdue, which makes a developing fault more likely.")
                flags.append("overdue")
                actions_extra.append("Carry out the overdue service (inspection and lubrication) as part of this repair.")
        if _RECUR.search(aff):
            had_content = True
            note_pts.append("Notes: the problem has happened before or keeps coming back, which fits a real, "
                            "developing fault." if not is_normal else
                            "Notes: the problem keeps coming back, even though the reading is healthy.")
            if not is_normal:
                flags.append("repeat")
                actions_extra.append("The problem keeps repeating: look for the underlying cause, not just the bearing.")
        if any(_HEALTHY_NOTE.search(c) and not _NEGATION.search(c) for c in npz.clauses(mt)):
            if not is_normal and not rep:
                had_content = True
                note_pts.append("Notes say the machine has been running well, which does not match the predicted fault. "
                                "Treat the prediction with some care until it is checked.")
                disagree += 1
        if not had_content:
            note_pts.append("Notes: nothing in the text could be compared with the predicted fault, so it was not used.")

    # ---------------- Recorded history ----------------
    if history_summary and history_summary.get("total"):
        h = history_summary
        if h["total"] < 3:
            hist_pts.append(f"Recorded history: only {h['total']} prediction(s) so far, which is too few to judge a pattern.")
        elif is_normal:
            if h["normal_total"] >= h["fault_total"]:
                hist_pts.append(f"Recorded history: {h['normal_total']} healthy and {h['fault_total']} fault readings, "
                                "mostly healthy, which agrees.")
                agree += 1
            else:
                hist_pts.append(f"Recorded history: {h['fault_total']} fault and {h['normal_total']} healthy readings, "
                                "mostly faults, which does not agree with a healthy prediction.")
                disagree += 1
        elif h["cur_count"] >= 2:
            hist_pts.append(f"Recorded history: this fault was predicted {h['cur_count']} times out of {h['total']} readings, "
                            "so it keeps repeating, which agrees.")
            agree += 1
            if h["cur_count"] >= 3:
                flags.append("history")
        elif h["cur_family_count"] > 1:
            hist_pts.append(f"Recorded history: this exact fault is new, but other {cur_family} readings were recorded "
                            f"before ({h['cur_family_count']} in total), which partly agrees.")
            agree += 1
        else:
            hist_pts.append("Recorded history: this is the first time this fault is recorded, so there is no repeat pattern yet.")

    # ---------------- Overall ----------------
    checks = agree + disagree
    pc = lambda n: f"{n} piece of evidence" if n == 1 else f"{n} pieces of evidence"
    if checks == 0:
        overall = "Overall: none of the extra evidence could be compared with the prediction yet."
    elif disagree == 0:
        overall = (f"Overall: all {agree} comparable pieces of evidence agree with the model's prediction."
                   if agree > 1 else "Overall: the one comparable piece of evidence agrees with the model's prediction.")
    elif agree > disagree:
        overall = (f"Overall: {pc(agree)} agree{'s' if agree == 1 else ''} and {disagree} do{'es' if disagree == 1 else ''} not. "
                   "The prediction is mostly supported, but re-check the reading that does not match.")
    elif agree == disagree:
        overall = f"Overall: {agree} agree and {disagree} do not. The evidence is mixed, so check the bearing on site."
    elif agree == 0:
        overall = ("Overall: the one comparable piece of evidence does not agree with the prediction: re-check the "
                   "reading and inspect the bearing by hand." if disagree == 1 else
                   f"Overall: none of the {disagree} comparable pieces of evidence agree with the prediction: re-check "
                   "the sensors and readings and inspect the bearing by hand.")
    else:
        overall = (f"Overall: only {pc(agree)} agree{'s' if agree == 1 else ''} with the prediction while {disagree} "
                   f"do{'es' if disagree == 1 else ''} not. Most of the evidence does not match: re-check the sensors "
                   "and readings and inspect the bearing by hand.")
    if confidence is not None and confidence < 0.6:
        overall += " The model's own confidence is low, so give the other evidence extra weight."

    # ---------------- Assemble ----------------
    L = [f"### Multimodal Fusion: {predicted_display}", "",
         "| Item | Result |", "|---|---|",
         f"| Predicted fault | {predicted_display} |",
         f"| Model confidence | {confidence * 100:.0f}% |" if confidence is not None else "| Model confidence | n/a |",
         f"| Base risk | {risk} |", ""]

    groups = [g for g in (temp_pts, sound_pts, note_pts, hist_pts) if g]
    rng.shuffle(groups)                      # order of the groups changes, each group stays together
    pts: list[str] = []
    for g in groups:
        pts.extend(g)
    L += ["#### Reasoning: comparing the readings with the prediction", ""]
    if pts:
        L += [f"- {p}" for p in pts]
    else:
        L.append("- Only the vibration model has data so far. Enter a temperature, a sound note or a maintenance note "
                 "(or run more predictions so the recorded history can be used) to compare them with the prediction.")
    L.append(f"- {overall}")

    # ---------------- Recommendations ----------------
    recs = [FIRST_ACTION[cname]]
    pool = list(dict.fromkeys(actions_extra))
    if not is_normal:
        pool.append("Measure the vibration again while the machine is running and compare it with this reading.")
        if disagree > agree:
            pool.append("Re-seat the sensors and repeat the measurements, then confirm with a hands-on inspection.")
        if flags and risk != "Critical":
            pool.append("Because of the warning signs above, treat this as more urgent than the base risk alone suggests.")
    else:
        if disagree:
            pool.append("Because some evidence does not agree with the healthy prediction, inspect the bearing by hand.")
        pool.append("Keep recording readings so any change shows up early.")
    rng.shuffle(pool)
    recs += pool
    seen, final = set(), []
    for r in recs:
        if r not in seen:
            seen.add(r)
            final.append(r)
    L += ["", "#### What to do next", ""] + [f"- {r}" for r in final]
    return "\n".join(L)


def _short(text: str, limit: int = 90) -> str:
    t = " ".join(text.replace("|", "/").replace('"', "'").split())
    return t if len(t) <= limit else t[: limit - 3] + "..."
