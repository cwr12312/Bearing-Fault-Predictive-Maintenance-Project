"""
utils/note_parser.py
=====================
Small, deterministic text helpers shared by Root Cause, Maintenance History
and Multimodal Fusion. No randomness, no network, no Streamlit.

What it does
------------
* splits typed notes into sentences and short clauses,
* decides, for a given topic (grease, dirt, loose housing ...), whether a clause
  says the topic IS a problem or says it is fine / ruled out
  ("grease is dry"  -> problem,  "grease is fine" -> ruled out,
   "not enough grease" -> problem, "no dirt" -> ruled out),
* finds out whether (and how long ago) a bearing was replaced.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, straighten curly quotes, squash whitespace."""
    t = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(t.lower().split()) if "\n" not in t else t.lower()


_SENT_SPLIT = re.compile(r"[.;!?\n]+")
_CLAUSE_SPLIT = re.compile(r"[.;,!?\n:]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\band\b|\bwhile\b|\bwhereas\b")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(normalise(text)) if s.strip()]


def clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(normalise(text)) if c.strip()]


# --------------------------------------------------------------------------
# Topic polarity: problem / ruled out / not mentioned
# --------------------------------------------------------------------------

_NEG = re.compile(r"\b(no|not|without|never|none|nor|free of|free from|absence of|isn't|aren't|doesn't|don't|"
                  r"didn't|hasn't|haven't|wasn't|weren't|nothing|ruled out|cleared|zero)\b")
_OK = re.compile(r"\b(fine|ok|okay|good|clean|correct|fresh|adequate|plenty|tight|secure|secured|aligned|"
                 r"normal|healthy|proper|properly|within spec|in spec|all clear|satisfactory|sound)\b")

# "grease is dry", "not enough grease": here the PROBLEM is a shortage / wrong state.
_LACK_NOT_PROBLEM = re.compile(r"\b(not|isn't|wasn't|aren't|weren't|no longer|never)\s+(too\s+|very\s+)?"
                               r"(dry|low|old|stale|hardened|wrong|empty|insufficient|dirty|contaminated|"
                               r"missing|lacking|short)\b")
_LACK_PROBLEM = re.compile(r"\b(not enough|isn't enough|wasn't enough|insufficient|lack(?:ing)?(?: of)?|ran out|"
                           r"run out|running low|low on|empty|dry|old|stale|hardened|wrong|incorrect|"
                           r"under-?\s?(?:greased|lubricated|filled)|over-?\s?(?:greased|lubricated|filled)|"
                           r"too (?:much|little|thin|thick)|overfilled|missing|overdue|"
                           r"never (?:been )?(?:re)?(?:greased|lubricated)|"
                           r"not (?:been )?(?:re)?(?:greased|lubricated)|"
                           r"no (?:grease|lubricant|lubrication|oil)\b(?! (?:problem|issue|leak|trouble)))")
_NO_ISSUE = re.compile(r"\b(no|not|without|nor|zero)\s+(\w+\s+){0,2}(problem|problems|issue|issues|fault|trouble|"
                       r"concern|concerns|shortage)\b")


def topic_polarity(clause: str, pattern: re.Pattern, lack_type: bool = False,
                   problem_phrases: re.Pattern | None = None) -> int:
    """
    +1  the clause mentions the topic as a PROBLEM (or just mentions it)
    -1  the clause mentions the topic as FINE / ruled out
     0  the topic is not in this clause
    lack_type=True for topics where a shortage is the problem (grease, lubrication).
    """
    if not pattern.search(clause):
        return 0
    if problem_phrases is not None and problem_phrases.search(clause):
        return 1
    if lack_type:
        if _LACK_NOT_PROBLEM.search(clause) or _NO_ISSUE.search(clause):
            return -1
        if _LACK_PROBLEM.search(clause):
            return 1
        if _OK.search(clause) or re.search(r"\b(recently|just|newly|freshly|topped up|re-?greased|re-?lubricated|"
                                            r"enough|full|adequate)\b", clause) or \
                re.search(r"\b(?:re-?)?(?:greased|lubricated|oiled)\b", clause):
            return -1
        return 1
    if _NEG.search(clause) or _OK.search(clause):
        return -1
    return 1


# --------------------------------------------------------------------------
# Was the bearing replaced, and how long ago?
# --------------------------------------------------------------------------

_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple of": 2, "a couple of": 2,
              "few": 3, "a few": 3, "several": 4}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_REPLACE_VERB = re.compile(r"\b(replac\w*|re-?install\w*|install\w*|fitted|fit|swapp?ed|renewed|changed|put in)\b")
_PLAN_WORDS = re.compile(r"\b(plan(?:ned|ning)?|schedul\w*|due|needs? to be|need to|should be|will be|to be replaced|"
                         r"going to|recommend\w*|waiting|awaiting|pending|next (?:week|month|shutdown|stop)|soon|"
                         r"tomorrow|yet to|overdue|ordered|requested|propos\w*|consider\w*|might|may|could|if)\b")
_NEG_BEFORE_VERB = re.compile(r"\b(not|never|n't|yet to|haven't|hasn't|didn't|wasn't|weren't|without)\s+"
                              r"(\w+\s+){0,2}(replac|re-?install|install|fit|swap|renew|chang)")
_OTHER_PART = re.compile(r"\b(oil|filter|belt|seal|grease|coupling|motor|fan|sensor|gasket|pump|valve|hose|"
                         r"cable|bolt|bolts|guard|lubricant)\b")
_NEW_BEARING = re.compile(r"\b(brand[\s-]new|new|fresh|just[\s-]installed)\s+(\w+\s+)?bearing|"
                          r"bearing\s+(?:is|was|looks|looked)\s+(?:brand[\s-]new|new)\b")

_AGO = re.compile(r"\b(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                  r"a couple of|couple of|a few|few|several)\s+(day|week|month|year)s?\s+(?:ago|back|earlier)\b")


def _days_from_sentence(s: str) -> int | None:
    m = _AGO.search(s)
    if m:
        raw, unit = m.group(1), m.group(2)
        n = int(raw) if raw.isdigit() else _NUM_WORDS.get(raw, 1)
        return n * _UNIT_DAYS[unit]
    if re.search(r"\byesterday\b|\blast night\b", s):
        return 1
    if re.search(r"\btoday\b|\bthis morning\b|\bjust now\b|\bjust (?:been )?(?:replaced|installed|fitted)\b|"
                 r"\bjust\b", s):
        return 0
    if re.search(r"\bthis week\b", s):
        return 3
    if re.search(r"\blast week\b", s):
        return 7
    if re.search(r"\bearlier this month\b|\bthis month\b", s):
        return 15
    if re.search(r"\blast month\b", s):
        return 30
    if re.search(r"\blast year\b", s):
        return 365
    if re.search(r"\brecent(?:ly)?\b|\bnewly\b", s):
        return 7
    return None


def parse_replacement(text: str) -> dict | None:
    """
    Looks for 'bearing replaced/installed ...' in typed notes.
    Returns None if no completed replacement is described, otherwise
    {"days": int | None, "new": bool, "sentence": str}
    (days is None when no time was given). When several are found, the most recent one wins.
    """
    found = []
    for s in sentences(text):
        has_bearing = "bearing" in s
        verb = _REPLACE_VERB.search(s)
        newb = bool(_NEW_BEARING.search(s))
        if not (verb or newb):
            continue
        if _OTHER_PART.search(s) and not has_bearing:
            continue
        if verb and not has_bearing and not re.search(r"\b(replac|swap|renew)", s):
            continue
        if _PLAN_WORDS.search(s) or _NEG_BEFORE_VERB.search(s):
            continue
        # a bare "new"/"fit" with no bearing and no real replacement verb is not enough
        if not has_bearing and not re.search(r"\b(replac\w*|swapp?ed|renewed)\b", s):
            continue
        found.append({"days": _days_from_sentence(s), "new": newb, "sentence": s})
    if not found:
        return None
    dated = [f for f in found if f["days"] is not None]
    return min(dated, key=lambda f: f["days"]) if dated else found[0]


def replacement_tier(rep: dict | None) -> str | None:
    """'recent' (<= 30 days or described as new), 'early' (31-180 days), 'old' (> 180), 'unknown' (no time), or None."""
    if not rep:
        return None
    d = rep["days"]
    if d is None:
        return "recent" if rep["new"] else "unknown"
    if d <= 30:
        return "recent"
    if d <= 180:
        return "early"
    return "old"


def describe_age(rep: dict) -> str:
    d = rep["days"]
    if d is None:
        return "recently" if rep["new"] else "at a time that was not stated"
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    if d < 14:
        return f"about {d} days ago"
    if d < 60:
        return f"about {round(d / 7)} weeks ago"
    if d < 730:
        return f"about {round(d / 30)} months ago"
    return f"about {round(d / 365)} years ago"
