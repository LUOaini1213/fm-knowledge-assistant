"""Document intelligence: pull quantitative requirements out of code clauses.

A requirement is (comparator, value, unit, subject, sentence), e.g. "exit doors shall have a clear width of not less
than 850 mm" -> (min, 850, mm, "exit doors ... clear width"). Rule-based on purpose: every extraction can be traced
to the words that produced it, and the rules are measured against a hand-labelled sample (eval/extraction_gold.jsonl).
"""
import re
from dataclasses import asdict, dataclass

UNITS = r"(mm|cm|m2|m²|m3|m³|m|km|kpa|pa|mpa|kn|n|kg|lux|lx|db\(a\)|dba|db|°c|degc|minutes?|mins?|hours?|hrs?|h|seconds?|s|" \
        r"days?|weeks?|months?|years?|%|percent|persons?|storeys?|l/s|litres?|liters?|kw|w|v|a|ach|air changes per hour)"
NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
MIN_WORDS = r"(?:not less than|no less than|at least|a minimum of|minimum of|minimum|min\.|not lower than|of at least|>=|=>|≥)"
MAX_WORDS = r"(?:not more than|no more than|not exceeding|shall not exceed|not exceed|does not exceed|up to|a maximum of|"             r"maximum of|maximum|max\.|not greater than|no higher than|not higher than|at most|less than|lower than|<=|≤)"
GT_WORDS = r"(?:more than|greater than|exceeding|exceeds|above)"   # applicability thresholds, read as lower bounds
END = r"(?![a-z0-9])"
QTY = NUM + r"\s*-?\s*" + UNITS + END
PATTERNS = [
    ("range", re.compile(r"between\s+(?:the\s+\w+\s+of\s+)?" + NUM + r"\s*" + UNITS + r"?\s+and\s+" + NUM + r"\s*" + UNITS + END, re.I)),
    ("max", re.compile(r"(?:not (?:be )?steeper than|not more than|not exceeding|maximum(?: gradient)? of)\s+1\s*:\s*(\d+(?:\.\d+)?)()", re.I)),
    ("min", re.compile(MIN_WORDS + r"\s*" + QTY, re.I)),
    ("max", re.compile(MAX_WORDS + r"\s*" + QTY, re.I)),
    # "a minimum clear width of 1800 mm", "a maximum travel distance of 60 m": a few words between the word and "of"
    ("min", re.compile(r"minimum(?:\s+[a-z-]+){1,3}\s+of\s+" + QTY, re.I)),
    ("max", re.compile(r"maximum(?:\s+[a-z-]+){1,3}\s+of\s+" + QTY, re.I)),
    ("max", re.compile(r"within\s+(?:a\s+)?(?:(?:travel\s+)?distance\s+of\s+)?" + QTY, re.I)),
    ("max", re.compile(r"\bno\b[^.;]{0,40}?\bmore than\s+" + QTY, re.I)),        # "no change in level of more than 25 mm"
    ("min", re.compile(GT_WORDS + r"\s+(?:the\s+)?" + QTY, re.I)),
    ("max", re.compile(r"(?:not be located\s+)?above\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+(storeys?)" + END, re.I)),
    ("min", re.compile(QTY.replace(END, "") + r"\s+or\s+more" + END, re.I)),
    ("max", re.compile(QTY.replace(END, "") + r"\s+or\s+less" + END, re.I)),
]
SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")
CANON = {"m²": "m2", "m³": "m3", "lx": "lux", "dba": "dB(A)", "db(a)": "dB(A)", "degc": "°C", "percent": "%",
         "min": "minutes", "mins": "minutes", "minute": "minutes", "hr": "hours", "hrs": "hours", "h": "hours",
         "hour": "hours", "second": "seconds", "s": "seconds", "day": "days", "week": "weeks", "month": "months",
         "year": "years", "person": "persons", "storey": "storeys", "litre": "litres", "liter": "litres", "liters": "litres",
         "air changes per hour": "ach"}


@dataclass
class Requirement:
    comparator: str
    value: float
    unit: str
    subject: str
    sentence: str
    value_max: float | None = None


def _num(s):
    return float(s.replace(",", ""))


def _unit(u):
    u = (u or "").lower()
    return CANON.get(u, u)


SPLIT = re.compile(r"[,;:()]|\b(?:which|that|shall|must|should|is|are|be|and|or|have|has|where|when|if)\b", re.I)
ARTICLE = re.compile(r"^(?:a|an|the|of|with|to|at|in)\s+", re.I)
MEASURE = re.compile(r"(?:width|height|length|depth|distance|level|gradient|area|rating|resistance|size|diameter|"
                     r"clearance|headroom|rate|capacity|duration|interval|period|temperature)(?: of)?$", re.I)


def _clean(phrase):
    phrase = " ".join(phrase.split())
    while ARTICLE.match(phrase):
        phrase = ARTICLE.sub("", phrase, count=1)
    return phrase.strip(" .")


def _subject(sentence, start, end, floor=0):
    """What the requirement constrains. Usually the last phrase before it, after any earlier requirement in the same
    sentence; a bare measure ("a clear width of") takes the phrase before it too ("exit doors ... clear width").
    When nothing usable precedes it ("A minimum of 2 hours fire resistance"), the words right after it."""
    parts = [_clean(x) for x in SPLIT.split(sentence[floor:start])]
    parts = [x for x in parts if x]
    if parts:
        subj = parts[-1]
        if MEASURE.search(subj) and len(subj.split()) <= 3 and len(parts) > 1:
            subj = parts[-2] + " / " + subj
        return " ".join(subj.split()[-10:])
    after = re.split(r"[,;.]|\b(?:is|are|shall|must|for|to|and)\b", sentence[end:])[0]
    return " ".join(_clean(after).split()[:5])


def extract(text):
    out = []
    for sentence in SENT.split(" ".join(text.split())):
        found = []
        for comp, pat in PATTERNS:
            for m in pat.finditer(sentence):
                if any(a < m.end() and m.start() < b for a, b, _, _ in found):
                    continue
                found.append((m.start(), m.end(), comp, m))
        found.sort()
        prev_end = 0
        for start, end, comp, m in found:
            subj = _subject(sentence, start, end, prev_end)
            if comp == "range":
                lo, u1, hi, u2 = m.groups()
                out.append(Requirement("range", _num(lo), _unit(u2 or u1), subj, sentence, _num(hi)))
            elif m.group(2) == "":  # gradient written as 1:N
                out.append(Requirement(comp, 1 / _num(m.group(1)), "gradient (1:N)", subj, sentence))
            else:
                out.append(Requirement(comp, _num(m.group(1)), _unit(m.group(2)), subj, sentence))
            prev_end = end
    return out


def to_dicts(reqs):
    return [asdict(r) for r in reqs]
