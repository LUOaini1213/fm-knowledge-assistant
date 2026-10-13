"""Conservative, English-only quantity/citation binding; not semantic entailment.

We accept a bounded grammar with explicit subjects, known unit aliases and exact
comparison operators. No unit conversions, inferred subjects or arbitrary synonyms
are supported. Unsupported/ambiguous generated quantities fail closed; extractive
answers do not use this module. Source metadata remains useful for reference-only
claims, but cannot authorize a physical quantity.
"""
import re
import unicodedata
from dataclasses import dataclass

from .numeric import DIGIT_SPAN, WORD_SPAN, PRONOUN_ONE, scan_numbers

CITATIONS = re.compile(r"\[\d+\](?:\s*\[\d+\])*")
SENTENCE = re.compile(r"[.!?;](?:\s+|$)|\n+")
# Do not split a number spelling/range or the second bound of a shared subject.
COORD = re.compile(r"(?:,\s*|\s+)(?:and|but|while|whereas)\s+(?!\d|at\b|not\b|no\b|"
                   r"less\b|more\b|greater\b|lower\b|higher\b|one\b|two\b|three\b|four\b|"
                   r"five\b|six\b|seven\b|eight\b|nine\b|ten\b|eleven\b|twelve\b|"
                   r"thirteen\b|fourteen\b|fifteen\b|sixteen\b|seventeen\b|eighteen\b|"
                   r"nineteen\b|twenty\b|thirty\b|forty\b|fifty\b|sixty\b|seventy\b|"
                   r"eighty\b|ninety\b|hundred\b|thousand\b|million\b)(?=[a-z])", re.I)

ALIASES = {
    "mm": "mm", "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m", "km": "km",
    "m2": "m2", "square metres": "m2", "square meters": "m2", "cm2": "cm2", "mm2": "mm2",
    "m3": "m3", "cubic metres": "m3", "cubic meters": "m3", "cm3": "cm3", "mm3": "mm3",
    "s": "seconds", "second": "seconds", "seconds": "seconds", "min": "minutes", "mins": "minutes",
    "minute": "minutes", "minutes": "minutes", "h": "hours", "hr": "hours", "hrs": "hours",
    "hour": "hours", "hours": "hours", "day": "days", "days": "days", "week": "weeks", "weeks": "weeks",
    "month": "months", "months": "months", "year": "years", "years": "years",
    "litre": "litres", "litres": "litres", "liter": "litres", "liters": "litres", "l/s": "l/s",
    "lux": "lux", "lx": "lux", "%": "%", "percent": "%", "°c": "°c", "degc": "°c",
    "degrees celsius": "°c", "degrees": "degrees", "degree": "degrees",
    "pa": "pa", "kpa": "kpa", "mpa": "mpa", "n": "n", "kn": "kn", "kg": "kg",
    "db(a)": "dba", "dba": "dba", "db": "db", "kw": "kw", "w": "w", "v": "v", "a": "a",
    "person": "persons", "persons": "persons", "storey": "storeys", "storeys": "storeys",
    "ach": "ach", "air changes per hour": "ach",
}
UNIT = re.compile(r"\s*(?:-\s*)?(" + "|".join(re.escape(u) for u in sorted(ALIASES, key=len, reverse=True))
                  + r")(?![a-z0-9])", re.I)
# Long alternatives first; > and >= have different meanings, as do < and <=.
OPERATORS = [
    ("ge", r"not less than|no less than|at least|a minimum of|minimum of|minimum|not lower than|>=|≥"),
    ("le", r"not more than|no more than|not greater than|no greater than|no higher than|not higher than|"
           r"shall not exceed|must not exceed|does not exceed|not exceed(?:ing)?|at most|a maximum of|maximum of|maximum|up to|<=|≤"),
    ("gt", r"more than|greater than|exceed(?:s|ing)?|above|>"),
    ("lt", r"less than|lower than|below|<"),
    ("every", r"every"),
]
OP = re.compile("|".join(f"(?P<{name}>(?<![a-z])(?:{pattern})(?![a-z]))" for name, pattern in OPERATORS), re.I)
STOP = set("the a an of for to in on at is are be been being shall must need needs have has with "
           "and or it by from as required requirement requirements occur occurs".split())
PRONOUNS = {"it", "its", "they", "their", "this", "that", "these", "those", "them"}
UNSUPPORTED = re.compile(r"[\u3400-\u9fff]|\b(?:approximately|about|around|nearly|roughly|dozen|"
                         r"twice|once|half|double|triple|respectively|unless|except)\b", re.I)
REFERENCE = re.compile(r"(?:see\s+)?(?:(?:clause|cl\.|section)\s+\d+(?:\.\d+)*"
                       r"(?:\s+on\s+page\s+\d+)?|page\s+\d+)"
                       r"(?:\s+of\s+(?:the\s+)?\d{4}\s+code)?", re.I)


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[int, ...]


def _clauses(text):
    return [p.strip(" \t\r\n.,;:!?\"“”") for sentence in SENTENCE.split(text)
            for p in COORD.split(sentence) if p.strip(" \t\r\n.,;:!?\"“”")]


def _has_content(text):
    return any(ch.isalnum() for ch in text)


def _sentences(segment, lead):
    """Sentences of a citation-free segment; only the first carries a preceding leading group."""
    sentences = [[s, ()] for s in SENTENCE.split(segment) if _has_content(s)]
    if sentences:
        sentences[0][1] = lead
    return sentences


def claims(draft):
    """Bind each citation group to exactly one adjacent sentence.

    Adjacent citations form one group. A group written directly after text in the
    same sentence ("A [1].", "A [1] and B [2].") belongs to that text. A group at a
    sentence boundary ("A. [1]", "[1] A.", "- [1] A") belongs to the uncited sentence
    just before it if there is one, otherwise to the sentence just after it. A
    group never covers a second sentence in either direction: neither earlier nor
    later sentences can borrow it. Coordination inside a statement is separated
    while retaining the statement's group.
    """
    out, start, lead = [], 0, ()

    def emit(sentences, last_refs):
        for pos, (sentence, refs) in enumerate(sentences):
            if pos == len(sentences) - 1:
                refs = tuple(dict.fromkeys(refs + last_refs))
            out.extend(Claim(c, refs) for c in _clauses(sentence))

    for match in CITATIONS.finditer(draft):
        segment = draft[start:match.start()]
        refs = tuple(dict.fromkeys(int(c) for c in re.findall(r"\d+", match[0])))
        sentences, lead = _sentences(segment, lead), ()
        tail = re.sub(r"^.*[^\W_]", "", segment, flags=re.S)  # text after the last letter/digit
        at_boundary = not sentences or bool(re.search(r"[.!?;\n]", tail))
        if not at_boundary or (sentences and not sentences[-1][1]):
            emit(sentences, refs)  # trailing: the immediately preceding (uncited) sentence
        else:
            emit(sentences, ())
            lead = refs            # leading: the next sentence only
        start = match.end()
    emit(_sentences(draft[start:], lead), ())
    return out


def _text(text):
    text = unicodedata.normalize("NFKC", text).replace("−", "-").replace("‐", "-").lower()
    # Only the header generated by ask, not arbitrary parenthesized source content.
    return re.sub(r"^\([^)]*,\s*clause\s+[^)]*,\s*page\s+[^)]*\)\s*", "", text)


def _mentions(text):
    spans = []
    for match in WORD_SPAN.finditer(text):
        raw = match[0].rstrip()
        if raw.lower() == "point" or (raw.lower() == "one" and PRONOUN_ONE.match(text[match.end():])):
            continue
        # numeric.scan_numbers permits a trailing coordinating 'and'.
        raw = re.sub(r"\s+and$", "", raw, flags=re.I)
        spans.append((match.start(), match.start() + len(raw), raw))
    spans.extend((m.start(), m.end(), m[0]) for m in DIGIT_SPAN.finditer(text)
                 if not any(a <= m.start() < b for a, b, _ in spans))
    return sorted(spans)


def _signature(text):
    """Exact value/unit/operator vector plus a conservative explicit subject signature."""
    text = _text(text)
    if any(c.isalpha() and not c.isascii() for c in text):
        return None, "unsupported claim language/subject spelling"
    spans = _mentions(text)
    if not spans:
        return None, "unsupported expression" if UNSUPPORTED.search(text) else None
    if UNSUPPORTED.search(text):
        return None, "unsupported expression"
    quantities, erase = [], []
    for start, end, raw in spans:
        values, bad = scan_numbers(raw)
        unit = UNIT.match(text[end:])
        if bad or len(values) != 1 or not unit:
            return None, "unsupported or missing unit/quantity"
        value = next(iter(values))
        unit_end = end + unit.end()
        # A comparator must immediately precede this quantity; all unparsed
        # qualifiers remain in the subject/predicate signature rather than vanish.
        preceding = text[:start].rstrip()
        matches = list(OP.finditer(preceding))
        operator = matches[-1] if matches and not preceding[matches[-1].end():].strip() else None
        comparator = operator.lastgroup if operator else "eq"
        op_start = operator.start() if operator else start
        if re.search(r"\bbetween\s*$", preceding):
            comparator, op_start = "range-low", re.search(r"\bbetween\s*$", preceding).start()
        elif quantities and quantities[-1][2] == "range-low" and re.search(r"\band\s*$", preceding):
            comparator, op_start = "range-high", re.search(r"\band\s*$", preceding).start()
        if erase and not re.fullmatch(r"\s*(?:and\s*)?", text[erase[-1][1]:op_start]):
            return None, "ambiguous multi-quantity subject/property binding"
        quantities.append((value, ALIASES[unit[1].lower()], comparator))
        erase.append((op_start, unit_end))
    if len(quantities) > 1 and [q[2] for q in quantities] not in (["ge", "le"], ["le", "ge"], ["range-low", "range-high"]):
        return None, "unsupported multi-quantity binding"
    if any(q[2].startswith("range-") for q in quantities) and [q[2] for q in quantities] != ["range-low", "range-high"]:
        return None, "ambiguous range"
    for start, end in reversed(erase):
        text = text[:start] + " " + text[end:]
    if re.search(r"[<>=!~]", text):
        return None, "unsupported comparison expression"
    words = re.findall(r"[a-z]+", text)
    if PRONOUNS.intersection(words):
        return None, "implicit or ambiguous subject"
    # Plural equipment/measurement spelling only, not general synonym inference.
    # "A pump" uses an article; "Pump A" uses an equipment label. Do not erase
    # a post-noun label while cleaning ordinary leading/structural articles.
    kept = [w for n, w in enumerate(words) if w not in STOP or
            (w in {"a", "an", "the"} and n and words[n - 1] not in STOP)]
    subject = tuple(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in kept)
    # ask prepends a heading: "Exit doors Exit doors shall ...". Remove only
    # an exact adjacent repeated prefix, preserving the order of all other words.
    while True:
        size = next((s for s in range(len(subject) // 2, 0, -1) if subject[:s] == subject[s:2 * s]), None)
        if size is None:
            break
        subject = subject[size:]
    if not subject:
        return None, "missing explicit subject"
    return (subject, tuple(quantities)), None


def verify_claims(draft, passages):
    problems = []
    statements = claims(draft)
    if not statements:
        return ["no claim to verify"]
    for n, claim in enumerate(statements, 1):
        refs = [c for c in claim.citations if 1 <= c <= len(passages)]
        if not refs:
            problems.append(f"claim {n}: no valid local citation")
            continue
        values, unsupported = scan_numbers(claim.text)
        local_numbers = set().union(*(scan_numbers(passages[c - 1])[0] for c in refs))
        missing = sorted(values - local_numbers)
        if missing:
            problems.append(f"claim {n}: numbers absent from its local citations: {missing}")
        if unsupported:
            problems.append(f"claim {n}: unsupported numeric expression")
        # Only a complete metadata-reference grammar is exempt from physical
        # unit/subject parsing. Its numbers must still belong to this local cite.
        if REFERENCE.fullmatch(_text(claim.text)):
            continue
        signature, error = _signature(claim.text)
        if error:
            problems.append(f"claim {n}: {error}")
        elif signature is not None:
            candidates = [_signature(piece)[0] for c in refs for piece in _clauses(_text(passages[c - 1]))]
            if signature not in candidates:
                problems.append(f"claim {n}: quantity value/unit/comparator/subject not supported by its local citations")
    return problems
