"""Bounded lexical numeric membership, shared by source passages and draft answers.

English cardinals through millions and decimals with digit words after ``point``
are supported. This does not compare units, subjects, or the meaning of a claim.
Unsupported numeric-looking expressions are reported rather than split into values.
"""
import re
import unicodedata
from decimal import Decimal, InvalidOperation

SMALL = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                 "sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
SCALES = {"million": 1_000_000, "thousand": 1000}
WORD = "(?:" + "|".join(sorted(SMALL.keys() | TENS.keys() | SCALES.keys() | {"hundred", "point"},
                              key=len, reverse=True)) + ")"
# Consume recognized but unsupported suffixes too: "one billion" or "one fifth"
# must not silently authorize the component "one".
UNSUPPORTED_WORD = r"(?:billion|trillion|half|halves|quarters?|thirds?|fourths?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)"
WORD_SPAN = re.compile(r"\b(?:(?:minus|negative|plus)\s+)?" + WORD
                       + r"(?:[\s-]+(?:and|a(?=\s+" + UNSUPPORTED_WORD + r"\b)|" + WORD + "|"
                       + UNSUPPORTED_WORD + r"|\d+(?:[.,]\d+)*))*(?:\b|(?<=\d)(?=[a-z]))", re.I)
DIGIT_SPAN = re.compile(r"(?<![\w.])[+-]?(?:\d[\d,]*(?:\.\d+)*|\.\d+)"
                        r"(?:[eE][+-]?[\d.]*)?(?:(?=(?:mm|cm|km|m)[23](?!\w))|(?![\w.]*\d))", re.I)
LITERAL = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")
CLAUSE = re.compile(r"\d+(?:\.\d+){2,}")
PRONOUN_ONE = re.compile(r"\s+(?:can|could|may|might|must|should|shall|would|will)\b", re.I)


def _canonical(text):
    """An exact decimal string, without float conversion or context rounding."""
    value = Decimal(text.replace(",", ""))
    sign, digits, exponent = value.as_tuple()
    if not value.is_finite():
        raise ValueError("non-finite number")
    digits = "".join(map(str, digits)).lstrip("0") or "0"
    if digits == "0":
        return "0"
    while digits.endswith("0"):
        digits, exponent = digits[:-1], exponent + 1
    point = len(digits) + exponent
    # Avoid expanding an adversarial exponent into an enormous output string.
    if abs(point) > 1000 or len(digits) > 1000:
        raise ValueError("numeric expression exceeds supported length")
    if point <= 0:
        result = "0." + "0" * -point + digits
    elif point >= len(digits):
        result = digits + "0" * (point - len(digits))
    else:
        result = digits[:point] + "." + digits[point:]
    return ("-" if sign else "") + result


def _under_hundred(words):
    if len(words) == 1:
        return SMALL.get(words[0], TENS.get(words[0]))
    if len(words) == 2 and words[0] in TENS and 1 <= SMALL.get(words[1], -1) <= 9:
        return TENS[words[0]] + SMALL[words[1]]
    return None


def _under_thousand(words):
    if "hundred" not in words:
        return _under_hundred(words)
    if words.count("hundred") != 1:
        return None
    pos = words.index("hundred")
    head = 1 if pos == 0 else SMALL.get(words[0]) if pos == 1 else None
    if head is None or not 1 <= head <= 9:
        return None
    tail = words[pos + 1:]
    if tail[:1] == ["and"]:
        tail = tail[1:]
        if not tail:
            return None
    low = _under_hundred(tail) if tail else 0
    return head * 100 + low if low is not None else None


def _integer(words):
    total, start, previous_scale = 0, 0, 1_000_000_000
    for pos, word in enumerate(words):
        if word not in SCALES:
            continue
        scale = SCALES[word]
        part = words[start:pos]
        group = _under_thousand(part) if part else 1 if start == 0 else None
        if scale >= previous_scale or group is None or group == 0:
            return None
        total += group * scale
        start, previous_scale = pos + 1, scale
    tail = words[start:]
    if start and tail[:1] == ["and"]:
        tail = tail[1:]
        if not tail:
            return None
    low = _under_thousand(tail) if tail else 0 if start else None
    return total + low if low is not None else None


def _english(words):
    sign = ""
    if words[:1] in (["minus"], ["negative"], ["plus"]):
        sign = "-" if words[0] != "plus" else ""
        words = words[1:]
    if words.count("point") == 1:
        pos = words.index("point")
        whole = _integer(words[:pos]) if pos else 0
        fractional = words[pos + 1:]
        if whole is None or not fractional or any(w not in SMALL or SMALL[w] > 9 for w in fractional):
            return None
        return _canonical(sign + str(whole) + "." + "".join(str(SMALL[w]) for w in fractional))
    whole = _integer(words)
    return _canonical(sign + str(whole)) if whole is not None else None


def scan_numbers(text):
    """Return (canonical values, unsupported expressions), without borrowing components.

    A small pronoun rule excludes ``one should/can/...``; this is not a complete
    English parser. Fractions, ordinals and informal number spellings are outside
    the supported grammar. Header years, pages and complete clause IDs remain values.
    """
    text = unicodedata.normalize("NFKC", text).replace("−", "-").replace("‐", "-")
    text = re.sub(r"\bcl(?:ause)?\.(?=\d)", "clause ", text, flags=re.I)
    values, unsupported, word_spans = set(), [], []
    for match in WORD_SPAN.finditer(text):
        words = re.split(r"[\s-]+", match[0].lower())
        if words[-1:] == ["and"]:
            words = words[:-1]
        if words == ["point"] or (words == ["one"] and PRONOUN_ONE.match(text[match.end():])):
            continue
        word_spans.append(match.span())
        value = _english(words)
        if value is not None:
            values.add(value)
        elif "and" in words and not (set(words) & (SCALES.keys() | {"hundred", "point"})):
            # An ordinary list, "one and two", is not the compound number three.
            parts = re.split(r"\band\b", match[0].lower())
            parsed = [_english(re.split(r"[\s-]+", p.strip())) for p in parts]
            if all(v is not None for v in parsed):
                values.update(parsed)
            else:
                unsupported.append(match[0])
        else:
            unsupported.append(match[0])
    for match in DIGIT_SPAN.finditer(text):
        if any(start <= match.start() < end for start, end in word_spans):
            continue
        literal = match[0].rstrip(",.")
        if CLAUSE.fullmatch(literal):
            values.add(literal)
            continue
        try:
            if not LITERAL.fullmatch(literal):
                raise ValueError("unsupported numeric expression")
            values.add(_canonical(literal))
        except (ValueError, InvalidOperation):
            unsupported.append(literal)
    return values, unsupported


def numbers(text):
    """The recognized complete numeric values in text."""
    return scan_numbers(text)[0]
