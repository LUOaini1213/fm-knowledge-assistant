"""Numeric membership uses whole quantities, not fragments of their spelling."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka.answer import numbers, verify  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("ten months", {"10"}),
    ("twelve months", {"12"}),
    ("twenty-one days", {"21"}),
    ("one hundred and five mm", {"105"}),
    ("one thousand two hundred and thirty-four mm", {"1234"}),
    ("one million two hundred thousand and five litres", {"1200005"}),
    ("one point five m", {"1.5"}),
    ("zero point zero five m", {"0.05"}),
    ("negative five degrees", {"-5"}),
    ("minus one point five degrees", {"-1.5"}),
    ("one and two hours", {"1", "2"}),
    ("1,000,000 litres", {"1000000"}),
    ("1,000.5 litres", {"1000.5"}),
    (".85 m", {"0.85"}),
    ("1.50 m", {"1.5"}),
    ("-5 degrees", {"-5"}),
    ("+5 degrees", {"5"}),
    ("1e3 mm", {"1000"}),
    ("1e-3 m", {"0.001"}),
    ("clause 2.2.7", {"2.2.7"}),
    ("Cl.2.2.7", {"2.2.7"}),
    ("850mm and 5,000 m2", {"850", "5000"}),
    ("1:12", {"1", "12"}),
    ("850-900 mm", {"850", "900"}),
    ("One should follow the cited clause.", set()),
    ("No one can infer a cleaning interval.", set()),
    ("one is required", {"1"}),
    ("one\tand\ttwo hours", {"1", "2"}),
    ("one\nand\ntwo hours", {"1", "2"}),
    ("one hundred\nand five mm", {"105"}),
    ("850m2 and 900m²", {"850", "900"}),
    ("1,000m3 and 2,000cm³", {"1000", "2000"}),
])
def test_complete_numeric_mentions(text, expected):
    assert numbers(text) == expected


@pytest.mark.parametrize("draft,source", [
    ("The interval is twelve months [1].", "The interval is 12 months."),
    ("The interval is 12 months [1].", "The interval is twelve months."),
    ("The width is one hundred and five mm [1].", "The width is 105 mm."),
    ("The width is 105 mm [1].", "The width is one hundred and five mm."),
    ("The width is twenty-one mm [1].", "The width is 21 mm."),
    ("The width is one point five m [1].", "The width is 1.5 m."),
    ("The width is 1.5 m [1].", "The width is one point five m."),
    ("The capacity is 1,000.5 litres [1].", "The capacity is 1000.5 litres."),
    ("The width is .85 m [1].", "The width is 0.85 m."),
    ("The width is 1e3 mm [1].", "The width is 1000 mm."),
    ("The temperature is negative five degrees [1].", "The temperature is -5 degrees."),
    ("See clause 2.2.7 on page 35 of the 2023 code [1].", "(2023 Code, clause 2.2.7, page 35) Door width 850mm."),
    ("One should follow the cited clause [1].", "Follow the cited clause."),
])
def test_equivalent_numeric_spellings_in_cited_sources(draft, source):
    assert verify(draft, [source]) == ([1], [])


@pytest.mark.parametrize("draft,source", [
    ("The interval is ten months [1].", "The interval is 2 months."),
    ("The width is nine hundred mm [1].", "The width is 850 mm."),
    ("The width is 1 mm [1].", "The width is one hundred and five mm."),
    ("The width is 5 mm [1].", "The width is one hundred and five mm."),
    ("The width is 1 mm [1].", "The width is twenty-one mm."),
    ("The width is 1 m [1].", "The width is one point five m."),
    ("The width is 5 m [1].", "The width is one point five m."),
    ("The capacity is 1,000,000 litres [1].", "The capacity is 1,000 litres."),
    ("The width is .85 m [1].", "The width is 5 m."),
    ("The temperature is -5 degrees [1].", "The temperature is 5 degrees."),
    ("The width is 1e3 mm [1].", "The width is 850 mm."),
    ("See clause 9.9.9 [1].", "(Code, clause 2.2.7, page 35) Door width 850mm."),
    ("The area is 900m2 [1].", "The area is 850m2."),
    ("The area is 900m² [1].", "The area is 850m²."),
])
def test_unsupported_values_cannot_hide_or_borrow_components(draft, source):
    cited, problems = verify(draft, [source])
    assert cited == [1]
    assert any("numbers not found" in p for p in problems)


@pytest.mark.parametrize("expression", ["one point twenty five", "one billion", "one fifth", "one and a half",
                                        "one point 5m", "1,00", "1e", "1e3.5"])
def test_unrecognised_numeric_expressions_are_reported(expression):
    cited, problems = verify(f"The width is {expression} mm [1].", ["The width is 1, 3, 5, 20 or 25 mm."])
    assert cited == [1]
    assert problems


def test_numeric_check_keeps_citation_scope_and_header_support():
    passages = ["(Code, clause 2.2.7, page 35) Clear width 850mm.", "The interval is ten months."]
    assert verify("The interval is ten months [1].", passages)[1]
    assert verify("The interval is ten months [2].", passages) == ([2], [])
    assert verify("Clear width 850 mm [1].", passages) == ([1], [])
    assert verify("Clear width 850 mm [0].", passages)[1]
    assert verify("Clear width 850 mm [3].", passages)[1]
    assert verify("Clear width 850 mm.", passages)[1]


def test_passage_boundaries_cannot_form_a_new_compound_number():
    assert verify("The width is 21 mm [1][2].", ["The limit is twenty", "one mm."])[1]


@pytest.mark.parametrize("source", ["one point twenty five", "one point 5", "one billion", "one fifth", "one and a half"])
def test_unsupported_source_expressions_do_not_authorize_components(source):
    assert verify("The width is 1 mm [1].", [f"The width is {source} mm."])[1]


def test_attached_unit_does_not_expose_a_component_of_an_unsupported_decimal():
    assert verify("The width is 5 m [1].", ["The width is one point 5m."])[1]


def test_check_remains_numeric_membership_not_unit_or_subject_entailment():
    # The lexical check must not be advertised as checking units or which subject a number describes.
    assert verify("The width is 12 m [1].", ["Duct cleaning occurs every 12 months."]) == ([1], [])
