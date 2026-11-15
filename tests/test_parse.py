"""[C] parser: every status, plus messy-but-legal real-world formats."""

import pytest

from llm_judge.parse import parse_output, parse_record, parser_hash


@pytest.mark.parametrize("text,verdict,conf", [
    ('{"verdict": "correct", "confidence": 0.85}', "correct", 0.85),
    ('{"verdict": "Incorrect", "confidence": 1}', "incorrect", 1.0),
    ('```json\n{"verdict": "correct", "confidence": 0.7}\n```', "correct", 0.7),
    ('Sure! Here is my judgment:\n{"verdict": "incorrect", "confidence": 0.62}\nThanks.', "incorrect", 0.62),
    ('{"verdict": "correct", "confidence": "0.9"}', "correct", 0.9),
    ('{"Verdict": " correct ", "Confidence": 0.5}', "correct", 0.5),
    ('<think>maybe {"verdict": "incorrect", "confidence": 0.1}</think>{"verdict": "correct", "confidence": 0.8}', "correct", 0.8),
    ('reasoning ... </think>\n{"verdict": "incorrect", "confidence": 0.95}', "incorrect", 0.95),
    ('{"note": "x"} then {"verdict": "correct", "confidence": 0.6}', "correct", 0.6),
    ('{"verdict": "correct", "confidence": 0.0}', "correct", 0.0),
    ('{"verdict": "correct", "confidence": 0.9, "reason": "brace } inside string"}', "correct", 0.9),
])
def test_valid(text, verdict, conf):
    j = parse_output(text, "stop", "binary")
    assert j.status == "valid", j.error
    assert j.verdict == verdict
    assert j.confidence == pytest.approx(conf)


def test_last_verdict_object_wins():
    j = parse_output('{"verdict": "correct", "confidence": 0.3} {"verdict": "incorrect", "confidence": 0.9}', "stop", "binary")
    assert (j.verdict, j.confidence) == ("incorrect", 0.9)


@pytest.mark.parametrize("text", [
    "The answer is correct.",
    '{"verdict": "partially correct", "confidence": 0.7}',
    '{"verdict": true, "confidence": 0.7}',
    '{"confidence": 0.7}',
    '{"verdict": "correct", "confidence": 0.7',  # unbalanced
    "",
])
def test_invalid_format(text):
    assert parse_output(text, "stop", "binary").status == "invalid_format"


def test_abstain_is_illegal_in_binary_mode():
    assert parse_output('{"verdict": "abstain", "confidence": 0.5}', "stop", "binary").status == "invalid_format"


@pytest.mark.parametrize("text", [
    '{"verdict": "correct"}',
    '{"verdict": "correct", "confidence": 1.5}',
    '{"verdict": "correct", "confidence": -0.1}',
    '{"verdict": "correct", "confidence": 85}',
    '{"verdict": "correct", "confidence": "high"}',
    '{"verdict": "correct", "confidence": true}',
    '{"verdict": "correct", "confidence": NaN}',
    '{"verdict": "correct", "confidence": null}',
])
def test_invalid_conf(text):
    j = parse_output(text, "stop", "binary")
    assert j.status == "invalid_conf"
    assert j.confidence is None


def test_truncated_takes_precedence():
    assert parse_output('{"verdict": "correct", "confidence": 0.9}', "length", "binary").status == "truncated"
    assert parse_output('{"verdict": "cor', "length", "binary").status == "truncated"


def test_exec_failure():
    assert parse_output(None, None, "binary").status == "exec_failure"
    assert parse_record({"status": "exec_failure", "text": None}, "binary").status == "exec_failure"


def test_ternary_abstain():
    j = parse_output('{"verdict": "abstain", "confidence": 0.4}', "stop", "ternary")
    assert j.status == "abstain" and j.verdict == "abstain" and j.confidence == 0.4
    # confidence is optional for an abstention
    assert parse_output('{"verdict": "Abstain"}', "stop", "ternary").status == "abstain"
    # ternary still accepts binary verdicts
    assert parse_output('{"verdict": "correct", "confidence": 0.8}', "stop", "ternary").status == "valid"


def test_verdict_bool():
    assert parse_output('{"verdict": "correct", "confidence": 0.8}', "stop", "binary").verdict_bool is True
    assert parse_output('{"verdict": "incorrect", "confidence": 0.8}', "stop", "binary").verdict_bool is False


def test_bad_output_mode():
    with pytest.raises(ValueError):
        parse_output("{}", "stop", "quaternary")


def test_parser_hash_is_stable():
    assert parser_hash() == parser_hash()
    assert len(parser_hash()) == 16
