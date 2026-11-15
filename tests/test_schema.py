"""validate-data: fixtures pass, the broken fixture triggers every check, manifest verification."""

import json
import shutil

import pytest

from llm_judge.schema import (DataError, extract_final_number, load_cases, mc_choice, parse_number, validate_file)


@pytest.mark.parametrize("ds", ["gsm8k", "medqa"])
def test_fixtures_are_clean(fixtures, ds):
    rep = validate_file(fixtures / ds / f"cases_{ds}.jsonl", require_manifest=True)
    assert rep.ok, rep.summary()
    assert not rep.warnings, rep.summary()
    assert rep.split_counts == {"pilot": 2, "probe": 2, "calib": 3, "test": 3}


def test_broken_fixture_hits_every_check(fixtures):
    rep = validate_file(fixtures / "broken" / "cases_broken.jsonl")
    assert not rep.ok
    checks = {i.check for i in rep.errors}
    for expected in ("schema", "label", "numeric", "duplicate", "split_leak", "leakage", "probe_overlap",
                     "multiple_choice"):
        assert expected in checks, f"{expected} not reported:\n{rep.summary()}"
    msgs = "\n".join(str(i) for i in rep.errors)
    assert "missing field 'reference_answer'" in msgs
    assert "field 'label' has type str" in msgs
    assert "split 'dev'" in msgs
    assert "case_id repeated" in msgs
    assert "omitted_step" in msgs
    assert "not one of the options" in msgs


def test_bad_json_line(fixtures):
    rep = validate_file(fixtures / "broken" / "bad_json.jsonl")
    assert not rep.ok and rep.errors[0].check == "json"


def test_manifest_hash_mismatch(tmp_path, fixtures):
    d = tmp_path / "g"
    shutil.copytree(fixtures / "gsm8k", d)
    data = d / "cases_gsm8k.jsonl"
    with open(data, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n")  # whitespace-only change still changes the hash
    rep = validate_file(data)
    assert any(i.check == "manifest" and "SHA256 mismatch" in i.message for i in rep.errors)
    with pytest.raises(DataError):
        load_cases(data, verify_manifest=True)


def test_missing_manifest_is_warning_unless_required(tmp_path, fixtures):
    data = tmp_path / "cases_gsm8k.jsonl"
    shutil.copy(fixtures / "gsm8k" / "cases_gsm8k.jsonl", data)
    assert validate_file(data).ok
    assert not validate_file(data, require_manifest=True).ok
    with pytest.raises(DataError):
        load_cases(data, verify_manifest=True)


def test_manifest_split_count_mismatch(tmp_path, fixtures):
    d = tmp_path / "g"
    shutil.copytree(fixtures / "gsm8k", d)
    man = json.loads((d / "manifest.json").read_text())
    man["split_counts"]["test"] = 99
    (d / "manifest.json").write_text(json.dumps(man))
    rep = validate_file(d / "cases_gsm8k.jsonl")
    assert any(i.check == "split_counts" for i in rep.errors)


def test_synthetic_full_size_is_clean(synth_data, synth_med):
    for p in (synth_data, synth_med):
        rep = validate_file(p, require_manifest=True)
        assert rep.ok, rep.summary()
        assert rep.split_counts == {"pilot": 40, "probe": 100, "calib": 60, "test": 140}


@pytest.mark.parametrize("s,v", [("18", 18), ("1,000", 1000), ("$12.50", 12.5), ("3/4", 0.75), ("-7", -7), ("45%", 45),
                                 ("abc", None)])
def test_parse_number(s, v):
    assert parse_number(s) == v


@pytest.mark.parametrize("s,v", [("2 + 3 = 5. #### 5", 5), ("so the answer is 12 apples.", 12), ("total 3 then 9", 9),
                                 ("nothing", None), ("Answer: 1,250", 1250)])
def test_extract_final_number(s, v):
    assert extract_final_number(s) == v


def test_mc_choice():
    opts = {"A": "Gout", "B": "Pseudogout"}
    assert mc_choice("B", opts) == "B"
    assert mc_choice("(b)", opts) == "B"
    assert mc_choice("A. Gout", opts) == "A"
    assert mc_choice("pseudogout", opts) == "B"
    assert mc_choice("C", opts) is None
