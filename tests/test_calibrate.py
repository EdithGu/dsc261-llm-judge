"""[D] thresholds: tie-breaking rule, exact k on calib, unreachable targets, write-once freezing."""

import random

import pytest

from llm_judge.calibrate import (ThresholdFreezeError, fit_thresholds, freeze_thresholds, load_frozen, ranked_valid,
                                 retained, target_k, tie_hash)


def _rows(make_row, confs, statuses=None, errs=None):
    statuses = statuses or ["valid"] * len(confs)
    errs = errs or [False] * len(confs)
    return [make_row(f"c{i:03d}", status=s, conf=c, err=e) for i, (c, s, e) in enumerate(zip(confs, statuses, errs))]


def test_target_k_rounding():
    assert [target_k(t, 60) for t in (0.5, 0.75, 0.9)] == [30, 45, 54]
    assert target_k(0.75, 10) == 8  # 7.5 rounds half up


def test_exact_k_on_calib_with_heavy_ties(row):
    # 60 cases, confidences concentrated on 0.9 / 0.95 like real LLM self-reports
    confs = [0.95] * 25 + [0.9] * 25 + [0.7] * 10
    rows = _rows(row, confs)
    thr = fit_thresholds(rows)
    for key, t in thr.items():
        kept = sum(retained(r["confidence"], r["case_id"], t) for r in rows)
        assert kept == t["k_calib"] == target_k(float(key), 60)


def test_tie_break_is_deterministic_and_order_independent(row):
    confs = [0.9] * 40 + [0.8] * 20
    rows = _rows(row, confs)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    assert fit_thresholds(rows) == fit_thresholds(shuffled)
    assert [r["case_id"] for r in ranked_valid(rows)] == [r["case_id"] for r in ranked_valid(shuffled)]


def test_tie_break_is_label_independent(row):
    confs = [0.9] * 60
    a = _rows(row, confs, errs=[False] * 60)
    b = _rows(row, confs, errs=[i % 2 == 0 for i in range(60)])
    for r in b:
        r["label"] = not r["label"]
    assert fit_thresholds(a) == fit_thresholds(b)


def test_tie_hash_depends_on_salt():
    assert tie_hash("x", "s1") != tie_hash("x", "s2")
    assert tie_hash("x", "s1") == tie_hash("x", "s1")


def test_unreachable_when_invalid_rate_too_high(row):
    # 60 cases, 10 invalid -> max coverage 50/60 = 0.833: 50% and 75% reachable, 90% not
    statuses = ["valid"] * 50 + ["invalid_format"] * 10
    rows = _rows(row, [0.8] * 60, statuses=statuses)
    thr = fit_thresholds(rows)
    assert thr["0.50"]["reachable"] and thr["0.75"]["reachable"]
    assert not thr["0.90"]["reachable"]
    assert thr["0.90"]["max_coverage_calib"] == pytest.approx(50 / 60)
    assert not retained(0.99, "anything", thr["0.90"])


def test_coverage_denominator_includes_invalid(row):
    statuses = ["valid"] * 54 + ["truncated"] * 6
    rows = _rows(row, [0.5 + i / 200 for i in range(60)], statuses=statuses)
    thr = fit_thresholds(rows)
    assert thr["0.90"]["k_calib"] == 54 and thr["0.90"]["reachable"]
    assert thr["0.90"]["calib_coverage"] == pytest.approx(0.9)


def test_apply_to_new_cases_uses_hash_cutoff_for_ties(row):
    rows = _rows(row, [0.9] * 60)
    t = fit_thresholds(rows)["0.50"]
    assert t["tau"] == 0.9
    test_ids = [f"t{i}" for i in range(1000)]
    kept = sum(retained(0.9, cid, t) for cid in test_ids)
    assert 300 < kept < 700  # roughly half of new tied cases pass the hash cutoff
    assert all(retained(0.95, cid, t) for cid in test_ids[:10])
    assert not any(retained(0.85, cid, t) for cid in test_ids[:10])


def test_freeze_is_write_once(tmp_path, row):
    path = tmp_path / "thresholds.json"
    rows = _rows(row, [0.9, 0.8, 0.7, 0.6] * 15)
    thr = fit_thresholds(rows)
    ids = [r["case_id"] for r in rows]
    e1 = freeze_thresholds(path, "d/m/E1", "abc", thr, "salt", ids)
    # identical re-calibration is a no-op
    e2 = freeze_thresholds(path, "d/m/E1", "abc", thr, "salt", ids)
    assert e1["entry_sha256"] == e2["entry_sha256"]
    assert load_frozen(path, "d/m/E1")["config_hash"] == "abc"
    # anything different is refused
    with pytest.raises(ThresholdFreezeError):
        freeze_thresholds(path, "d/m/E1", "zzz", thr, "salt", ids)
    freeze_thresholds(path, "d/m/E1", "zzz", thr, "salt", ids, force=True)
    assert load_frozen(path, "d/m/E1")["config_hash"] == "zzz"
