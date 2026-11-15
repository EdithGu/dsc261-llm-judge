"""D2/D3: invalid outputs are 'not kept, not abstained' in the primary analysis and confidence-0 errors
ranked last in the sensitivity analysis."""

import pytest

from llm_judge import metrics as M
from llm_judge.calibrate import fit_thresholds
from llm_judge.evaluate import SENS_CONF, build_rows, sensitivity_rows
from llm_judge.policies import confidence_policy, policy_summary
from llm_judge.schema import Case


def _case(i, label=True):
    return Case(case_id=f"c{i:02d}", question_id=f"q{i:02d}", dataset_id="d", split="test", task_type="numeric",
                question="q", options=None, candidate_answer="1", reference_answer="1", label=label,
                candidate_type="correct" if label else "arithmetic")


def test_build_rows_statuses_and_missing():
    cases = [_case(0), _case(1, False), _case(2), _case(3)]
    recs = {
        "c00": {"case_id": "c00", "status": "ok", "text": '{"verdict": "correct", "confidence": 0.9}', "finish_reason": "stop"},
        "c01": {"case_id": "c01", "status": "ok", "text": '{"verdict": "correct", "confidence": 0.8}', "finish_reason": "stop"},
        "c02": {"case_id": "c02", "status": "ok", "text": "garbage", "finish_reason": "stop"},
    }
    rows = {r["case_id"]: r for r in build_rows(cases, recs, "binary")}
    assert rows["c00"]["status"] == "valid" and rows["c00"]["judge_error"] is False
    assert rows["c01"]["judge_error"] is True  # incorrect candidate judged correct
    assert rows["c02"]["status"] == "invalid_format" and rows["c02"]["judge_error"] is None
    assert rows["c03"]["status"] == "exec_failure" and rows["c03"]["missing"]


def test_primary_coverage_cap(row):
    rows = [row(f"v{i}", conf=0.5 + i / 100) for i in range(8)] + [row("x1", status="truncated"), row("x2", status="invalid_conf")]
    s = policy_summary(rows, [r["status"] == "valid" for r in rows], len(rows))
    assert s["coverage"] == pytest.approx(0.8)  # the maximum: 1 - invalid rate
    thr = fit_thresholds(rows)
    assert not thr["0.90"]["reachable"]


def test_sensitivity_rows_rank_invalid_last_as_errors(row):
    rows = [row("v1", conf=0.0, err=False), row("x1", status="invalid_format"), row("v2", conf=0.9, err=False)]
    sens = sensitivity_rows(rows)
    x = next(r for r in sens if r["case_id"] == "x1")
    assert x["status"] == "valid" and x["judge_error"] is True and x["confidence"] == SENS_CONF
    a = M.arrays(sens)
    order = [sens[i]["case_id"] for i in sorted(range(3), key=lambda i: a["rank"][i])]
    assert order == ["v2", "v1", "x1"]  # invalid after even a confidence-0 valid judgment


def test_sensitivity_makes_unreachable_target_reachable(row):
    rows = [row(f"v{i}", conf=0.9) for i in range(50)] + [row(f"x{i}", status="invalid_format") for i in range(10)]
    assert not fit_thresholds(rows)["0.90"]["reachable"]
    sens = sensitivity_rows(rows)
    thr = fit_thresholds(sens)["0.90"]
    assert thr["reachable"] and thr["tau"] == SENS_CONF
    keep = confidence_policy(sens, thr)
    assert sum(keep) == 54
    kept_rows = [r for r, k in zip(sens, keep) if k]
    assert sum(r.get("imputed_invalid", False) for r in kept_rows) == 4


def test_abstain_untouched_by_sensitivity(row):
    r = row("a", status="abstain", verdict="abstain", conf=0.4)
    assert sensitivity_rows([r])[0]["status"] == "abstain"
