"""[E] policies, incl. the random-baseline analytic cross-check."""

import numpy as np
import pytest

from llm_judge import policies as P
from llm_judge.calibrate import fit_thresholds


def test_random_baseline_matches_analytic_expectation():
    rng = np.random.default_rng(0)
    pool = (rng.random(140) < 0.2).astype(float)
    for k in (10, 70, 105, 139):
        rb = P.random_baseline(pool, k, R=10_000, seed=123)
        assert rb["feasible"]
        assert rb["analytic_expected_risk"] == pytest.approx(pool.mean())
        # SE of the mean over R reps is tiny; allow 4 SE
        se = rb["sd_risk"] / np.sqrt(10_000)
        assert abs(rb["mean_risk"] - pool.mean()) < 4 * se + 1e-12


def test_random_baseline_full_pool_is_exact():
    pool = np.array([1, 0, 0, 1, 0], dtype=float)
    rb = P.random_baseline(pool, 5, R=100, seed=1)
    assert rb["mean_risk"] == pytest.approx(0.4) and rb["sd_risk"] == pytest.approx(0.0)


def test_random_baseline_seed_reproducible():
    pool = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    assert P.random_baseline(pool, 3, R=500, seed=9) == P.random_baseline(pool, 3, R=500, seed=9)


def test_random_baseline_infeasible():
    assert not P.random_baseline(np.zeros(3), 5)["feasible"]
    assert not P.random_baseline(np.zeros(3), 0)["feasible"]


def test_policy_summary_and_operational_metrics(row):
    rows = [row("a", conf=0.99, err=False), row("b", conf=0.95, err=True), row("c", conf=0.6, err=True),
            row("d", conf=0.5, err=False, label=False), row("e", status="invalid_format")]
    keep = [True, True, False, False, False]
    s = P.policy_summary(rows, keep, len(rows))
    assert s["k"] == 2 and s["coverage"] == pytest.approx(0.4)
    assert s["selective_risk"] == pytest.approx(0.5)
    assert s["referral_volume"] == 3
    assert s["referral_breakdown"] == {"abstained_valid": 2, "abstain_verdict": 0, "invalid": 1}
    # judge errors among valid: b, c ; caught by abstaining: c
    assert s["error_capture_rate"] == pytest.approx(0.5)


def test_no_abstention_keeps_only_valid(row):
    rows = [row("a"), row("b", status="truncated"), row("c", status="invalid_conf")]
    assert P.no_abstention(rows) == [True, False, False]


def test_confidence_policy_never_keeps_invalid(row):
    calib = [row(f"c{i}", conf=0.9) for i in range(20)]
    thr = fit_thresholds(calib)["0.50"]
    test = [row("t1", conf=0.99), row("t2", status="invalid_format"), row("t3", conf=0.1)]
    assert P.confidence_policy(test, thr) == [True, False, False]


def test_direct_abstention(row):
    rows = [row("a"), row("b", status="abstain", verdict="abstain", conf=0.3), row("c", status="invalid_format")]
    assert P.direct_abstention(rows) == [True, False, False]
    s = P.policy_summary(rows, P.direct_abstention(rows), 3)
    assert s["referral_breakdown"]["abstain_verdict"] == 1
