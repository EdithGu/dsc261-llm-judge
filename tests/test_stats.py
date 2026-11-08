"""[G] Holm, McNemar exact, Wilson, stratified cluster bootstrap."""

import numpy as np
import pytest

from llm_judge import stats as S


def test_holm_known_values():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d .005*4=.02, a .01*3=.03, c .03*2=.06, b .04*1=.04 -> monotone .06
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})


def test_holm_none_is_p_one_and_caps():
    adj = S.holm({"a": 0.2, "b": None, "c": 0.6})
    assert adj["b"] == 1.0 and max(adj.values()) <= 1.0
    assert adj["a"] == pytest.approx(0.6)


def test_mcnemar_exact():
    assert S.mcnemar_exact(0, 0) == 1.0
    # b=1, c=9: 2 * P(X<=1 | n=10, .5) = 2 * 11/1024
    assert S.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert S.mcnemar_exact(5, 5) == 1.0
    assert S.mcnemar_exact(3, 7) == S.mcnemar_exact(7, 3)


def test_wilson():
    lo, hi = S.wilson_ci(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3) and hi == pytest.approx(0.1744, abs=1e-3)
    assert S.wilson_ci(0, 0) == (None, None)


def test_bootstrap_strata_preserved():
    n = 140
    ids = [f"c{i}" for i in range(n)]
    labels = [i % 2 == 0 for i in range(n)]
    I = S.bootstrap_indices(ids, ids, labels, B=200, seed=1)
    assert I.shape == (200, n)
    lab = np.array(labels)
    assert np.all(lab[I].sum(axis=1) == 70)  # each resample keeps 70 correct / 70 incorrect


def test_bootstrap_deterministic_and_order_independent():
    ids = [f"c{i}" for i in range(30)]
    labels = [i % 3 == 0 for i in range(30)]
    I1 = S.bootstrap_indices(ids, ids, labels, B=50, seed=5)
    I2 = S.bootstrap_indices(ids, ids, labels, B=50, seed=5)
    assert np.array_equal(I1, I2)
    I3 = S.bootstrap_indices(ids, ids, labels, B=50, seed=6)
    assert not np.array_equal(I1, I3)


def test_bootstrap_clusters_by_question():
    # two cases per question (one correct, one incorrect candidate) -> resampled together
    ids = [f"c{i}" for i in range(20)]
    qids = [f"q{i // 2}" for i in range(20)]
    labels = [i % 2 == 0 for i in range(20)]
    I = S.bootstrap_indices(ids, qids, labels, B=100, seed=0)
    q = np.array(qids)
    for b in range(10):
        drawn = q[I[b]]
        _, counts = np.unique(drawn, return_counts=True)
        assert np.all(counts % 2 == 0)


def test_bootstrap_ci_covers_truth_roughly():
    rng = np.random.default_rng(0)
    err = (rng.random(140) < 0.2).astype(float)
    ids = [f"c{i}" for i in range(140)]
    I = S.bootstrap_indices(ids, ids, [True] * 140, B=2000, seed=3)
    samples = err[I].mean(axis=1)
    lo, hi = S.percentile_ci(samples)
    assert lo < err.mean() < hi


def test_bootstrap_p_value():
    s = np.random.default_rng(0).normal(-0.1, 0.02, size=5000)
    assert S.bootstrap_p_two_sided(s) < 0.01
    s0 = np.random.default_rng(0).normal(0, 0.02, size=5000)
    assert S.bootstrap_p_two_sided(s0) > 0.2
