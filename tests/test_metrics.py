"""[F] metrics: hand-computed AURC (both versions), E-AURC, Brier, reliability, curve."""

import numpy as np
import pytest

from llm_judge import metrics as M


def _a(make_row, spec):
    """spec: list of (conf, err) for valid rows or None for an invalid row."""
    rows = []
    for i, s in enumerate(spec):
        if s is None:
            rows.append(make_row(f"r{i}", status="invalid_format"))
        else:
            rows.append(make_row(f"r{i}", conf=s[0], err=s[1]))
    return rows, M.arrays(rows)


def test_aurc_hand_computed(row):
    # ordering by confidence: E(0.9) C(0.8) E(0.7) C(0.6)
    rows, a = _a(row, [(0.8, False), (0.9, True), (0.6, False), (0.7, True)])
    # risks at k=1..4: 1/1, 1/2, 2/3, 2/4
    expected = np.mean([1.0, 0.5, 2 / 3, 0.5])
    m = M.aurc_metrics(a)
    assert m["aurc"] == pytest.approx(expected)
    assert m["aurc_sens"] == pytest.approx(expected)  # no invalid -> identical
    assert m["aurc_area"] == pytest.approx(expected)
    # optimal: C C E E -> 0, 0, 1/3, 2/4
    assert m["eaurc"] == pytest.approx(expected - np.mean([0, 0, 1 / 3, 0.5]))


def test_aurc_with_invalid_primary_vs_sensitivity(row):
    rows, a = _a(row, [(0.9, False), (0.8, True), None, None])
    m = M.aurc_metrics(a)
    # primary: only 2 valid points, risks 0, 1/2 ; max coverage 0.5
    assert m["max_coverage"] == pytest.approx(0.5)
    assert m["aurc"] == pytest.approx(0.25)
    assert m["aurc_area"] == pytest.approx(0.25 * 0.5)
    # sensitivity: invalid = errors ranked last -> risks 0, 1/2, 2/3, 3/4
    assert m["aurc_sens"] == pytest.approx(np.mean([0, 0.5, 2 / 3, 0.75]))


def test_perfect_ordering_has_zero_eaurc(row):
    rows, a = _a(row, [(0.99, False), (0.9, False), (0.5, True), (0.4, True)])
    assert M.aurc_metrics(a)["eaurc"] == pytest.approx(0.0)


def test_curve(row):
    rows, a = _a(row, [(0.9, False), (0.8, True), None])
    c = M.risk_coverage_curve(a)
    assert c["coverage"] == pytest.approx([1 / 3, 2 / 3])
    assert c["risk"] == pytest.approx([0.0, 0.5])
    assert c["risk_sens"] == pytest.approx([0.0, 0.5, 2 / 3])
    assert c["max_coverage"] == pytest.approx(2 / 3)


def test_curve_point(row):
    rows, a = _a(row, [(0.9, False), (0.8, True), (0.7, True)])
    assert M.curve_point(a, 1) == 0.0
    assert M.curve_point(a, 3) == pytest.approx(2 / 3)
    assert M.curve_point(a, 4) is None


def test_batched_aurc_equals_single(row):
    rows, a = _a(row, [(0.9, False), (0.8, True), None, (0.95, True), (0.3, False)])
    rng = np.random.default_rng(0)
    I = rng.integers(0, len(rows), size=(50, len(rows)))
    idx = M.sort_by_rank(I, a["rank"])
    batch = M.aurc_batch(a["err"][idx], a["valid"][idx])
    for b in range(5):
        sub = [rows[i] for i in I[b]]
        # rebuild with unique ids but identical conf / order
        sub_rows = [dict(r, case_id=f"{r['case_id']}#{j}") for j, r in enumerate(sub)]
        sa = M.arrays(sub_rows)
        sa["rank"] = a["rank"][I[b]]  # keep the global ordering (ties among duplicates are harmless)
        assert batch[b] == pytest.approx(M.aurc_metrics(sa)["aurc"])


def test_brier_and_reliability(row):
    rows, a = _a(row, [(1.0, False), (0.8, True), (0.6, False)])
    # targets: 1, 0, 1
    assert M.brier(a) == pytest.approx(np.mean([0.0, 0.64, 0.16]))
    rel = M.reliability(a)
    assert sum(b["n"] for b in rel) == 3
    assert rel[9]["n"] == 1 and rel[9]["accuracy"] == 1.0  # conf 1.0 lands in the last bin


def test_accuracy_and_validity(row):
    rows, a = _a(row, [(0.9, False), (0.8, True), None, None])
    acc = M.accuracy(a)
    assert acc["accuracy_valid"] == pytest.approx(0.5)
    assert acc["accuracy_sens"] == pytest.approx(0.25)
    v = M.validity_table(rows)
    assert v["valid"]["n"] == 2 and v["invalid_format"]["share"] == pytest.approx(0.5) and v["n_total"] == 4


def test_bias_split(row):
    rows = [row("a", label=True, err=False), row("b", label=True, err=True),
            row("c", label=False, err=True), row("d", label=False, err=False)]
    b = M.bias_split(rows)
    assert b["false_reject_rate"] == pytest.approx(0.5)  # b: correct candidate judged incorrect
    assert b["false_accept_rate"] == pytest.approx(0.5)  # c: wrong candidate judged correct


def test_confidence_stats_mode_share(row):
    rows = [row(f"r{i}", conf=0.9) for i in range(8)] + [row("x", conf=0.5), row("y", conf=0.6)]
    cs = M.confidence_stats(rows)
    assert cs["mode"] == 0.9 and cs["mode_share"] == pytest.approx(0.8)
