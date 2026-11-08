"""Section 6.1 gates."""

from llm_judge import gates as G


def _rows(make_row, n, n_err, conf_fn=lambda i: 0.5 + (i % 50) / 100, n_invalid=0, n_trunc=0):
    rows = []
    for i in range(n):
        rows.append(make_row(f"c{i}", conf=conf_fn(i), err=i < n_err, label=i % 2 == 0))
    for i in range(n_invalid):
        rows.append(make_row(f"x{i}", status="invalid_format"))
    for i in range(n_trunc):
        rows.append(make_row(f"t{i}", status="truncated"))
    return rows


def _status(checks, prefix):
    return next(c["status"] for c in checks if c["check"].startswith(prefix))


def test_error_rate_bands(row):
    assert _status(G.gate_checks(G.run_stats(_rows(row, 100, 15))), "judge too accurate") == G.PASS
    assert _status(G.gate_checks(G.run_stats(_rows(row, 100, 8))), "judge too accurate") == G.BORDER
    assert _status(G.gate_checks(G.run_stats(_rows(row, 100, 3))), "judge too accurate") == G.FAIL
    assert _status(G.gate_checks(G.run_stats(_rows(row, 100, 45))), "judge too poor") == G.FAIL


def test_degenerate_confidence(row):
    s = G.run_stats(_rows(row, 100, 15, conf_fn=lambda i: 0.9 if i < 80 else 0.7))
    assert _status(G.gate_checks(s), "degenerate confidence") == G.FAIL


def test_one_sided_verdicts(row):
    rows = [row(f"c{i}", label=True, err=False) for i in range(90)] + [row(f"d{i}", label=False, err=False) for i in range(10)]
    assert _status(G.gate_checks(G.run_stats(rows)), "one-sided") == G.FAIL


def test_output_stability(row):
    s = G.run_stats(_rows(row, 95, 15, n_invalid=2, n_trunc=3))
    assert _status(G.gate_checks(s), "output stability E1") == G.FAIL
    assert _status(G.gate_checks(s), "output stability E2") == G.NA


def test_reproducibility(row):
    a = [row(f"c{i}", conf=0.9, err=False) for i in range(100)]
    b = [dict(r) for r in a]
    for r in b[:5]:
        r["verdict"] = "incorrect" if r["verdict"] == "correct" else "correct"
        r["confidence"] = 0.6
    ag = G.replicate_agreement(a, b)
    assert ag["verdict_agreement"] == 0.95 and ag["max_abs_conf_diff"] > 0.29
    assert _status(G.gate_checks(G.run_stats(a), replicate_agreement=ag), "reproducibility") == G.FAIL
    assert _status(G.gate_checks(G.run_stats(a), replicate_agreement=G.replicate_agreement(a, a)), "reproducibility") == G.PASS


def test_cost_gate():
    recs = [{"latency": 10.0}] * 10
    c = G.cost_estimate(recs, batch_size=10, formal_cases=200, formal_runs=10, budget_gpu_hours=0.1)
    assert c["est_total_gpu_hours"] > 0.5
    checks = G.gate_checks({"error_rate": 0.2, "error_rate_ci": (0.1, 0.3), "n_valid": 100, "verdict_max_share": 0.5,
                            "verdict_counts": {}, "conf_mode_share": 0.2, "conf_mode": 0.9, "valid_rate": 1.0,
                            "truncation_rate": 0.0, "status_counts": {}}, cost=c)
    assert _status(checks, "cost") == G.FAIL


def test_small_sample_note(row):
    checks = G.gate_checks(G.run_stats(_rows(row, 20, 2)))
    note = next(c["note"] for c in checks if c["check"].startswith("judge too accurate"))
    assert "probe split" in note


def test_markdown_and_difficulty_report(row):
    rows = _rows(row, 100, 15)
    md = G.checks_markdown(G.gate_checks(G.run_stats(rows)))
    assert "Overall" in md
    rep = G.difficulty_report(rows, "x")
    assert "false accept" in rep and "candidate_type" in rep
