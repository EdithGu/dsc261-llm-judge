"""Section 6.1 hard gates (`pilot-report`, `screen`) and the probe difficulty report.

Gate thresholds live in GATES so a change is a one-line, reviewable diff (and must go in decision_log.md).
"""

from __future__ import annotations

from collections import Counter

from llm_judge import metrics as M
from llm_judge.stats import wilson_ci

GATES = {
    "error_rate_pass": 0.10,        # >= pass
    "error_rate_borderline": 0.06,  # [0.06, 0.10) borderline; < 0.06 fail
    "error_rate_max": 0.40,         # > fail (judge too poor)
    "verdict_share_max": 0.85,      # single verdict share > fail
    "conf_mode_share_max": 0.70,    # one confidence value > fail
    "valid_rate_min": 0.95,
    "truncation_max": 0.02,
    "repro_agreement_min": 0.98,
    "min_probe_n_for_error_gate": 100,
}

PASS, BORDER, FAIL, NA, INFO = "PASS", "BORDERLINE", "FAIL", "N/A", "INFO"


def _fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.3g}"


def run_stats(rows: list[dict]) -> dict:
    n = len(rows)
    valid = [r for r in rows if r["status"] == "valid"]
    n_err = sum(bool(r["judge_error"]) for r in valid)
    st = Counter(r["status"] for r in rows)
    verdicts = Counter(r["verdict"] for r in valid)
    cs = M.confidence_stats(rows)
    return {
        "n": n,
        "n_valid": len(valid),
        "n_errors": n_err,
        "error_rate": n_err / len(valid) if valid else None,
        "error_rate_ci": wilson_ci(n_err, len(valid)),
        "valid_rate": len(valid) / n if n else None,
        "truncation_rate": st.get("truncated", 0) / n if n else None,
        "status_counts": dict(st),
        "verdict_max_share": max(verdicts.values()) / len(valid) if valid else None,
        "verdict_counts": dict(verdicts),
        "conf_mode_share": cs.get("mode_share"),
        "conf_mode": cs.get("mode"),
        "splits": dict(Counter(r["split"] for r in rows)),
    }


def gate_checks(primary: dict, thinking: dict | None = None, replicate_agreement: dict | None = None,
                cost: dict | None = None) -> list[dict]:
    """primary: run_stats of the E1-type run; thinking: run_stats of the E2-type run (optional)."""
    g = GATES
    checks = []

    er = primary["error_rate"]
    lo, hi = primary["error_rate_ci"]
    n_valid = primary["n_valid"]
    note = f"n_valid={n_valid}, 95% CI [{_fmt(lo)}, {_fmt(hi)}]"
    if n_valid < g["min_probe_n_for_error_gate"]:
        note += f" — fewer than {g['min_probe_n_for_error_gate']} cases: use a probe split of >=100 for this gate"
    if er is None:
        st = NA
    elif er < g["error_rate_borderline"]:
        st = FAIL
    elif er < g["error_rate_pass"]:
        st = BORDER
    else:
        st = PASS
    checks.append({"check": "judge too accurate (E1 error rate >= 10%)", "status": st, "value": _fmt(er), "note": note})
    checks.append({"check": "judge too poor (E1 error rate <= 40%)",
                   "status": NA if er is None else (FAIL if er > g["error_rate_max"] else PASS),
                   "value": _fmt(er), "note": note})

    vs = primary["verdict_max_share"]
    checks.append({"check": "one-sided verdicts (max verdict share <= 85%)",
                   "status": NA if vs is None else (FAIL if vs > g["verdict_share_max"] else PASS),
                   "value": _fmt(vs), "note": str(primary["verdict_counts"])})

    ms = primary["conf_mode_share"]
    checks.append({"check": "degenerate confidence (mode share <= 70%)",
                   "status": NA if ms is None else (FAIL if ms > g["conf_mode_share_max"] else PASS),
                   "value": _fmt(ms), "note": f"mode={primary['conf_mode']}"})

    for name, s in (("E1", primary), ("E2", thinking)):
        if s is None:
            checks.append({"check": f"output stability {name} (valid >= 95%, truncated <= 2%)", "status": NA,
                           "value": "—", "note": "run not provided"})
            continue
        bad = (s["valid_rate"] or 0) < g["valid_rate_min"] or (s["truncation_rate"] or 0) > g["truncation_max"]
        checks.append({"check": f"output stability {name} (valid >= 95%, truncated <= 2%)",
                       "status": FAIL if bad else PASS,
                       "value": f"valid {_fmt(s['valid_rate'])}, truncated {_fmt(s['truncation_rate'])}",
                       "note": str(s["status_counts"])})

    if replicate_agreement is None:
        checks.append({"check": "reproducibility (verdict agreement >= 98%)", "status": NA, "value": "—",
                       "note": "no replicate run given (--replicate)"})
    else:
        ag = replicate_agreement["verdict_agreement"]
        checks.append({"check": "reproducibility (verdict agreement >= 98%)",
                       "status": NA if ag is None else (FAIL if ag < g["repro_agreement_min"] else PASS),
                       "value": _fmt(ag),
                       "note": f"n_paired={replicate_agreement['n_paired']}, "
                               f"mean |Δconf|={_fmt(replicate_agreement['mean_abs_conf_diff'], pct=False)}"})

    if cost is None:
        checks.append({"check": "cost within budget", "status": INFO, "value": "—", "note": "no cost estimate"})
    else:
        est, budget = cost.get("est_total_gpu_hours"), cost.get("budget_gpu_hours")
        st = INFO if budget is None or est is None else (FAIL if est > budget else PASS)
        checks.append({"check": "cost within budget", "status": st,
                       "value": f"{est:.1f} GPU-h (est.)" if est is not None else "—",
                       "note": cost.get("note", "") + (f"; budget {budget} GPU-h" if budget is not None else "; no --budget-gpu-hours given")})
    return checks


def replicate_agreement(rows_a: list[dict], rows_b: list[dict]) -> dict:
    b = {r["case_id"]: r for r in rows_b}
    pairs = [(r, b[r["case_id"]]) for r in rows_a if r["case_id"] in b]
    same_status = [(x, y) for x, y in pairs if x["status"] == y["status"]]
    both_valid = [(x, y) for x, y in pairs if x["status"] == "valid" and y["status"] == "valid"]
    agree = sum((x["status"], x["verdict"]) == (y["status"], y["verdict"]) for x, y in pairs)
    diffs = [abs(x["confidence"] - y["confidence"]) for x, y in both_valid]
    return {
        "n_paired": len(pairs),
        "verdict_agreement": agree / len(pairs) if pairs else None,
        "status_agreement": len(same_status) / len(pairs) if pairs else None,
        "mean_abs_conf_diff": sum(diffs) / len(diffs) if diffs else None,
        "max_abs_conf_diff": max(diffs) if diffs else None,
    }


def cost_estimate(records: list[dict], batch_size: int, formal_cases: int = 200, formal_runs: int = 10,
                  budget_gpu_hours: float | None = None, gpus: int = 1) -> dict:
    lat = [r["latency"] for r in records if r.get("latency") is not None]
    if not lat:
        return {"est_total_gpu_hours": None, "budget_gpu_hours": budget_gpu_hours, "note": "no latency data"}
    mean_lat = sum(lat) / len(lat)
    per_run_h = mean_lat * formal_cases / max(batch_size, 1) / 3600 * gpus
    return {
        "mean_latency_s": mean_lat,
        "batch_size": batch_size,
        "est_gpu_hours_per_run": per_run_h,
        "est_total_gpu_hours": per_run_h * formal_runs,
        "budget_gpu_hours": budget_gpu_hours,
        "note": f"{formal_runs} runs x {formal_cases} cases, mean latency {mean_lat:.2f}s / batch {batch_size}, {gpus} GPU(s)",
    }


def checks_markdown(checks: list[dict], title: str = "Gate checks (section 6.1)") -> str:
    lines = [f"### {title}", "", "| Check | Status | Value | Note |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | **{c['status']}** | {c['value']} | {c['note']} |")
    overall = FAIL if any(c["status"] == FAIL for c in checks) else BORDER if any(
        c["status"] == BORDER for c in checks) else PASS
    lines += ["", f"**Overall: {overall}**  (BORDERLINE = group decision; FAIL -> follow section 6.2 and log in decision_log.md)"]
    return "\n".join(lines)


def difficulty_report(rows: list[dict], title: str) -> str:
    """`probe` output: error rates on correct vs incorrect candidates, by candidate_type, confidence spread."""
    s = run_stats(rows)
    bias = M.bias_split(rows)
    by_type = M.by_candidate_type(rows)
    cs = M.confidence_stats(rows)
    lo, hi = s["error_rate_ci"]
    out = [f"## Difficulty report — {title}", "",
           f"- cases: {s['n']}  (splits {s['splits']}), valid: {s['n_valid']} ({_fmt(s['valid_rate'])})",
           f"- **judge error rate** (valid only): {_fmt(s['error_rate'])}  95% CI [{_fmt(lo)}, {_fmt(hi)}]",
           f"- on CORRECT candidates (false reject): {_fmt(bias['false_reject_rate'])}  (n={bias['n_correct_candidates']})",
           f"- on INCORRECT candidates (false accept): {_fmt(bias['false_accept_rate'])}  (n={bias['n_incorrect_candidates']})",
           f"- status counts: {s['status_counts']}",
           "", "| candidate_type | n | valid | judge errors | error rate |", "|---|---|---|---|---|"]
    for t, d in sorted(by_type.items()):
        out.append(f"| {t} | {d['n']} | {d['n_valid']} | {d['n_errors']} | {_fmt(d['error_rate'])} |")
    out += ["", "Confidence distribution (valid judgments):"]
    if cs.get("n"):
        q = cs["quantiles"]
        out.append(f"- mean {cs['mean']:.3f}; min {q[0.0]:.2f} / p10 {q[0.1]:.2f} / p25 {q[0.25]:.2f} / "
                   f"median {q[0.5]:.2f} / p75 {q[0.75]:.2f} / p90 {q[0.9]:.2f} / max {q[1.0]:.2f}")
        out.append(f"- distinct values: {cs['n_distinct']}; most common: " +
                   ", ".join(f"{v['value']} ({_fmt(v['share'])})" for v in cs["top_values"]))
        hist = Counter(min(int(r["confidence"] * 10), 9) for r in rows if r["status"] == "valid")
        for b in range(10):
            n = hist.get(b, 0)
            out.append(f"    [{b / 10:.1f}, {(b + 1) / 10:.1f}{']' if b == 9 else ')'} {'#' * int(round(40 * n / cs['n']))} {n}")
    else:
        out.append("- no valid judgments")
    out += ["", "Reading: error rate well below 10% -> errors too easy to spot (make them subtler, section 6.2 step 1); "
                "above 40% -> judge near chance or errors ambiguous."]
    return "\n".join(out)
