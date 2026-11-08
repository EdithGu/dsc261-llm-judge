"""Acceptance criterion (section 9.1): with the mock backend,
validate-data -> smoke -> pilot-report -> run -> evaluate -> report runs end to end.

Run on the 10-case fixtures (format demo) and on full-size synthetic data (60 calib / 140 test, both domains),
the latter exercising the 8-test Holm table.
"""

import json
import shutil

import pytest

from llm_judge.cli import main
from llm_judge.utils import read_json


def _cli(*args):
    rc = main([str(a) for a in args])
    assert rc == 0, f"command failed: {' '.join(map(str, args))}"


def _only_run_dir(out, dataset, model):
    dirs = [p for p in (out / "raw" / dataset / model).iterdir() if p.is_dir()]
    return dirs


@pytest.mark.parametrize("ds", ["gsm8k", "medqa"])
def test_fixture_flow(tmp_path, fixtures, ds):
    data_dir = tmp_path / ds
    shutil.copytree(fixtures / ds, data_dir)
    data = data_dir / f"cases_{ds}.jsonl"
    out = tmp_path / "outputs"

    _cli("validate-data", "--data", data, "--require-manifest")
    _cli("smoke", "--data", data, "--n", 20, "--backend", "mock", "--out", tmp_path / "smoke", "--B", 200, "--R", 200)
    assert (tmp_path / "smoke" / "reports" / "mock" / "report.md").exists()

    _cli("probe", "--data", data, "--n", 100, "--backend", "mock", "--model", "mock", "--configs", "E1,E2",
         "--split", "pilot,probe", "--out", out)
    _cli("probe", "--data", data, "--n", 100, "--backend", "mock", "--model", "mock", "--configs", "E1",
         "--split", "pilot,probe", "--out", out, "--tag", "r2")
    runs = sorted(p for p in (out / "raw" / ds / "mock").iterdir() if not p.name.endswith("-r2"))
    _cli("pilot-report", *sum([["--run", r] for r in runs], []), "--data", data)

    _cli("freeze", "--datasets", ds, "--models", "mock", "--out", out)
    _cli("run", "--data", data, "--model", "mock", "--out", out)
    _cli("evaluate", "--data", data, "--model", "mock", "--out", out, "--B", 200, "--R", 200)
    _cli("report", "--model", "mock", "--datasets", ds, "--out", out, "--reports-dir", tmp_path / "reports", "--B", 200)
    rep = (tmp_path / "reports" / "mock" / "report.md").read_text(encoding="utf-8")
    assert "Primary endpoint" in rep
    assert (tmp_path / "reports" / "mock" / "figures" / f"risk_coverage_{ds}.png").exists()


def test_full_size_two_domain_flow(tmp_path, synth_data, synth_med):
    out = tmp_path / "outputs"
    for data in (synth_data, synth_med):
        _cli("validate-data", "--data", data, "--require-manifest")
    _cli("freeze", "--datasets", "synth_math,synth_med", "--models", "qwen3.8-27b@mock", "--out", out)
    for data in (synth_data, synth_med):
        _cli("run", "--data", data, "--model", "qwen3.8-27b@mock", "--out", out)
        _cli("evaluate", "--data", data, "--model", "qwen3.8-27b@mock", "--out", out, "--B", 1000, "--R", 2000)
    _cli("report", "--model", "qwen3.8-27b@mock", "--datasets", "synth_math,synth_med", "--out", out,
         "--reports-dir", tmp_path / "reports", "--B", 500)

    ev = read_json(out / "eval" / "synth_math" / "qwen3.8-27b@mock" / "evaluation.json")
    assert ev["meta"]["n_calib"] == 60 and ev["meta"]["n_test"] == 140
    assert set(ev["comparisons"]["primary_endpoint"]) == {"E1", "E2", "E3", "E4"}
    assert ev["comparisons"]["interaction"]
    assert "E5" in ev["comparisons"]["direct_abstention"]
    # random baseline simulation agrees with its analytic expectation
    for c in ("E1", "E2", "E3", "E4"):
        for tk, p in ev["configs"][c]["test"]["policies"]["confidence"].items():
            if p.get("reachable"):
                assert p["random"]["abs_dev_from_analytic"] < 0.01
    # the mock judge's confidence is informative, so confidence abstention should beat random on average
    diffs = [ev["comparisons"]["primary_endpoint"][c]["estimate"] for c in ("E1", "E2", "E3", "E4")]
    assert sum(diffs) / 4 < 0

    data = read_json(tmp_path / "reports" / "qwen3.8-27b@mock" / "report_data.json")
    assert len(data["primary"]) == 8  # E1-E4 x 2 domains -> Holm over 8 tests
    assert all(r["p_holm"] >= r["p_boot"] for r in data["primary"] if r["reachable"])

    # thresholds were frozen and evaluation is reproducible (identical re-run does not modify them)
    thr_before = (out / "frozen" / "thresholds.json").read_text()
    _cli("evaluate", "--data", synth_data, "--model", "qwen3.8-27b@mock", "--out", out, "--B", 1000, "--R", 2000)
    assert (out / "frozen" / "thresholds.json").read_text() == thr_before


def test_run_refused_after_prompt_change(tmp_path, synth_data, root):
    """Changing a prompt after freeze -> formal run refused (rc 2)."""
    import os

    work = tmp_path / "repo"
    for d in ("registry", "configs", "prompts"):
        shutil.copytree(root / d, work / d)
    out = tmp_path / "outputs"
    assert main(["--root", str(work), "freeze", "--datasets", "synth_math", "--models", "mock", "--out", str(out)]) == 0
    sk = work / "prompts" / "skeletons" / "numeric_minimal_binary.md"
    sk.write_text(sk.read_text(encoding="utf-8") + "\nBe careful.\n", encoding="utf-8")
    rc = main(["--root", str(work), "run", "--data", str(synth_data), "--model", "mock", "--configs", "E1", "--out", str(out)])
    assert rc == 2


def test_unreachable_coverage_reported(tmp_path, synth_data):
    out = tmp_path / "outputs"
    _cli("freeze", "--datasets", "synth_math", "--models", "mock", "--configs", "E1", "--out", out)
    # 20% invalid outputs -> 90% coverage unreachable
    _cli("run", "--data", synth_data, "--model", "mock", "--configs", "E1", "--out", out,
         "--mock-opt", "invalid_rate=0.2")
    _cli("evaluate", "--data", synth_data, "--model", "mock", "--configs", "E1", "--out", out, "--B", 200, "--R", 200)
    ev = read_json(out / "eval" / "synth_math" / "mock" / "evaluation.json")
    assert ev["configs"]["E1"]["test"]["policies"]["confidence"]["0.90"]["reachable"] is False
    assert ev["configs"]["E1"]["test"]["policies_sensitivity"]["confidence"]["0.90"]["reachable"] is True
