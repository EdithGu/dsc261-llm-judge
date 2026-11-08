"""`report`: combine evaluation.json files (one per dataset) for a model into tables, figures and report.md.

Holm correction for the pre-registered primary endpoint is applied HERE, across all (config x domain) tests
that were evaluated (8 in the full design: E1-E4 x {GSM8K, MedQA}). Unreachable tests enter Holm as p = 1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from llm_judge import stats as S
from llm_judge.utils import now_iso, read_json, write_json

# Categorical slots 1-5 of the reference palette (validated light-mode; contrast WARN -> direct labels + tables)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "*"]
RELIABILITY_MIN_N = 5
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _f(x, pct=True, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if pct:
        return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def _pct(x):
    v = _f(x)
    return v if v == "—" else f"{v}%"


def _p(x):
    return "—" if x is None else f"{x:.3g}"


def _ci(block, pct=True):
    if not block or block.get("ci95") is None:
        return "—"
    lo, hi = block["ci95"]
    if lo is None:
        return "—"
    return f"[{_f(lo, pct, True)}, {_f(hi, pct, True)}]"


def load_evaluations(eval_root: Path, model_id: str, datasets: list[str]) -> dict[str, dict]:
    out = {}
    for d in datasets:
        p = Path(eval_root) / d / model_id / "evaluation.json"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run `evaluate` for {d} first")
        out[d] = read_json(p)
    return out


def primary_table(evals: dict[str, dict]) -> list[dict]:
    rows = []
    for d, ev in evals.items():
        for c, blk in (ev["comparisons"].get("primary_endpoint") or {}).items():
            pol = ev["configs"][c]["test"]["policies"]["confidence"].get(f"{ev['meta']['primary_target']:.2f}", {})
            rows.append({
                "dataset": d, "domain": ev["meta"]["domain"], "config": c,
                "reachable": blk.get("reachable", False),
                "coverage": pol.get("coverage"), "k": pol.get("k"),
                "risk_conf": pol.get("selective_risk"),
                "risk_random": (pol.get("random") or {}).get("analytic_expected_risk"),
                "diff": blk.get("estimate"), "ci95": blk.get("ci95"), "p_boot": blk.get("p_boot"),
            })
    adj = S.holm({f"{r['dataset']}/{r['config']}": (r["p_boot"] if r["reachable"] else None) for r in rows})
    for r in rows:
        r["p_holm"] = adj[f"{r['dataset']}/{r['config']}"]
        r["reject_h0_at_0.05"] = r["p_holm"] is not None and r["p_holm"] < 0.05
    return rows


def domain_differences(eval_root: Path, model_id: str, evals: dict[str, dict], B: int, seed: int) -> list[dict]:
    """Unpaired bootstrap of (domain A - domain B) for full-coverage accuracy and the 75% risk difference."""
    ds = list(evals)
    if len(ds) != 2:
        return []
    from llm_judge import metrics as M
    from llm_judge.policies import confidence_policy
    from llm_judge.calibrate import load_frozen, threshold_key

    per = {d: read_json(Path(eval_root) / d / model_id / "per_case.json")["test"] for d in ds}
    out = []
    common = [c for c in per[ds[0]] if c in per[ds[1]]]
    for c in common:
        samples, points = {}, {}
        for i, d in enumerate(ds):
            rows = per[d][c]
            a = M.arrays(rows)
            I = S.bootstrap_indices([r["case_id"] for r in rows], [r["question_id"] for r in rows],
                                    [r["label"] for r in rows], B, seed + 1000 * (i + 1))
            acc = 1 - S.ratio_rows((a["err"] * a["valid"])[I].sum(1), a["valid"].astype(float)[I].sum(1))
            points[(d, "acc")] = M.accuracy(a)["accuracy_valid"]
            samples[(d, "acc")] = acc
            ev = evals[d]
            if ev["configs"][c].get("output_mode") == "binary":
                fr = load_frozen(Path(ev["meta"]["thresholds_file"]), threshold_key(d, model_id, c))
                thr = fr["thresholds"]["primary"][f"{ev['meta']['primary_target']:.2f}"] if fr else None
                if thr and thr["reachable"]:
                    keep = np.array(confidence_policy(rows, thr, ev["meta"]["salt"]), dtype=float)
                    risk = S.ratio_rows((a["err"] * keep)[I].sum(1), keep[I].sum(1))
                    pool = S.ratio_rows((a["err"] * a["valid"])[I].sum(1), a["valid"].astype(float)[I].sum(1))
                    samples[(d, "diff")] = risk - pool
                    points[(d, "diff")] = ev["comparisons"]["primary_endpoint"][c]["estimate"]
        row = {"config": c, "contrast": f"{ds[0]} - {ds[1]}"}
        for m in ("acc", "diff"):
            if (ds[0], m) in samples and (ds[1], m) in samples:
                p0, p1 = points[(ds[0], m)], points[(ds[1], m)]
                row[m] = S.summarize(None if p0 is None or p1 is None else p0 - p1,
                                     samples[(ds[0], m)] - samples[(ds[1], m)])
        out.append(row)
    return out


# --------------------------------------------------------------------------- figures

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_risk_coverage(ev: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    configs = list(ev["configs"])
    for i, c in enumerate(configs):
        r = ev["configs"][c]
        if r.get("na"):
            continue
        col, mk = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
        t = r["test"]
        if r["output_mode"] == "binary":
            cov, risk = t["curve"]["coverage"], t["curve"]["risk"]
            if not cov:
                continue
            ax.plot(cov, risk, color=col, linewidth=2, label=f"{c} confidence", zorder=3)
            ax.text(cov[-1] + 0.01, risk[-1], c, color=INK2, fontsize=8, va="center")
            for tk, p in t["policies"]["confidence"].items():
                if p.get("reachable") and p.get("selective_risk") is not None:
                    ax.plot(p["coverage"], p["selective_risk"], marker=mk, markersize=8, color=col,
                            markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=4)
            full = t["policies"]["no_abstention"]["selective_risk"]
            if full is not None:
                ax.hlines(full, 0, cov[-1], colors=col, linestyles=(0, (3, 3)), linewidth=1, alpha=0.7, zorder=2)
        else:
            da = t["direct_abstention"]
            if da["selective_risk"] is not None:
                ax.plot(da["coverage"], da["selective_risk"], marker=mk, markersize=12, color=col,
                        markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label=f"{c} direct abstention", zorder=5)
                ax.annotate(c, (da["coverage"], da["selective_risk"]), xytext=(6, 6), textcoords="offset points",
                            color=INK2, fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(bottom=0)
    _style(ax, f"Risk–coverage on test — {ev['meta']['dataset_id']} ({ev['meta']['model_id']})",
           "coverage (kept judgments / all cases)", "selective risk (judge error rate among kept)")
    ax.text(0.0, -0.30, "dashed = random abstention expectation (= full-coverage error rate); markers = frozen "
            "thresholds at 50/75/90% target", transform=ax.transAxes, fontsize=7, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(ev: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
    for i, c in enumerate(ev["configs"]):
        r = ev["configs"][c]
        if r.get("na"):
            continue
        bins = [b for b in r["test"]["reliability"] if b["n"] >= RELIABILITY_MIN_N]
        if not bins:
            continue
        xs, ys = [b["mean_conf"] for b in bins], [b["accuracy"] for b in bins]
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, marker=MARKERS[i % len(MARKERS)], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{c} (Brier {_f(r['test']['brier'], False)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, f"Reliability — {ev['meta']['dataset_id']}", "stated confidence", "observed verdict accuracy")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.text(0.0, -0.14, f"bins with fewer than {RELIABILITY_MIN_N} judgments are hidden (all bins are in evaluation.json)",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- tables + markdown

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def build_report(eval_root: Path, model_id: str, datasets: list[str], out_dir: Path, B: int = 10_000,
                 seed: int = 20261107) -> Path:
    evals = load_evaluations(eval_root, model_id, datasets)
    out_dir = Path(out_dir)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    prim = primary_table(evals)
    _write_csv(tab_dir / "primary_endpoint.csv", [r | {"ci95": r["ci95"]} for r in prim])

    metric_rows, validity_rows, cost_rows, secondary_rows, e5_rows = [], [], [], [], []
    for d, ev in evals.items():
        cmp_ = ev["comparisons"]
        for c, r in ev["configs"].items():
            if r.get("na"):
                metric_rows.append({"dataset": d, "config": c, "status": f"n/a ({r['na_reason']})"})
                continue
            t = r["test"]
            row = {"dataset": d, "config": c, "status": "ok", "n_test": t["n_test"],
                   "acc_valid": t["accuracy"]["accuracy_valid"], "acc_sens": t["accuracy"]["accuracy_sens"],
                   "brier": t["brier"], "false_accept": t["bias"]["false_accept_rate"],
                   "false_reject": t["bias"]["false_reject_rate"]}
            if r["output_mode"] == "binary":
                row |= {"aurc": t["aurc"]["aurc"], "aurc_sens": t["aurc"]["aurc_sens"], "eaurc": t["aurc"]["eaurc"],
                        "max_coverage": t["aurc"]["max_coverage"]}
                for tk, p in t["policies"]["confidence"].items():
                    if not p.get("reachable"):
                        row[f"cov@{tk}"] = "unreachable"
                        continue
                    row[f"cov@{tk}"] = p["coverage"]
                    row[f"risk@{tk}"] = p["selective_risk"]
                    row[f"referral@{tk}"] = p["referral_volume"]
                    row[f"err_capture@{tk}"] = p["error_capture_rate"]
                for tk, blk in (cmp_.get("selective_secondary", {}).get(c) or {}).items():
                    secondary_rows.append({"dataset": d, "config": c, "target": tk, "diff": blk.get("estimate"),
                                           "ci95": blk.get("ci95"), "reachable": blk.get("reachable")})
            metric_rows.append(row)
            validity_rows.append({"dataset": d, "config": c, **{s: v["share"] for s, v in t["validity"].items() if s != "n_total"},
                                  "n": t["validity"]["n_total"]})
            cost_rows.append({"dataset": d, "config": c, **t["cost"]})
        for c, blk in (cmp_.get("direct_abstention") or {}).items():
            e5_rows.append({"dataset": d, "config": c, "coverage": blk["coverage"], "risk": blk["selective_risk"],
                            "diff_vs_random": (blk.get("vs_base_random_same_k") or {}).get("diff", {}).get("estimate"),
                            "ci_vs_random": (blk.get("vs_base_random_same_k") or {}).get("diff", {}).get("ci95"),
                            "diff_vs_curve_descriptive": (blk.get("vs_base_confidence_curve_same_coverage") or {}).get("diff", {}).get("estimate")})
        plot_risk_coverage(ev, fig_dir / f"risk_coverage_{d}.png")
        plot_reliability(ev, fig_dir / f"reliability_{d}.png")

    _write_csv(tab_dir / "metrics.csv", metric_rows)
    _write_csv(tab_dir / "validity.csv", validity_rows)
    _write_csv(tab_dir / "cost.csv", cost_rows)
    _write_csv(tab_dir / "selective_secondary.csv", secondary_rows)
    _write_csv(tab_dir / "direct_abstention.csv", e5_rows)
    dom = domain_differences(eval_root, model_id, evals, B=min(B, 2000), seed=seed)

    md = [f"# Report — {model_id}", "", f"_generated {now_iso()} from {', '.join(datasets)}_", ""]
    if any(ev["meta"].get("role_override") for ev in evals.values()):
        md += ["> **SMOKE / pseudo-split run**: calib/test roles were assigned by hash, not by the data's split. "
               "Numbers validate the plumbing only.", ""]
    md += ["## Primary endpoint (pre-registered)", "",
           f"Selective risk (confidence) − selective risk (random) at {int(100 * next(iter(evals.values()))['meta']['primary_target'])}% target coverage, "
           f"paired bootstrap (B={next(iter(evals.values()))['meta']['B']}), Holm across {len(prim)} tests. Negative = confidence abstention is better.", "",
           "| dataset | config | coverage (k) | risk conf | risk random | diff (pp) | 95% CI (pp) | p boot | p Holm |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in prim:
        if not r["reachable"]:
            md.append(f"| {r['dataset']} | {r['config']} | unreachable | — | — | — | — | — | {_p(r['p_holm'])} |")
            continue
        md.append(f"| {r['dataset']} | {r['config']} | {_pct(r['coverage'])} ({r['k']}) | {_pct(r['risk_conf'])} | "
                  f"{_pct(r['risk_random'])} | {_f(r['diff'], signed=True)} | {_ci(r)} | "
                  f"{_p(r['p_boot'])} | {_p(r['p_holm'])}{' *' if r['reject_h0_at_0.05'] else ''} |")
    md += ["", "## Per-config metrics (test)", "",
           "| dataset | config | acc (valid) | acc (sens) | AURC | AURC sens | E-AURC | max cov | Brier | false accept | false reject |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in metric_rows:
        if r.get("status") != "ok":
            md.append(f"| {r['dataset']} | {r['config']} | {r['status']} | | | | | | | | |")
            continue
        md.append(f"| {r['dataset']} | {r['config']} | {_pct(r['acc_valid'])} | {_pct(r['acc_sens'])} | "
                  f"{_f(r.get('aurc'), False)} | {_f(r.get('aurc_sens'), False)} | {_f(r.get('eaurc'), False)} | "
                  f"{_pct(r.get('max_coverage'))} | {_f(r['brier'], False)} | {_pct(r['false_accept'])} | {_pct(r['false_reject'])} |")
    md += ["", "## Secondary comparisons (CI only, no correction, no significance claims)", ""]
    if secondary_rows:
        md += ["| dataset | config | target | diff (pp) | 95% CI (pp) |", "|---|---|---|---|---|"]
        for r in secondary_rows:
            md.append(f"| {r['dataset']} | {r['config']} | {r['target']} | "
                      f"{_f(r['diff'], signed=True) if r['reachable'] else 'unreachable'} | {_ci(r)} |")
    for d, ev in evals.items():
        cmp_ = ev["comparisons"]
        if cmp_.get("accuracy_vs_base"):
            md += ["", f"**{d}: accuracy vs base (McNemar exact + bootstrap CI)**", "",
                   "| config | acc | base acc | diff (pp) | 95% CI | McNemar p |", "|---|---|---|---|---|---|"]
            for c, blk in cmp_["accuracy_vs_base"].items():
                md.append(f"| {c} | {_pct(blk['accuracy'])} | {_pct(blk['accuracy_base'])} | "
                          f"{_f(blk['diff']['estimate'], signed=True)} | {_ci(blk['diff'])} | {_p(blk['mcnemar']['p_exact'])} |")
        if cmp_.get("aurc_diff_vs_base"):
            md += ["", f"**{d}: AURC difference vs base**", "", "| config | ΔAURC | 95% CI | ΔAURC sens | 95% CI |",
                   "|---|---|---|---|---|"]
            for c, blk in cmp_["aurc_diff_vs_base"].items():
                md.append(f"| {c} | {_f(blk['primary']['estimate'], False, True)} | {_ci(blk['primary'], False)} | "
                          f"{_f(blk['sensitivity']['estimate'], False, True)} | {_ci(blk['sensitivity'], False)} |")
        if cmp_.get("interaction"):
            it = cmp_["interaction"]
            md += ["", f"**{d}: thinking × rubric interaction** {it['definition']}: accuracy "
                   f"{_f(it['accuracy']['estimate'], signed=True)} pp {_ci(it['accuracy'])}; AURC "
                   f"{_f(it['aurc']['estimate'], False, True)} {_ci(it['aurc'], False)}"]
    if e5_rows:
        md += ["", "## Direct abstention (E5)", "",
               "| dataset | config | coverage | risk | diff vs base random @ same k (pp) | 95% CI | diff vs base curve @ same coverage (descriptive) |",
               "|---|---|---|---|---|---|---|"]
        for r in e5_rows:
            md.append(f"| {r['dataset']} | {r['config']} | {_pct(r['coverage'])} | {_pct(r['risk'])} | "
                      f"{_f(r['diff_vs_random'], signed=True)} | {_ci({'ci95': r['ci_vs_random']})} | "
                      f"{_f(r['diff_vs_curve_descriptive'], signed=True)} |")
    if dom:
        md += ["", "## Domain differences (unpaired bootstrap)", "", "| config | contrast | Δ accuracy (pp) | 95% CI | Δ primary diff (pp) | 95% CI |",
               "|---|---|---|---|---|---|"]
        for r in dom:
            a, df = r.get("acc") or {}, r.get("diff") or {}
            md.append(f"| {r['config']} | {r['contrast']} | {_f(a.get('estimate'), signed=True)} | {_ci(a)} | "
                      f"{_f(df.get('estimate'), signed=True)} | {_ci(df)} |")
    md += ["", "## Sensitivity analysis (invalid = wrong judgment, confidence 0, ranked last)", "",
           "| dataset | config | target | diff (pp) | 95% CI |", "|---|---|---|---|---|"]
    for d, ev in evals.items():
        for c, blocks in (ev["comparisons"].get("sensitivity_endpoint") or {}).items():
            for tk, blk in blocks.items():
                md.append(f"| {d} | {c} | {tk} | {_f(blk.get('estimate'), signed=True) if blk.get('reachable') else 'unreachable'} | {_ci(blk)} |")
    md += ["", "## Validity (share of test cases per status)", "",
           "| dataset | config | valid | invalid_format | invalid_conf | truncated | exec_failure | abstain |",
           "|---|---|---|---|---|---|---|---|"]
    for r in validity_rows:
        md.append(f"| {r['dataset']} | {r['config']} | " + " | ".join(
            f"{_pct(r[s])}" for s in ("valid", "invalid_format", "invalid_conf", "truncated", "exec_failure", "abstain")) + " |")
    md += ["", "## Cost", "", "| dataset | config | mean prompt tok | mean completion tok | mean reasoning tok | mean latency s | wallclock s |",
           "|---|---|---|---|---|---|---|"]
    for r in cost_rows:
        md.append(f"| {r['dataset']} | {r['config']} | {_f(r['mean_prompt_tokens'], False)} | {_f(r['mean_completion_tokens'], False)} | "
                  f"{_f(r['mean_reasoning_tokens'], False)} | {_f(r['mean_latency_s'], False)} | {_f(r['wallclock_s'], False)} |")
    md += ["", "## Figures", ""]
    for d in datasets:
        md += [f"![risk-coverage {d}](figures/risk_coverage_{d}.png)", f"![reliability {d}](figures/reliability_{d}.png)", ""]

    path = out_dir / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(out_dir / "report_data.json", {"primary": prim, "domain_differences": dom, "metrics": metric_rows})
    return path
