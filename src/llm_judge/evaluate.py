"""`evaluate`: parse raw outputs -> calibrate on calib (freeze) -> policies on test -> metrics -> paired bootstrap.

One call = one dataset x one model x a set of configs. Holm correction across datasets happens in `report`.

Sensitivity analysis (D2/D3): invalid outputs are turned into wrong judgments with a confidence that ranks
strictly last (internally -1.0; reported as "confidence 0, ranked last") and the same calibration / policy /
metric code is re-run on those transformed rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from llm_judge import metrics as M
from llm_judge import policies as P
from llm_judge import stats as S
from llm_judge.calibrate import (DEFAULT_SALT, TARGETS, fit_thresholds, freeze_thresholds, threshold_key)
from llm_judge.parse import PARSER_VERSION, parse_record, parser_hash
from llm_judge.registry import build_prompt_plan, get_dataset, get_model, load_config
from llm_judge.runner import load_records, run_dir_for
from llm_judge.schema import Case
from llm_judge.utils import now_iso, read_json, write_json

SENS_CONF = -1.0
PRIMARY_TARGET = 0.75


class EvaluationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- rows

def build_rows(cases: list[Case], records: dict[str, dict], output_mode: str) -> list[dict]:
    rows = []
    for c in sorted(cases, key=lambda c: c.case_id):
        rec = records.get(c.case_id)
        if rec is None:
            j_status, verdict, conf, err = "exec_failure", None, None, "missing raw output"
        else:
            j = parse_record(rec, output_mode)
            j_status, verdict, conf, err = j.status, j.verdict, j.confidence, j.error
        judge_error = None
        if j_status == "valid":
            judge_error = (verdict == "correct") != bool(c.label)
        rows.append({
            "case_id": c.case_id,
            "question_id": c.question_id,
            "split": c.split,
            "label": bool(c.label),
            "candidate_type": c.candidate_type,
            "status": j_status,
            "verdict": verdict,
            "confidence": conf,
            "judge_error": judge_error,
            "parse_error": err,
            "missing": rec is None,
        })
    return rows


def sensitivity_rows(rows: list[dict]) -> list[dict]:
    """Invalid outputs -> wrong judgments ranked last (D2). Abstentions are left untouched."""
    out = []
    for r in rows:
        if r["status"] in ("valid", "abstain"):
            out.append(r)
        else:
            out.append(r | {"status": "valid", "verdict": None, "confidence": SENS_CONF, "judge_error": True,
                            "imputed_invalid": True})
    return out


# --------------------------------------------------------------------------- per-config test evaluation

def _policy_block(rows: list[dict], thr_all: dict, R: int, seed: int, salt: str) -> dict:
    n = len(rows)
    out = {"no_abstention": P.policy_summary(rows, P.no_abstention(rows), n), "confidence": {}}
    pool = P.valid_error_pool(rows)
    for tkey, thr in thr_all.items():
        if not thr["reachable"]:
            out["confidence"][tkey] = {"reachable": False, "target": thr["target"],
                                       "note": "target coverage exceeds the calib maximum (1 - invalid rate)"}
            continue
        keep = P.confidence_policy(rows, thr, salt)
        summ = P.policy_summary(rows, keep, n)
        rb = P.random_baseline(pool, summ["k"], R=R, seed=seed)
        diff = None
        if summ["selective_risk"] is not None and rb.get("analytic_expected_risk") is not None:
            diff = summ["selective_risk"] - rb["analytic_expected_risk"]
        out["confidence"][tkey] = summ | {
            "reachable": True,
            "target": thr["target"],
            "coverage_gap": summ["coverage"] - thr["target"],
            "random": rb,
            "risk_diff_vs_random": diff,
        }
    return out


def evaluate_config_test(rows: list[dict], thresholds: dict, records: list[dict], manifest: dict | None,
                         R: int, seed: int, salt: str, output_mode: str) -> dict:
    a = M.arrays(rows, salt)
    res = {
        "n_test": len(rows),
        "validity": M.validity_table(rows),
        "accuracy": M.accuracy(a),
        "bias": M.bias_split(rows),
        "by_candidate_type": M.by_candidate_type(rows),
        "confidence_stats": M.confidence_stats(rows),
        "brier": M.brier(a),
        "reliability": M.reliability(a),
        "cost": M.cost(records, manifest),
    }
    if output_mode == "binary":
        res["aurc"] = M.aurc_metrics(a)
        res["curve"] = M.risk_coverage_curve(a)
        res["policies"] = _policy_block(rows, thresholds["primary"], R, seed, salt)
        sens = sensitivity_rows(rows)
        res["policies_sensitivity"] = _policy_block(sens, thresholds["sensitivity"], R, seed, salt)
    else:
        keep = P.direct_abstention(rows)
        res["direct_abstention"] = P.policy_summary(rows, keep, len(rows))
    return res


# --------------------------------------------------------------------------- bootstrap comparisons

class _Boot:
    """Holds the shared index matrix and per-config arrays for one domain's test split."""

    def __init__(self, test_rows_by_cfg: dict[str, list[dict]], B: int, seed: int, salt: str):
        any_rows = next(iter(test_rows_by_cfg.values()))
        self.case_ids = [r["case_id"] for r in any_rows]
        for cid, rows in test_rows_by_cfg.items():
            if [r["case_id"] for r in rows] != self.case_ids:
                raise EvaluationError(f"config {cid}: test case set differs from other configs; cannot pair")
        self.I = S.bootstrap_indices(self.case_ids, [r["question_id"] for r in any_rows],
                                     [r["label"] for r in any_rows], B, seed)
        self.I0 = np.arange(len(self.case_ids))[None, :]  # identity "resample" -> point estimate
        self.B = B
        self.salt = salt
        self.arr = {c: M.arrays(rows, salt) for c, rows in test_rows_by_cfg.items()}
        self.arr_sens = {c: M.arrays(sensitivity_rows(rows), salt) for c, rows in test_rows_by_cfg.items()}

    def rate(self, num: np.ndarray, den: np.ndarray, I: np.ndarray | None = None) -> np.ndarray:
        I = self.I if I is None else I
        return S.ratio_rows(num[I].sum(axis=1), den[I].sum(axis=1))

    def selective_diff(self, a: dict, keep: np.ndarray, I: np.ndarray | None = None) -> np.ndarray:
        """Selective risk under the confidence policy minus the random-abstention expectation
        (== error rate of the valid pool, exactly, for any k <= pool size)."""
        risk = self.rate(a["err"] * keep, keep.astype(float), I)
        pool = self.rate(a["err"] * a["valid"], a["valid"].astype(float), I)
        return risk - pool

    def aurc(self, a: dict, sensitivity: bool) -> np.ndarray:
        idx = M.sort_by_rank(self.I, a["rank"])
        return M.aurc_batch(a["err"][idx], a["valid"][idx], sensitivity=sensitivity)

    def acc(self, a: dict) -> np.ndarray:
        return 1.0 - self.rate(a["err"] * a["valid"], a["valid"].astype(float))

    def brier(self, a: dict) -> np.ndarray:
        sq = np.where(a["valid"], (a["conf"] - (1.0 - a["err"])) ** 2, 0.0)
        return self.rate(sq, a["valid"].astype(float))


def _keep_mask(rows: list[dict], thr: dict, salt: str) -> np.ndarray | None:
    if not thr["reachable"]:
        return None
    return np.array(P.confidence_policy(rows, thr, salt), dtype=float)


def _factorial_roles(configs: dict) -> dict:
    """Map (thinking, rubric) roles among binary configs: base / think / rubric / both."""
    roles = {}
    for cid, cfg in configs.items():
        if cfg.output_mode != "binary":
            continue
        role = {(False, False): "base", (True, False): "think", (False, True): "rubric", (True, True): "both"}[
            (cfg.thinking, cfg.rubric)]
        roles.setdefault(role, cid)
    return roles


def compare(test_rows: dict[str, list[dict]], configs: dict, thresholds: dict, B: int, seed: int, salt: str) -> dict:
    binary = [c for c in test_rows if configs[c].output_mode == "binary"]
    ternary = [c for c in test_rows if configs[c].output_mode == "ternary"]
    if not test_rows:
        return {}
    bt = _Boot(test_rows, B, seed, salt)
    out: dict = {"B": B, "seed": seed, "primary_endpoint": {}, "selective_secondary": {}, "sensitivity_endpoint": {},
                 "aurc": {}, "aurc_diff_vs_base": {}, "accuracy_vs_base": {}, "brier": {}, "interaction": {},
                 "direct_abstention": {}}
    roles = _factorial_roles({c: configs[c] for c in test_rows})  # n/a configs have no rows -> no role
    base = roles.get("base")

    for c in binary:
        a, a_s = bt.arr[c], bt.arr_sens[c]
        for tkey, thr in thresholds[c]["primary"].items():
            keep = _keep_mask(test_rows[c], thr, salt)
            target = thr["target"]
            if keep is None:
                block = {"reachable": False, "estimate": None, "ci95": [None, None], "p_boot": None}
            else:
                samples = bt.selective_diff(a, keep)
                point = float(bt.selective_diff(a, keep, bt.I0)[0])
                block = {"reachable": True} | S.summarize(point, samples, with_p=True)
            if abs(target - PRIMARY_TARGET) < 1e-9:
                out["primary_endpoint"][c] = block
            else:
                out["selective_secondary"].setdefault(c, {})[tkey] = block
            # sensitivity version of the same comparison
            sens_rows = sensitivity_rows(test_rows[c])
            keep_s = _keep_mask(sens_rows, thresholds[c]["sensitivity"][tkey], salt)
            if keep_s is not None:
                samples_s = bt.selective_diff(a_s, keep_s)
                point_s = float(bt.selective_diff(a_s, keep_s, bt.I0)[0])
                out["sensitivity_endpoint"].setdefault(c, {})[tkey] = {"reachable": True} | S.summarize(point_s, samples_s, with_p=True)
            else:
                out["sensitivity_endpoint"].setdefault(c, {})[tkey] = {"reachable": False}

        aur = bt.aurc(a, False)
        aur_s = bt.aurc(a, True)
        pm = M.aurc_metrics(a)
        out["aurc"][c] = {"primary": S.summarize(pm["aurc"], aur), "sensitivity": S.summarize(pm["aurc_sens"], aur_s)}
        out["brier"][c] = S.summarize(M.brier(a), bt.brier(a))

    if base is not None:
        a0 = bt.arr[base]
        aur0, aur0_s = bt.aurc(a0, False), bt.aurc(a0, True)
        acc0 = bt.acc(a0)
        pm0 = M.aurc_metrics(a0)
        for c in binary:
            if c == base:
                continue
            a = bt.arr[c]
            pm = M.aurc_metrics(a)
            out["aurc_diff_vs_base"][c] = {
                "base": base,
                "primary": S.summarize(_sub(pm["aurc"], pm0["aurc"]), bt.aurc(a, False) - aur0),
                "sensitivity": S.summarize(_sub(pm["aurc_sens"], pm0["aurc_sens"]), bt.aurc(a, True) - aur0_s),
            }
            both = a["valid"] & a0["valid"]
            right_c = both & (a["err"] == 0)
            right_0 = both & (a0["err"] == 0)
            b_cnt = int(np.sum(right_0 & ~right_c))  # base right, c wrong
            c_cnt = int(np.sum(~right_0 & right_c & both))
            acc_c, acc_b = M.accuracy(a)["accuracy_valid"], M.accuracy(a0)["accuracy_valid"]
            out["accuracy_vs_base"][c] = {
                "base": base,
                "accuracy": acc_c,
                "accuracy_base": acc_b,
                "diff": S.summarize(_sub(acc_c, acc_b), bt.acc(a) - acc0),
                "mcnemar": {"n_both_valid": int(both.sum()), "base_right_other_wrong": b_cnt,
                            "base_wrong_other_right": c_cnt, "p_exact": S.mcnemar_exact(b_cnt, c_cnt)},
            }

    if all(r in roles for r in ("base", "think", "rubric", "both")):
        ab, at, ar, ah = (bt.arr[roles[r]] for r in ("base", "think", "rubric", "both"))
        acc = {r: M.accuracy(bt.arr[roles[r]])["accuracy_valid"] for r in roles}
        aurc_pt = {r: M.aurc_metrics(bt.arr[roles[r]])["aurc"] for r in roles}
        inter_acc = (bt.acc(ah) - bt.acc(ar)) - (bt.acc(at) - bt.acc(ab))
        inter_aurc = (bt.aurc(ah, False) - bt.aurc(ar, False)) - (bt.aurc(at, False) - bt.aurc(ab, False))
        out["interaction"] = {
            "roles": roles,
            "definition": "(both - rubric) - (think - base)",
            "accuracy": S.summarize(_inter(acc), inter_acc),
            "aurc": S.summarize(_inter(aurc_pt), inter_aurc),
        }

    for c in ternary:
        a5 = bt.arr[c]
        keep5 = a5["valid"].astype(float)
        risk5 = bt.rate(a5["err"] * keep5, keep5)
        k5 = int(keep5.sum())
        n = len(keep5)
        block = {"k": k5, "coverage": k5 / n if n else 0.0,
                 "selective_risk": float(a5["err"][a5["valid"]].mean()) if k5 else None}
        if base is not None:
            a1 = bt.arr[base]
            pool1 = bt.rate(a1["err"] * a1["valid"], a1["valid"].astype(float))
            pool1_pt = float(a1["err"][a1["valid"]].mean()) if a1["valid"].any() else None
            rb = P.random_baseline(a1["err"][a1["valid"]], k5, R=10_000, seed=seed)
            block["vs_base_random_same_k"] = {
                "base": base,
                "random": rb,
                "diff": S.summarize(_sub(block["selective_risk"], pool1_pt), risk5 - pool1),
            }
            # descriptive: base config's confidence curve at the same coverage (coverage matched ON TEST)
            idx = M.sort_by_rank(bt.I, a1["rank"])
            risk_mat = M.risk_matrix(a1["err"][idx], a1["valid"][idx], sensitivity=False)
            nv1 = a1["valid"][bt.I].sum(axis=1)
            kb = keep5[bt.I].sum(axis=1).astype(int)
            ok = (kb >= 1) & (kb <= nv1)
            curve_b = np.full(bt.B, np.nan)
            curve_b[ok] = risk_mat[np.arange(bt.B)[ok], kb[ok] - 1]
            pt_curve = M.curve_point(a1, k5)
            block["vs_base_confidence_curve_same_coverage"] = {
                "base": base,
                "descriptive_only": True,
                "note": "coverage matched on the test split -> descriptive comparison, not a test",
                "base_curve_risk": pt_curve,
                "diff": S.summarize(_sub(block["selective_risk"], pt_curve), risk5 - curve_b),
            }
        out["direct_abstention"][c] = block
    return out


def _sub(x, y):
    return None if x is None or y is None else x - y


def _inter(v: dict):
    try:
        return (v["both"] - v["rubric"]) - (v["think"] - v["base"])
    except TypeError:
        return None


# --------------------------------------------------------------------------- top-level

def evaluate_dataset(
    root: Path,
    cases: list[Case],
    model_id: str,
    config_ids: list[str],
    out_root: Path,
    eval_dir: Path | None = None,
    thresholds_path: Path | None = None,
    role_override: dict[str, str] | None = None,
    run_tag: str = "main",
    B: int = 10_000,
    R: int = 10_000,
    boot_seed: int = 20261107,
    random_seed: int = 12345,
    salt: str = DEFAULT_SALT,
    targets=TARGETS,
    force_thresholds: bool = False,
    allow_missing: bool = False,
    data_path: str | None = None,
) -> dict:
    root, out_root = Path(root), Path(out_root)
    dataset_id = cases[0].dataset_id
    ds = get_dataset(root, dataset_id)
    model = get_model(root, model_id)
    thresholds_path = Path(thresholds_path or out_root / "frozen" / "thresholds.json")
    eval_dir = Path(eval_dir or out_root / "eval" / dataset_id / model_id)

    role = {c.case_id: (role_override or {}).get(c.case_id, c.split) for c in cases}
    calib_cases = [c for c in cases if role[c.case_id] == "calib"]
    test_cases = [c for c in cases if role[c.case_id] == "test"]
    if not calib_cases or not test_cases:
        raise EvaluationError(f"need both calib and test cases (got calib={len(calib_cases)}, test={len(test_cases)})")

    configs, results, calib_rows, test_rows, thresholds, per_case = {}, {}, {}, {}, {}, {}
    for cid in config_ids:
        cfg = load_config(root, cid)
        plan = build_prompt_plan(root, ds, model, cfg)
        configs[cid] = cfg
        if plan.is_na:
            results[cid] = {"na": True, "na_reason": plan.na_reason, "config_hash": plan.config_hash}
            continue
        rdir = run_dir_for(out_root, plan, run_tag)
        if not rdir.exists():
            raise EvaluationError(f"no raw outputs for {cid} at {rdir} (config_hash {plan.config_hash}). "
                                  f"Has the prompt/config changed since the run?")
        recs = load_records(rdir)
        c_rows = build_rows(calib_cases, recs, cfg.output_mode)
        t_rows = build_rows(test_cases, recs, cfg.output_mode)
        n_missing = sum(r["missing"] for r in c_rows + t_rows)
        if n_missing and not allow_missing:
            raise EvaluationError(f"{cid}: {n_missing} calib/test cases have no raw output in {rdir}. "
                                  f"Finish the run (it resumes) or pass --allow-missing.")
        calib_rows[cid], test_rows[cid] = c_rows, t_rows
        man = read_json(rdir / "run_manifest.json") if (rdir / "run_manifest.json").exists() else None
        results[cid] = {"na": False, "config_hash": plan.config_hash, "run_dir": str(rdir),
                        "output_mode": cfg.output_mode, "thinking": cfg.thinking, "rubric": cfg.rubric,
                        "n_missing": n_missing, "_records": [recs[c.case_id] for c in test_cases if c.case_id in recs],
                        "_manifest": man}

    # ---- [D] calibrate on calib only, freeze BEFORE looking at test
    for cid, rows in calib_rows.items():
        if configs[cid].output_mode != "binary":
            continue
        thr = {"primary": fit_thresholds(rows, targets, salt),
               "sensitivity": fit_thresholds(sensitivity_rows(rows), targets, salt)}
        key = threshold_key(dataset_id, model_id, cid)
        entry = freeze_thresholds(thresholds_path, key, results[cid]["config_hash"], thr, salt,
                                  [r["case_id"] for r in rows], force=force_thresholds)
        thresholds[cid] = entry["thresholds"]
        results[cid]["calibration"] = {"key": key, "thresholds_file": str(thresholds_path),
                                       "entry_sha256": entry.get("entry_sha256"),
                                       "validity_calib": M.validity_table(rows)}

    # ---- [E][F] test
    for cid, rows in test_rows.items():
        r = results[cid]
        recs, man = r.pop("_records"), r.pop("_manifest")
        r["test"] = evaluate_config_test(rows, thresholds.get(cid, {}), recs, man, R, random_seed, salt,
                                         configs[cid].output_mode)
        per_case[cid] = rows
    for r in results.values():
        r.pop("_records", None)
        r.pop("_manifest", None)

    # ---- [G] paired bootstrap within this domain
    comparisons = compare(test_rows, configs, thresholds, B, boot_seed, salt) if test_rows else {}

    evaluation = {
        "meta": {
            "dataset_id": dataset_id,
            "domain": ds.domain,
            "task_type": ds.task_type,
            "model_id": model_id,
            "configs": config_ids,
            "n_calib": len(calib_cases),
            "n_test": len(test_cases),
            "targets": list(targets),
            "primary_target": PRIMARY_TARGET,
            "B": B, "R": R, "boot_seed": boot_seed, "random_seed": random_seed, "salt": salt,
            "parser_version": PARSER_VERSION, "parser_hash": parser_hash(),
            "thresholds_file": str(thresholds_path),
            "role_override": bool(role_override),
            "data_path": data_path,
            "created_at": now_iso(),
        },
        "configs": results,
        "comparisons": comparisons,
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "evaluation.json", evaluation)
    write_json(eval_dir / "per_case.json", {"calib": calib_rows, "test": per_case})
    _write_csv(eval_dir / "cases_test.csv", per_case)
    return evaluation


def _write_csv(path: Path, rows_by_cfg: dict[str, list[dict]]) -> None:
    cols = ["config_id", "case_id", "question_id", "split", "label", "candidate_type", "status", "verdict",
            "confidence", "judge_error", "parse_error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for cid, rows in rows_by_cfg.items():
            for r in rows:
                w.writerow({"config_id": cid, **r})
