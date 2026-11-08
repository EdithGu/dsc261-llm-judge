"""[F] Metrics: risk-coverage curve, AURC (primary + sensitivity, D3), E-AURC, Brier, reliability,
validity table, cost.

Ordering used everywhere (curve, AURC, bootstrap): valid judgments by (confidence desc, tie_hash asc);
for the sensitivity version, invalid outputs are appended AFTER all valid ones (as confidence-0 errors),
ordered among themselves by tie_hash (D2).

AURC definitions (k = 1..n points, risk_k = errors among top k / k):
    primary      mean of risk_k over k = 1..n_valid           == area under the curve on [0, max_coverage]
                                                                 divided by max_coverage
    primary_area sum of risk_k / N                            == raw area on [0, max_coverage]
    sensitivity  mean of risk_k over k = 1..N with invalid outputs counted as errors, ranked last
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from llm_judge.calibrate import DEFAULT_SALT, tie_hash
from llm_judge.parse import STATUSES


# --------------------------------------------------------------------------- ordering

def rank_array(rows: list[dict], salt: str = DEFAULT_SALT) -> np.ndarray:
    """Global rank of each row (0 = retained first). Valid rows first, then all others."""
    n = len(rows)
    valid_idx = [i for i, r in enumerate(rows) if r["status"] == "valid"]
    other_idx = [i for i, r in enumerate(rows) if r["status"] != "valid"]
    valid_idx.sort(key=lambda i: (-rows[i]["confidence"], tie_hash(rows[i]["case_id"], salt)))
    other_idx.sort(key=lambda i: tie_hash(rows[i]["case_id"], salt))
    rank = np.empty(n, dtype=np.int64)
    for pos, i in enumerate(valid_idx + other_idx):
        rank[i] = pos
    return rank


def arrays(rows: list[dict], salt: str = DEFAULT_SALT) -> dict:
    """Per-row numpy arrays used by metrics and the bootstrap."""
    valid = np.array([r["status"] == "valid" for r in rows], dtype=bool)
    err = np.array([bool(r["judge_error"]) if r["status"] == "valid" else False for r in rows], dtype=float)
    conf = np.array([r["confidence"] if r["status"] == "valid" else 0.0 for r in rows], dtype=float)
    label = np.array([bool(r["label"]) for r in rows], dtype=bool)
    return {"valid": valid, "err": err, "conf": conf, "label": label, "rank": rank_array(rows, salt)}


# --------------------------------------------------------------------------- curves / AURC (batched)

def sort_by_rank(idx: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """idx: (B, N) resample indices -> same indices re-ordered by global rank within each row."""
    order = np.argsort(rank[idx], axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def risk_matrix(err_sorted: np.ndarray, valid_sorted: np.ndarray, sensitivity: bool) -> np.ndarray:
    """(B, N) risk_k for k = 1..N. Primary: only the first n_valid columns are meaningful."""
    n = err_sorted.shape[1]
    k = np.arange(1, n + 1, dtype=float)
    e = np.where(valid_sorted, err_sorted, 1.0) if sensitivity else err_sorted * valid_sorted
    return np.cumsum(e, axis=1) / k


def aurc_batch(err_sorted: np.ndarray, valid_sorted: np.ndarray, sensitivity: bool = False) -> np.ndarray:
    risk = risk_matrix(err_sorted, valid_sorted, sensitivity)
    if sensitivity:
        return risk.mean(axis=1)
    n = err_sorted.shape[1]
    nv = valid_sorted.sum(axis=1)
    mask = np.arange(1, n + 1)[None, :] <= nv[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nv > 0, (risk * mask).sum(axis=1) / np.maximum(nv, 1), np.nan)


def _sorted_single(a: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(a["rank"], kind="stable")[None, :]
    return a["err"][idx], a["valid"][idx]


def risk_coverage_curve(a: dict) -> dict:
    """Full threshold sweep on one split. Returns coverage/risk arrays for the primary and sensitivity curves."""
    n = len(a["err"])
    if n == 0:
        return {"coverage": [], "risk": [], "coverage_sens": [], "risk_sens": [], "max_coverage": 0.0}
    es, vs = _sorted_single(a)
    nv = int(vs.sum())
    prim = risk_matrix(es, vs, False)[0][:nv]
    sens = risk_matrix(es, vs, True)[0]
    cov = np.arange(1, n + 1) / n
    return {
        "coverage": cov[:nv].tolist(),
        "risk": prim.tolist(),
        "coverage_sens": cov.tolist(),
        "risk_sens": sens.tolist(),
        "max_coverage": nv / n,
    }


def aurc_metrics(a: dict) -> dict:
    n = len(a["err"])
    if n == 0:
        return {"aurc": None, "aurc_area": None, "aurc_sens": None, "eaurc": None, "eaurc_sens": None}
    es, vs = _sorted_single(a)
    nv = int(vs.sum())
    prim = float(aurc_batch(es, vs, False)[0]) if nv else None
    sens = float(aurc_batch(es, vs, True)[0])
    # optimal orderings (errors last) for E-AURC
    ev = a["err"][a["valid"]]
    opt_prim = None
    if nv:
        e_opt = np.sort(ev)[None, :]
        opt_prim = float(aurc_batch(e_opt, np.ones_like(e_opt, dtype=bool), False)[0])
    e_all = np.where(a["valid"], a["err"], 1.0)
    e_opt_all = np.sort(e_all)[None, :]
    opt_sens = float(aurc_batch(e_opt_all, np.ones_like(e_opt_all, dtype=bool), True)[0])
    return {
        "aurc": prim,
        "aurc_area": prim * nv / n if prim is not None else None,
        "aurc_sens": sens,
        "eaurc": prim - opt_prim if prim is not None else None,
        "eaurc_sens": sens - opt_sens,
        "max_coverage": nv / n,
    }


def curve_point(a: dict, k: int) -> float | None:
    """Selective risk of the top-k valid judgments (confidence ordering)."""
    es, vs = _sorted_single(a)
    nv = int(vs.sum())
    if k <= 0 or k > nv:
        return None
    return float(es[0][:k].mean())


# --------------------------------------------------------------------------- calibration quality

def brier(a: dict) -> float | None:
    v = a["valid"]
    if not v.any():
        return None
    target = 1.0 - a["err"][v]  # 1 if the verdict is right
    return float(np.mean((a["conf"][v] - target) ** 2))


def reliability(a: dict, n_bins: int = 10) -> list[dict]:
    v = a["valid"]
    conf, right = a["conf"][v], 1.0 - a["err"][v]
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        out.append({
            "bin_lo": b / n_bins,
            "bin_hi": (b + 1) / n_bins,
            "n": int(m.sum()),
            "mean_conf": float(conf[m].mean()) if m.any() else None,
            "accuracy": float(right[m].mean()) if m.any() else None,
        })
    return out


def confidence_stats(rows: list[dict]) -> dict:
    confs = [r["confidence"] for r in rows if r["status"] == "valid"]
    if not confs:
        return {"n": 0}
    c = Counter(confs)
    mode_val, mode_n = c.most_common(1)[0]
    arr = np.array(confs)
    return {
        "n": len(confs),
        "mean": float(arr.mean()),
        "quantiles": {q: float(np.quantile(arr, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "n_distinct": len(c),
        "mode": mode_val,
        "mode_share": mode_n / len(confs),
        "top_values": [{"value": v, "share": n / len(confs)} for v, n in c.most_common(5)],
    }


# --------------------------------------------------------------------------- accuracy / validity / cost

def accuracy(a: dict) -> dict:
    n = len(a["err"])
    nv = int(a["valid"].sum())
    n_err = float(a["err"][a["valid"]].sum())
    return {
        "accuracy_valid": 1 - n_err / nv if nv else None,       # primary: among valid judgments
        "error_rate_valid": n_err / nv if nv else None,
        "accuracy_sens": (nv - n_err) / n if n else None,         # invalid counted as wrong
    }


def bias_split(rows: list[dict]) -> dict:
    """False accept (wrong candidate judged correct) / false reject (right candidate judged incorrect)."""
    v = [r for r in rows if r["status"] == "valid"]
    neg = [r for r in v if not r["label"]]
    pos = [r for r in v if r["label"]]
    return {
        "false_accept_rate": sum(r["verdict"] == "correct" for r in neg) / len(neg) if neg else None,
        "false_reject_rate": sum(r["verdict"] == "incorrect" for r in pos) / len(pos) if pos else None,
        "n_incorrect_candidates": len(neg),
        "n_correct_candidates": len(pos),
        "verdict_share_correct": sum(r["verdict"] == "correct" for r in v) / len(v) if v else None,
    }


def validity_table(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r["status"] for r in rows)
    return {s: {"n": c.get(s, 0), "share": c.get(s, 0) / n if n else 0.0} for s in STATUSES} | {"n_total": n}


def by_candidate_type(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["candidate_type"], {"n": 0, "n_valid": 0, "n_errors": 0})
        d["n"] += 1
        if r["status"] == "valid":
            d["n_valid"] += 1
            d["n_errors"] += bool(r["judge_error"])
    for d in out.values():
        d["error_rate"] = d["n_errors"] / d["n_valid"] if d["n_valid"] else None
    return out


def cost(records: list[dict], manifest: dict | None = None) -> dict:
    def mean_of(key):
        vals = [r["usage"].get(key) for r in records if r.get("usage") and r["usage"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    lat = [r["latency"] for r in records if r.get("latency") is not None]
    wall = None
    if manifest:
        wall = sum(s.get("seconds") or 0 for s in manifest.get("sessions", []))
    return {
        "n_records": len(records),
        "mean_prompt_tokens": mean_of("prompt_tokens"),
        "mean_completion_tokens": mean_of("completion_tokens"),
        "mean_reasoning_tokens": mean_of("reasoning_tokens"),
        "mean_latency_s": float(np.mean(lat)) if lat else None,
        "sum_latency_s": float(np.sum(lat)) if lat else None,
        "wallclock_s": wall,
        "n_retried": sum(bool(r.get("attempts")) for r in records),
    }
