"""[E] Abstention policies, applied on the test split only.

rows: list of dicts with keys case_id, status, verdict ('correct'|'incorrect'|'abstain'|None), confidence,
      label (bool), judge_error (bool | None; None if not a valid binary judgment).

    no_abstention   keep every valid judgment
    confidence      apply frozen (tau, hash_cutoff) -> realised coverage c and kept count k
    random          keep k of the valid judgments uniformly at random, R repetitions (fixed seed);
                    expectation == full-coverage error rate of the valid pool (analytic cross-check)
    direct (E5)     keep valid non-abstain judgments; report observed coverage
"""

from __future__ import annotations

import numpy as np

from llm_judge.calibrate import DEFAULT_SALT, retained


def _errors(rows: list[dict]) -> np.ndarray:
    return np.array([bool(r["judge_error"]) for r in rows], dtype=float)


def policy_summary(rows: list[dict], keep: list[bool], n_total: int) -> dict:
    """Coverage / selective risk / operational metrics for a retention mask over `rows`."""
    kept = [r for r, k in zip(rows, keep) if k]
    valid = [r for r in rows if r["status"] == "valid"]
    k = len(kept)
    n_err_kept = sum(bool(r["judge_error"]) for r in kept)
    n_err_valid = sum(bool(r["judge_error"]) for r in valid)
    abstained_valid = [r for r, kk in zip(rows, keep) if r["status"] == "valid" and not kk]
    caught = sum(bool(r["judge_error"]) for r in abstained_valid)
    neg = [r for r in kept if not r["label"]]
    pos = [r for r in kept if r["label"]]
    return {
        "k": k,
        "n_total": n_total,
        "coverage": k / n_total if n_total else 0.0,
        "selective_risk": n_err_kept / k if k else None,
        "n_errors_kept": n_err_kept,
        "referral_volume": n_total - k,  # everything not auto-accepted goes to a human (incl. invalid outputs)
        "referral_breakdown": {
            "abstained_valid": len(abstained_valid),
            "abstain_verdict": sum(r["status"] == "abstain" for r in rows),
            "invalid": sum(r["status"] not in ("valid", "abstain") for r in rows),
        },
        "error_capture_rate": caught / n_err_valid if n_err_valid else None,
        "false_accept_rate": (sum(r["verdict"] == "correct" for r in neg) / len(neg)) if neg else None,
        "false_reject_rate": (sum(r["verdict"] == "incorrect" for r in pos) / len(pos)) if pos else None,
        "n_kept_incorrect_candidates": len(neg),
        "n_kept_correct_candidates": len(pos),
    }


def no_abstention(rows: list[dict]) -> list[bool]:
    return [r["status"] == "valid" for r in rows]


def confidence_policy(rows: list[dict], thr: dict, salt: str = DEFAULT_SALT) -> list[bool]:
    return [r["status"] == "valid" and retained(r["confidence"], r["case_id"], thr, salt) for r in rows]


def direct_abstention(rows: list[dict]) -> list[bool]:
    return [r["status"] == "valid" and r["verdict"] in ("correct", "incorrect") for r in rows]


def random_baseline(errors_pool: np.ndarray, k: int, R: int = 10_000, seed: int = 0) -> dict:
    """Uniformly keep k of the pool, R times. Returns simulated + analytic expectation."""
    errors_pool = np.asarray(errors_pool, dtype=float)
    n = len(errors_pool)
    analytic = float(errors_pool.mean()) if n else None
    if k <= 0 or k > n:
        return {"k": k, "pool": n, "R": R, "seed": seed, "feasible": False, "analytic_expected_risk": analytic}
    rng = np.random.default_rng(seed)
    # argpartition of uniform keys == uniform k-subset without replacement, vectorised over R
    keys = rng.random((R, n))
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
    risks = errors_pool[idx].mean(axis=1)
    return {
        "k": k,
        "pool": n,
        "R": R,
        "seed": seed,
        "feasible": True,
        "mean_risk": float(risks.mean()),
        "sd_risk": float(risks.std(ddof=1)) if R > 1 else 0.0,
        "q025": float(np.quantile(risks, 0.025)),
        "q975": float(np.quantile(risks, 0.975)),
        "analytic_expected_risk": analytic,
        "abs_dev_from_analytic": abs(float(risks.mean()) - analytic),
    }


def valid_error_pool(rows: list[dict]) -> np.ndarray:
    return _errors([r for r in rows if r["status"] == "valid"])
