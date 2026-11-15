"""[D] Threshold calibration on the calib split only.

Ordering: valid judgments by confidence (desc); ties broken by a label-independent deterministic key
hash(case_id + salt) (asc). Coverage denominator = ALL cases, including invalid ones (D2).

Retention rule for a target coverage t, with k = round(t * N):
    keep  if conf > tau   or   (conf == tau and tie_hash < hash_cutoff)
where (tau, hash_cutoff) are taken from the k-th ranked calib judgment. On calib this keeps exactly k.
If k > number of valid judgments the target is "unreachable" and reported as such (never relaxed).
"""

from __future__ import annotations

import math
from pathlib import Path

from llm_judge.utils import now_iso, read_json, sha256_text, canonical_json, stable_int, write_json

DEFAULT_SALT = "dsc261-tiebreak-v1"
TARGETS = (0.50, 0.75, 0.90)


def tie_hash(case_id: str, salt: str = DEFAULT_SALT) -> int:
    return stable_int(f"{case_id}{salt}")


def rank_key(conf: float, case_id: str, salt: str = DEFAULT_SALT) -> tuple:
    return (-conf, tie_hash(case_id, salt))


def ranked_valid(rows: list[dict], salt: str = DEFAULT_SALT) -> list[dict]:
    """rows: dicts with case_id, status, confidence. Returns valid rows in retention order."""
    valid = [r for r in rows if r["status"] == "valid"]
    return sorted(valid, key=lambda r: rank_key(r["confidence"], r["case_id"], salt))


def target_k(target: float, n_total: int) -> int:
    return int(math.floor(target * n_total + 0.5))  # round half up, no banker's rounding


def fit_thresholds(calib_rows: list[dict], targets=TARGETS, salt: str = DEFAULT_SALT) -> dict:
    n_total = len(calib_rows)
    ranked = ranked_valid(calib_rows, salt)
    n_valid = len(ranked)
    out = {}
    for t in targets:
        k = target_k(t, n_total)
        entry = {
            "target": t,
            "k_calib": k,
            "n_calib": n_total,
            "n_valid_calib": n_valid,
            "max_coverage_calib": n_valid / n_total if n_total else 0.0,
        }
        if n_total == 0 or k > n_valid:
            entry.update({"reachable": False, "tau": None, "hash_cutoff": None, "calib_coverage": None})
        elif k == 0:
            entry.update({"reachable": True, "tau": math.inf, "hash_cutoff": 0, "calib_coverage": 0.0})
        else:
            kth = ranked[k - 1]
            entry.update({
                "reachable": True,
                "tau": float(kth["confidence"]),
                "hash_cutoff": str(tie_hash(kth["case_id"], salt) + 1),  # str: exceeds JSON-safe int range
                "calib_coverage": k / n_total,
            })
        out[f"{t:.2f}"] = entry
    return out


def retained(conf: float | None, case_id: str, thr: dict, salt: str = DEFAULT_SALT) -> bool:
    if not thr.get("reachable") or conf is None:
        return False
    tau = thr["tau"]
    if tau == math.inf:
        return False
    if conf > tau:
        return True
    return conf == tau and tie_hash(case_id, salt) < int(thr["hash_cutoff"])


# --------------------------------------------------------------------------- frozen thresholds file

def threshold_key(dataset_id: str, model_id: str, config_id: str) -> str:
    return f"{dataset_id}/{model_id}/{config_id}"


class ThresholdFreezeError(RuntimeError):
    pass


def freeze_thresholds(path: Path, key: str, config_hash: str, thresholds: dict, salt: str,
                      calib_case_ids: list[str], force: bool = False) -> dict:
    """Write once. Re-calibrating an existing key with the same config_hash + calib set must give identical
    thresholds (returned unchanged); anything else is refused unless force=True."""
    path = Path(path)
    data = read_json(path) if path.exists() else {"entries": {}}
    entry = {
        "config_hash": config_hash,
        "salt": salt,
        "calib_ids_sha256": sha256_text("\n".join(sorted(calib_case_ids)))[:16],
        "thresholds": thresholds,
    }
    old = data["entries"].get(key)
    if old is not None and not force:
        same = {k: old.get(k) for k in entry} == entry
        if same:
            return old
        raise ThresholdFreezeError(
            f"thresholds for {key} are already frozen in {path} with different content "
            f"(frozen config_hash={old.get('config_hash')}, now {config_hash}). "
            f"Thresholds are never modified after freezing; if the setup really changed, log it in "
            f"decision_log.md and re-run everything (section 6.4)."
        )
    entry["frozen_at"] = now_iso()
    entry["entry_sha256"] = sha256_text(canonical_json({k: entry[k] for k in ("config_hash", "salt", "calib_ids_sha256", "thresholds")}))[:16]
    data["entries"][key] = entry
    write_json(path, data)
    return entry


def load_frozen(path: Path, key: str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)["entries"].get(key)
