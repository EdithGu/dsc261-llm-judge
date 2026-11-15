"""[G] Statistical inference: paired bootstrap (shared indices), Holm, McNemar exact, Wilson CI.

Bootstrap unit = question_id (cases of the same question are resampled together); units are stratified by the
pattern of candidate labels they contain (e.g. "T", "F", or "FT"), which reduces to stratifying by candidate
correctness when each question contributes one case. The SAME index matrix is used for every config of a
domain, so all between-config differences are paired (section 7.4).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def bootstrap_indices(case_ids: list[str], question_ids: list[str], labels: list[bool], B: int, seed: int) -> np.ndarray:
    """(B, N) matrix of row indices into the case list, resampled by question within label strata.

    Each resample has exactly N rows when every question has the same number of cases within a stratum
    (the usual design); otherwise row counts vary slightly and rows are padded by repeating draws.
    """
    n = len(case_ids)
    if n == 0:
        return np.zeros((B, 0), dtype=np.int64)
    # deterministic unit ordering independent of input order
    units: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(question_ids):
        units[q].append(i)
    unit_ids = sorted(units)
    strata: dict[str, list[str]] = defaultdict(list)
    for u in unit_ids:
        units[u].sort(key=lambda i: case_ids[i])
        pattern = "".join(sorted("T" if labels[i] else "F" for i in units[u]))
        strata[pattern].append(u)

    rng = np.random.default_rng(seed)
    # fast path: within each stratum every unit has the same number of cases -> fully vectorised
    if all(len({len(units[u]) for u in us}) == 1 for us in strata.values()):
        blocks = []
        for pattern in sorted(strata):
            us = strata[pattern]
            table = np.array([units[u] for u in us], dtype=np.int64)  # (n_units, size)
            draw = rng.integers(0, len(us), size=(B, len(us)))
            blocks.append(table[draw].reshape(B, -1))
        return np.concatenate(blocks, axis=1)

    out = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        rows: list[int] = []
        for pattern in sorted(strata):
            us = strata[pattern]
            draw = rng.integers(0, len(us), size=len(us))
            for j in draw:
                rows.extend(units[us[j]])
        if len(rows) >= n:
            out[b] = rows[:n]
        else:  # pragma: no cover - only with unequal cluster sizes
            out[b] = rows + list(rng.choice(rows, n - len(rows)))
    return out


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float | None, float | None]:
    s = samples[~np.isnan(samples)]
    if s.size == 0:
        return None, None
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def bootstrap_p_two_sided(samples: np.ndarray, null: float = 0.0) -> float | None:
    """Two-sided bootstrap p-value: 2 * min(P(d <= null), P(d >= null)), with +1 smoothing."""
    s = samples[~np.isnan(samples)]
    if s.size == 0:
        return None
    lo = (np.sum(s <= null) + 1) / (s.size + 1)
    hi = (np.sum(s >= null) + 1) / (s.size + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def summarize(point: float | None, samples: np.ndarray, with_p: bool = False) -> dict:
    if point is not None and math.isnan(point):
        point = None  # e.g. nothing retained on test -> selective risk undefined
    lo, hi = percentile_ci(samples)
    d = {
        "estimate": point,
        "ci95": [lo, hi],
        "n_boot_valid": int(np.sum(~np.isnan(samples))),
        "boot_se": float(np.nanstd(samples, ddof=1)) if np.sum(~np.isnan(samples)) > 1 else None,
    }
    if with_p:
        d["p_boot"] = bootstrap_p_two_sided(samples)
    return d


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values. None entries (not testable) are treated as p = 1."""
    items = [(k, 1.0 if p is None else float(p)) for k, p in pvalues.items()]
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return {k: adj[k] for k, _ in items}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def ratio_rows(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
