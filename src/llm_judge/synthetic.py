"""Synthetic schema-valid datasets for tests and demos (NOT research data).

    python -m llm_judge.synthetic --task numeric --out data/synth_math/cases_synth_math.jsonl
    python -m llm_judge.synthetic --task multiple_choice --out data/synth_med/cases_synth_med.jsonl

Default split sizes mirror the real design: pilot 40, probe 100, calib 60, test 140 (half correct candidates).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm_judge.schema import make_manifest
from llm_judge.utils import write_json, write_jsonl

DEFAULT_SPLITS = {"pilot": 40, "probe": 100, "calib": 60, "test": 140}
NUM_ERRORS = ("arithmetic", "omitted_step", "unit_error")
MC_ERRORS = ("distractor", "near_miss")


def _numeric_case(rng: random.Random, i: int, split: str, correct: bool, dataset_id: str) -> dict:
    a, b, c = rng.randint(2, 40), rng.randint(2, 15), rng.randint(1, 30)
    q = (f"Shop {split[:2].upper()}-{i} sells {a} boxes of pencils. Each box holds {b} pencils. "  # unique text
         f"After {c} pencils are given away, how many pencils are left?")
    ans = a * b - c
    if correct:
        cand, ctype = f"{a} x {b} = {a * b}. {a * b} - {c} = {ans}. #### {ans}", "correct"
    else:
        ctype = NUM_ERRORS[i % len(NUM_ERRORS)]
        if ctype == "arithmetic":
            wrong = a * b + rng.choice([-1, 1]) * rng.randint(1, 9)
            cand = f"{a} x {b} = {wrong}. {wrong} - {c} = {wrong - c}. #### {wrong - c}"
        elif ctype == "omitted_step":
            cand = f"{a} x {b} = {a * b}. #### {a * b}"
        else:
            cand = f"{a} x {b} = {a * b} pencils, i.e. {a * b // 12} dozen; minus {c}: #### {a * b // 12 - c if a * b // 12 - c != ans else ans + 1}"
    return {
        "case_id": f"{dataset_id}-{split}-{i:04d}",
        "question_id": f"{dataset_id}-q{split[:2]}{i:04d}",
        "dataset_id": dataset_id,
        "split": split,
        "task_type": "numeric",
        "question": q,
        "options": None,
        "candidate_answer": cand,
        "reference_answer": str(ans),
        "label": correct,
        "candidate_type": ctype,
        "meta": {"synthetic": True},
    }


def _mc_case(rng: random.Random, i: int, split: str, correct: bool, dataset_id: str) -> dict:
    age = rng.randint(18, 85)
    keys = ["A", "B", "C", "D", "E"]
    opts = {k: f"Finding {k.lower()}-{rng.randint(100, 999)}" for k in keys}
    ref = rng.choice(keys)
    if correct:
        cand, ctype = ref, "correct"
    else:
        cand = rng.choice([k for k in keys if k != ref])
        ctype = MC_ERRORS[i % len(MC_ERRORS)]
    q = (f"Case {split}-{i}: A {age}-year-old patient presents with symptom pattern #{rng.randint(1000, 9999)}. "
         f"Which of the following is the most likely explanation?")
    return {
        "case_id": f"{dataset_id}-{split}-{i:04d}",
        "question_id": f"{dataset_id}-q{split[:2]}{i:04d}",
        "dataset_id": dataset_id,
        "split": split,
        "task_type": "multiple_choice",
        "question": q,
        "options": opts,
        "candidate_answer": cand,
        "reference_answer": ref,
        "label": correct,
        "candidate_type": ctype,
        "meta": {"synthetic": True},
    }


def make_synthetic(task: str, dataset_id: str, splits: dict | None = None, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    make = _numeric_case if task == "numeric" else _mc_case
    rows = []
    for split, n in (splits or DEFAULT_SPLITS).items():
        for i in range(n):
            rows.append(make(rng, i, split, i % 2 == 0, dataset_id))
    return rows


def write_synthetic(out: Path, task: str, dataset_id: str, splits: dict | None = None, seed: int = 0) -> Path:
    out = Path(out)
    write_jsonl(out, make_synthetic(task, dataset_id, splits, seed))
    write_json(out.parent / "manifest.json", make_manifest(out, seed, {"synthetic": True}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["numeric", "multiple_choice"], required=True)
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    for s, n in DEFAULT_SPLITS.items():
        ap.add_argument(f"--{s}", type=int, default=n)
    a = ap.parse_args(argv)
    did = a.dataset_id or ("synth_math" if a.task == "numeric" else "synth_med")
    p = write_synthetic(Path(a.out), a.task, did, {s: getattr(a, s) for s in DEFAULT_SPLITS}, a.seed)
    print(f"wrote {p} (+ manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
