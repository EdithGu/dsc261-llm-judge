import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def synth_data(tmp_path):
    """Full-size synthetic numeric dataset (pilot 40 / probe 100 / calib 60 / test 140) + manifest."""
    from llm_judge.synthetic import write_synthetic

    return write_synthetic(tmp_path / "data" / "cases_synth_math.jsonl", "numeric", "synth_math", seed=1)


@pytest.fixture
def synth_med(tmp_path):
    from llm_judge.synthetic import write_synthetic

    return write_synthetic(tmp_path / "data_med" / "cases_synth_med.jsonl", "multiple_choice", "synth_med", seed=2)


def make_row(case_id, status="valid", conf=0.9, err=False, label=True, verdict=None, qid=None, ctype=None):
    """Hand-built evaluation row (same shape as evaluate.build_rows output)."""
    if verdict is None and status == "valid":
        verdict = "correct" if (label != err) else "incorrect"
    return {
        "case_id": case_id,
        "question_id": qid or f"q-{case_id}",
        "split": "test",
        "label": label,
        "candidate_type": ctype or ("correct" if label else "arithmetic"),
        "status": status,
        "verdict": verdict,
        "confidence": conf if status in ("valid", "abstain") else None,
        "judge_error": err if status == "valid" else None,
        "parse_error": None,
        "missing": False,
    }


@pytest.fixture
def row():
    return make_row
