"""[B] runner: cache/resume, execution-only retries, lock check, manifest."""

import pytest

from llm_judge.backends.mock import MockBackend
from llm_judge.registry import build_prompt_plan, get_dataset, get_model, load_config
from llm_judge.runner import LockError, load_records, run_inference
from llm_judge.schema import load_cases
from llm_judge.utils import read_json, write_json


@pytest.fixture
def setup(root, fixtures):
    cases = load_cases(fixtures / "gsm8k" / "cases_gsm8k.jsonl")
    model = get_model(root, "mock")
    plan = build_prompt_plan(root, get_dataset(root, "gsm8k"), model, load_config(root, "E1"))
    return cases, model, plan


def test_cache_and_resume(tmp_path, setup, root):
    cases, model, plan = setup
    be = MockBackend()
    s1 = run_inference(cases[:4], plan, be, model, tmp_path, "smoke", root=root, sleep=lambda s: None)
    assert s1.n_generated == 4 and s1.n_cached == 0
    first = load_records(s1.run_dir)
    s2 = run_inference(cases, plan, be, model, tmp_path, "smoke", root=root, sleep=lambda s: None)
    assert s2.n_cached == 4 and s2.n_generated == 6
    after = load_records(s2.run_dir)
    for cid, rec in first.items():
        assert after[cid] == rec  # never overwritten
    man = read_json(s2.run_dir / "run_manifest.json")
    assert len(man["sessions"]) == 2 and man["config_hash"] == plan.config_hash
    assert man["sessions"][0]["batch_size"] == model.batch_size
    assert (s2.run_dir / "config.json").exists()


def test_retry_only_execution_failures(tmp_path, setup, root):
    cases, model, plan = setup
    sleeps = []
    be = MockBackend(exec_fail_rate=0.5, invalid_rate=0.5)
    s = run_inference(cases, plan, be, model, tmp_path, "smoke", max_retries=30, root=root, sleep=sleeps.append)
    recs = load_records(s.run_dir)
    assert s.n_retried > 0
    assert all(r["status"] == "ok" for r in recs.values())  # deterministic; 0.5^31 per case of total failure
    # format errors are recorded as-is, never retried: each record has at most the failed exec attempts
    for r in recs.values():
        assert all("simulated execution failure" in a["error"] for a in r["attempts"])


def test_exec_failure_after_max_retries(tmp_path, setup, root):
    cases, model, plan = setup
    be = MockBackend(exec_fail_rate=1.0)
    s = run_inference(cases[:2], plan, be, model, tmp_path, "smoke", max_retries=2, root=root, sleep=lambda s: None)
    recs = load_records(s.run_dir)
    assert s.n_exec_failure == 2
    for r in recs.values():
        assert r["status"] == "exec_failure" and r["text"] is None and len(r["attempts"]) == 3
    # cached failures are skipped unless retry_failed
    s2 = run_inference(cases[:2], plan, MockBackend(), model, tmp_path, "smoke", root=root, sleep=lambda s: None)
    assert s2.n_cached == 2
    s3 = run_inference(cases[:2], plan, MockBackend(), model, tmp_path, "smoke", retry_failed=True, root=root,
                       sleep=lambda s: None)
    assert s3.n_generated == 2
    assert all(r["status"] == "ok" for r in load_records(s3.run_dir).values())


def test_formal_run_requires_lock(tmp_path, setup, root):
    cases, model, plan = setup
    with pytest.raises(LockError, match="no freeze lock"):
        run_inference(cases, plan, MockBackend(), model, tmp_path, "run", root=root)


def test_lock_mismatch_refused(tmp_path, setup, root):
    from llm_judge.parse import parser_hash

    cases, model, plan = setup
    lock = tmp_path / "frozen" / "lock.json"
    key = f"{plan.dataset_id}/{plan.model_id}/{plan.config.id}"
    good = {"parser_hash": parser_hash(), "entries": {key: {"config_hash": plan.config_hash,
                                                             "components": plan.component_hashes()}}}
    write_json(lock, good)
    run_inference(cases[:1], plan, MockBackend(), model, tmp_path, "run", root=root)  # passes

    bad = {"parser_hash": parser_hash(), "entries": {key: {"config_hash": "000000000000",
                                                            "components": {**plan.component_hashes(), "prompt": "x"}}}}
    write_json(lock, bad)
    with pytest.raises(LockError, match="prompt"):
        run_inference(cases, plan, MockBackend(), model, tmp_path, "run", root=root)

    write_json(lock, {**good, "parser_hash": "deadbeef"})
    with pytest.raises(LockError, match="parser"):
        run_inference(cases, plan, MockBackend(), model, tmp_path, "run", root=root)


def test_probe_is_not_lock_checked(tmp_path, setup, root):
    cases, model, plan = setup
    run_inference(cases[:2], plan, MockBackend(), model, tmp_path, "probe", root=root)


def test_na_config_writes_no_records(tmp_path, root, fixtures):
    cases = load_cases(fixtures / "gsm8k" / "cases_gsm8k.jsonl")
    model = get_model(root, "deepseek-v4-flash-0731", "mock")
    plan = build_prompt_plan(root, get_dataset(root, "gsm8k"), model, load_config(root, "E2"))
    s = run_inference(cases, plan, MockBackend(), model, tmp_path, "probe", root=root)
    assert s.na_reason and not load_records(s.run_dir)
