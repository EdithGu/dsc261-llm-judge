"""Registry + prompt assembly + config_hash."""

import pytest

from llm_judge.registry import (build_prompt_plan, get_dataset, get_model, list_configs, load_config,
                                resolve_thinking)
from llm_judge.schema import Case


def _case():
    return Case(case_id="c", question_id="q", dataset_id="medqa", split="test", task_type="multiple_choice",
                question="Which?", options={"A": "x", "B": "y"}, candidate_answer="B", reference_answer="A",
                label=False, candidate_type="distractor")


def test_configs_present(root):
    assert list_configs(root) == ["E1", "E2", "E3", "E4", "E5"]
    e5 = load_config(root, "E5")
    assert e5.output_mode == "ternary" and not e5.thinking and not e5.rubric


def test_render_hides_reference_and_label(root):
    plan = build_prompt_plan(root, get_dataset(root, "medqa"), get_model(root, "qwen3.8-27b"), load_config(root, "E3"))
    msgs = plan.render(_case())
    text = "\n".join(m["content"] for m in msgs)
    assert "A. x" in text and "B. y" in text and "Candidate answer:\nB" in text
    assert "{{" not in text
    assert "Evaluation rubric" in text
    assert "probability that YOUR VERDICT is right" in text  # confidence semantics stated in every prompt


def test_config_hash_changes_with_inputs(root):
    ds, m = get_dataset(root, "gsm8k"), get_model(root, "qwen3.8-27b")
    hashes = {c: build_prompt_plan(root, ds, m, load_config(root, c)).config_hash for c in list_configs(root)}
    assert len(set(hashes.values())) == 5
    again = build_prompt_plan(root, ds, m, load_config(root, "E1")).config_hash
    assert again == hashes["E1"]
    other_ds = build_prompt_plan(root, get_dataset(root, "medqa"), m, load_config(root, "E1")).config_hash
    assert other_ds != hashes["E1"]


def test_thinking_switch_qwen(root):
    m = get_model(root, "qwen3.8-27b")
    assert resolve_thinking(m, True).chat_template_kwargs == {"enable_thinking": True}
    assert resolve_thinking(m, False).chat_template_kwargs == {"enable_thinking": False}


def test_fixed_reasoning_model_marks_thinking_configs_na(root):
    m = get_model(root, "deepseek-v4-flash-0731")
    ds = get_dataset(root, "gsm8k")
    for c, na in (("E1", False), ("E2", True), ("E3", False), ("E4", True), ("E5", False)):
        assert build_prompt_plan(root, ds, m, load_config(root, c)).is_na == na


def test_unsupported_thinking(root):
    m = get_model(root, "cheap-api")
    assert not resolve_thinking(m, True).available
    assert resolve_thinking(m, False).available


def test_unknown_dataset(root):
    with pytest.raises(ValueError):
        get_dataset(root, "nope")
