"""Backends: mock determinism/knobs, OpenAI-compatible request building + response parsing (no network)."""

import json

import pytest

from llm_judge.backends import ExecutionError, make_backend
from llm_judge.backends.mock import MockBackend
from llm_judge.backends.openai_compat import parse_chat_response, split_think
from llm_judge.backends.vllm import VLLMBackend
from llm_judge.backends.api import APIBackend
from llm_judge.parse import parse_output
from llm_judge.registry import get_model

MSG = [{"role": "user", "content": "judge this"}]


def _params(case_id="c1", label=True, mode="binary", attempt=1, tag="main", thinking=False):
    return {"model": "m", "max_tokens": 64, "decoding": {"temperature": 0}, "_attempt": attempt,
            "_mock_hint": {"case_id": case_id, "label": label, "output_mode": mode, "config_key": "E1",
                           "run_tag": tag, "thinking": thinking, "rubric": False}}


def test_mock_deterministic():
    m = MockBackend()
    a = m.generate(MSG, _params()).to_dict()
    b = m.generate(MSG, _params()).to_dict()
    assert a == b


def test_mock_outputs_parse_mostly_valid():
    m = MockBackend(invalid_rate=0.0, truncate_rate=0.0)
    for i in range(200):
        r = m.generate(MSG, _params(case_id=f"c{i}", label=i % 2 == 0))
        assert parse_output(r.text, r.finish_reason, "binary").status == "valid"


def test_mock_accuracy_knob():
    m = MockBackend(accuracy=0.95, invalid_rate=0, truncate_rate=0)
    right = 0
    for i in range(400):
        lab = i % 2 == 0
        r = m.generate(MSG, _params(case_id=f"c{i}", label=lab))
        j = parse_output(r.text, r.finish_reason, "binary")
        right += j.verdict_bool == lab
    assert right / 400 > 0.9


def test_mock_exec_failures_and_retry_attempts():
    m = MockBackend(exec_fail_rate=1.0)
    with pytest.raises(ExecutionError):
        m.generate(MSG, _params())


def test_mock_nondeterminism_only_for_replicates():
    m = MockBackend(nondeterminism=1.0)
    base = [m.generate(MSG, _params(case_id=f"c{i}")).text for i in range(50)]
    again = [m.generate(MSG, _params(case_id=f"c{i}")).text for i in range(50)]
    rep = [m.generate(MSG, _params(case_id=f"c{i}", tag="r2")).text for i in range(50)]
    assert base == again and base != rep


def test_mock_ternary_abstains_sometimes():
    m = MockBackend(abstain_rate=0.5, invalid_rate=0, truncate_rate=0)
    st = [parse_output(m.generate(MSG, _params(case_id=f"c{i}", mode="ternary")).text, "stop", "ternary").status
          for i in range(100)]
    assert "abstain" in st and "valid" in st


def test_mock_unknown_option():
    with pytest.raises(ValueError):
        MockBackend(nonsense=1)


def test_split_think():
    assert split_think("<think>abc</think>\nANSWER") == ("ANSWER", "abc")
    assert split_think("abc</think>ANSWER") == ("ANSWER", "abc")
    assert split_think("<think>still thinking") == ("", "still thinking")
    assert split_think("plain") == ("plain", None)


def test_parse_chat_response_vllm_reasoning_field():
    data = {"model": "Qwen/x", "choices": [{"message": {"content": '{"verdict": "correct", "confidence": 0.9}',
                                                        "reasoning_content": "hmm"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 15}}}
    r = parse_chat_response(data)
    assert r.reasoning == "hmm" and r.usage["reasoning_tokens"] == 15 and r.model_version == "Qwen/x"
    assert r.logprobs is None


def test_parse_chat_response_no_choices():
    with pytest.raises(ExecutionError):
        parse_chat_response({"choices": []})


def test_request_body_vllm_vs_api():
    params = {"model": "m", "max_tokens": 32, "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1},
              "chat_template_kwargs": {"enable_thinking": True}, "logprobs": False, "_mock_hint": {"label": True}}
    v = VLLMBackend("http://x/v1").build_body(MSG, params)
    assert v["top_k"] == 20 and v["chat_template_kwargs"] == {"enable_thinking": True} and v["seed"] == 1
    assert "_mock_hint" not in json.dumps(v)
    a = APIBackend("http://x/v1").build_body(MSG, params)
    assert "top_k" not in a and "chat_template_kwargs" not in a and a["temperature"] == 0.6


def test_logprobs_reserved_but_off_by_default(root):
    for mid in ("qwen3.8-27b", "deepseek-v4-flash-0731", "mock"):
        assert get_model(root, mid).logprobs is False


def test_make_backend_mock_override(root):
    spec = get_model(root, "qwen3.8-27b", "mock")
    assert spec.id == "qwen3.8-27b@mock"
    assert isinstance(make_backend(spec), MockBackend)
