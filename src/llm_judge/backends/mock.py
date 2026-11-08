"""Mock backend: no GPU, no network. Produces well-formed (and some deliberately malformed) judge outputs.

Modes
  noisy_oracle : peeks at the case label via params["_mock_hint"] and is right with probability `accuracy`
                 (+ bonuses for thinking / rubric). Confidence is higher when it is right, so confidence-based
                 abstention beats random abstention — enough for smoke outputs to look sensible.
  random       : uniform verdicts, uniform confidence.

Everything is a deterministic function of (seed, case_id, config, attempt, run tag), so reruns are reproducible.
It validates that data flows through the pipeline; it says nothing about real judge quality.
"""

from __future__ import annotations

import json
import random

from llm_judge.backends.base import Backend, ExecutionError, GenerationResult
from llm_judge.utils import stable_int

DEFAULTS = {
    "mode": "noisy_oracle",
    "accuracy": 0.80,
    "thinking_bonus": 0.05,
    "rubric_bonus": 0.03,
    "invalid_rate": 0.03,
    "truncate_rate": 0.01,
    "exec_fail_rate": 0.0,
    "abstain_rate": 0.10,
    "tie_rate": 0.40,
    "nondeterminism": 0.0,
    "seed": 0,
}


class MockBackend(Backend):
    name = "mock"
    retry_backoff = 0.0

    def __init__(self, **opts):
        unknown = set(opts) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown mock options {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
        self.opts = {**DEFAULTS, **opts}

    def server_info(self) -> dict:
        return {"backend": self.name, "mock_options": dict(self.opts)}

    def _rng(self, *parts) -> random.Random:
        return random.Random(stable_int("|".join(str(p) for p in (self.opts["seed"], *parts))))

    def generate(self, messages: list[dict], params: dict) -> GenerationResult:
        o = self.opts
        hint = params.get("_mock_hint") or {}
        case_id = hint.get("case_id") or str(stable_int(json.dumps(messages)))
        cfg = hint.get("config_key", "")
        thinking = bool(hint.get("thinking"))
        rubric = bool(hint.get("rubric"))
        ternary = hint.get("output_mode") == "ternary"
        attempt = int(params.get("_attempt", 1))
        tag = hint.get("run_tag", "main")

        # execution failures (retried by the runner)
        if o["exec_fail_rate"] > 0 and self._rng("exec", case_id, cfg, attempt, tag).random() < o["exec_fail_rate"]:
            raise ExecutionError("mock: simulated execution failure", retryable=True)

        rng = self._rng("gen", case_id, cfg)
        if tag != "main" and o["nondeterminism"] > 0 and self._rng("nd", case_id, cfg, tag).random() < o["nondeterminism"]:
            rng = self._rng("gen", case_id, cfg, tag)  # replicate diverges on this case

        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        reasoning = None
        reasoning_tokens = 0
        if thinking:
            reasoning_tokens = rng.randint(150, 900)
            reasoning = f"[mock reasoning: {reasoning_tokens} tokens of deliberation about case {case_id}]"

        u = rng.random()
        if u < o["truncate_rate"]:
            text = '{"verdict": "corr'
            return self._result(text, reasoning, prompt_tokens, params["max_tokens"], reasoning_tokens, "length")
        if u < o["truncate_rate"] + o["invalid_rate"]:
            kind = rng.choice(["prose", "bad_verdict", "bad_conf", "no_conf"])
            text = {
                "prose": "I believe the candidate answer is probably right.",
                "bad_verdict": '{"verdict": "partially correct", "confidence": 0.7}',
                "bad_conf": '{"verdict": "correct", "confidence": 1.7}',
                "no_conf": '{"verdict": "incorrect"}',
            }[kind]
            return self._result(text, reasoning, prompt_tokens, 12, reasoning_tokens, "stop")

        label = hint.get("label")
        if o["mode"] == "random" or label is None:
            verdict_bool = rng.random() < 0.5
            conf = rng.uniform(0.5, 1.0)
            right = None
        else:
            acc = min(0.99, o["accuracy"] + (o["thinking_bonus"] if thinking else 0) + (o["rubric_bonus"] if rubric else 0))
            right = rng.random() < acc
            verdict_bool = bool(label) if right else not bool(label)
            conf = rng.betavariate(8, 2) if right else rng.betavariate(5, 3)
            conf = 0.5 + 0.5 * conf
        if rng.random() < o["tie_rate"]:
            conf = rng.choice([0.9, 0.95])
        conf = round(min(1.0, max(0.0, conf)), 2)

        verdict = "correct" if verdict_bool else "incorrect"
        if ternary:
            p_abs = o["abstain_rate"] * (2.0 if right is False else 0.5 if right else 1.0)
            if rng.random() < p_abs:
                verdict = "abstain"
        text = json.dumps({"verdict": verdict, "confidence": conf})
        if rng.random() < 0.1:  # sometimes wrap in a code fence, as real models do
            text = f"```json\n{text}\n```"
        return self._result(text, reasoning, prompt_tokens, 16, reasoning_tokens, "stop")

    @staticmethod
    def _result(text, reasoning, pt, ct, rt, finish) -> GenerationResult:
        return GenerationResult(
            text=text,
            reasoning=reasoning,
            usage={"prompt_tokens": pt, "completion_tokens": ct + rt, "reasoning_tokens": rt or None},
            finish_reason=finish,
            model_version="mock-judge-v1",
        )
