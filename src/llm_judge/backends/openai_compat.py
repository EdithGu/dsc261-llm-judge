"""OpenAI-compatible /chat/completions client (stdlib only). Shared by the vllm and api backends."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

from llm_judge.backends.base import Backend, ExecutionError, GenerationResult
from llm_judge.registry import STANDARD_SAMPLING_KEYS

RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


def split_think(content: str) -> tuple[str, str | None]:
    """Separate a <think>...</think> block from the answer (Qwen-style output without a reasoning parser)."""
    if content is None:
        return "", None
    if "</think>" in content:
        head, tail = content.split("</think>", 1)
        reasoning = head.split("<think>", 1)[1] if "<think>" in head else head
        return tail.strip(), reasoning.strip()
    if "<think>" in content:  # opened but never closed -> truncated inside reasoning
        before, reasoning = content.split("<think>", 1)
        return before.strip(), reasoning.strip()
    return content, None


def parse_chat_response(data: dict, want_logprobs: bool = False) -> GenerationResult:
    choices = data.get("choices") or []
    if not choices:
        raise ExecutionError(f"response has no choices: {str(data)[:200]}", retryable=True)
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning is None:
        content, reasoning = split_think(content)
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return GenerationResult(
        text=content,
        reasoning=reasoning,
        usage={
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens") if details else usage.get("reasoning_tokens"),
        },
        finish_reason=choice.get("finish_reason"),
        logprobs=choice.get("logprobs") if want_logprobs else None,
        model_version=data.get("model"),
    )


class OpenAICompatBackend(Backend):
    name = "openai_compat"
    send_extra_top_level = False  # vLLM accepts top_k/min_p/... top-level; hosted APIs usually don't

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300, extra_body: dict | None = None):
        if not base_url:
            raise ValueError("base_url is required for an OpenAI-compatible backend")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}

    @classmethod
    def from_spec(cls, spec) -> "OpenAICompatBackend":
        key = os.environ.get(spec.api_key_env) if spec.api_key_env else None
        if spec.api_key_env and not key:
            raise ValueError(f"environment variable {spec.api_key_env} is not set (needed by model {spec.id})")
        return cls(spec.base_url, key, spec.request_timeout, spec.extra_body)

    def build_body(self, messages: list[dict], params: dict) -> dict:
        body: dict = {"model": params["model"], "messages": messages, "max_tokens": params["max_tokens"]}
        extra: dict = {}
        for k, v in (params.get("decoding") or {}).items():
            if v is None:
                continue
            (body if k in STANDARD_SAMPLING_KEYS else extra)[k] = v
        if params.get("chat_template_kwargs"):
            extra["chat_template_kwargs"] = params["chat_template_kwargs"]
        if params.get("logprobs"):
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if self.send_extra_top_level:
            body.update(extra)
        body.update(self.extra_body)
        return body

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise ExecutionError(f"HTTP {e.code}: {detail}", retryable=e.code in RETRYABLE_HTTP) from e
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise ExecutionError(f"connection error: {e}", retryable=True) from e
        except json.JSONDecodeError as e:
            raise ExecutionError(f"non-JSON response: {e}", retryable=True) from e

    def generate(self, messages: list[dict], params: dict) -> GenerationResult:
        data = self._post("/chat/completions", self.build_body(messages, params))
        return parse_chat_response(data, want_logprobs=bool(params.get("logprobs")))

    def server_info(self) -> dict:
        return {"backend": self.name, "base_url": self.base_url}
