"""Unified backend interface (section 4.2).

    generate(messages, params) -> GenerationResult(text, reasoning, usage, finish_reason, latency, logprobs, ...)

`params` keys:
    model          : checkpoint name
    max_tokens     : int
    decoding       : dict of sampling params (temperature, top_p, seed, top_k, ...)
    chat_template_kwargs : dict | None   (thinking switch, vLLM)
    logprobs       : bool (D5, default False)
    _mock_hint     : dict — only read by the mock backend (case label etc.); real backends ignore keys starting with "_"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


class ExecutionError(RuntimeError):
    """Execution failure (network, timeout, OOM, 5xx). The runner retries these; never format errors."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class GenerationResult:
    text: str
    reasoning: str | None = None
    usage: dict = field(default_factory=lambda: {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None})
    finish_reason: str | None = None
    latency: float | None = None
    logprobs: list | None = None  # reserved (D5); None unless params["logprobs"] is True
    model_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(ABC):
    name: str = "base"
    retry_backoff: float = 2.0  # seconds; exponential

    @abstractmethod
    def generate(self, messages: list[dict], params: dict) -> GenerationResult: ...

    def server_info(self) -> dict:
        """Version info for run_manifest.json (e.g. vLLM version)."""
        return {"backend": self.name}
