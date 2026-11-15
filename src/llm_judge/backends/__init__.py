"""Backend factory. Adding a backend = one module + one line here; the runner never changes."""

from __future__ import annotations

from llm_judge.backends.base import Backend, ExecutionError, GenerationResult

BACKENDS = ("vllm", "api", "mock")


def make_backend(spec, backend_override: str | None = None, mock_opts: dict | None = None) -> Backend:
    kind = backend_override or spec.backend
    if kind == "mock":
        from llm_judge.backends.mock import MockBackend

        return MockBackend(**{**(spec.mock or {}), **(mock_opts or {})})
    if kind == "vllm":
        from llm_judge.backends.vllm import VLLMBackend

        return VLLMBackend.from_spec(spec)
    if kind == "api":
        from llm_judge.backends.api import APIBackend

        return APIBackend.from_spec(spec)
    raise ValueError(f"unknown backend {kind!r} (choose from {BACKENDS})")


__all__ = ["Backend", "ExecutionError", "GenerationResult", "make_backend", "BACKENDS"]
