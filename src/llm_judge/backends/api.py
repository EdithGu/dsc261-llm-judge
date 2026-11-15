"""Hosted OpenAI-compatible API backend (DeepSeek replication, data group's `probe`).

Only standard OpenAI sampling keys are sent; anything provider-specific goes in `extra_body` in models.yaml.
"""

from __future__ import annotations

from llm_judge.backends.openai_compat import OpenAICompatBackend


class APIBackend(OpenAICompatBackend):
    name = "api"
    send_extra_top_level = False
