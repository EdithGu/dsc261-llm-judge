"""vLLM OpenAI-compatible server backend (main experiment).

Start the server e.g.:
    vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
        --reasoning-parser qwen3 --max-num-seqs 16 --seed 0
Fix the vLLM version and batch size for formal runs (D1); both are written to run_manifest.json.
"""

from __future__ import annotations

import json
import urllib.request

from llm_judge.backends.openai_compat import OpenAICompatBackend


class VLLMBackend(OpenAICompatBackend):
    name = "vllm"
    send_extra_top_level = True  # top_k, min_p, chat_template_kwargs are accepted top-level by vLLM

    def server_info(self) -> dict:
        info = {"backend": self.name, "base_url": self.base_url, "vllm_version": None}
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            with urllib.request.urlopen(root + "/version", timeout=5) as resp:
                info["vllm_version"] = json.loads(resp.read().decode("utf-8")).get("version")
        except Exception as e:  # server down / older vLLM — record and move on
            info["vllm_version_error"] = str(e)[:200]
        return info
