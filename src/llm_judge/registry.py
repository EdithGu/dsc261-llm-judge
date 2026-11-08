"""Registries (datasets / models / experiment configs) and module [A]: prompt assembly + config_hash."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_judge.utils import hash_obj, read_text_normalized

# Keys every OpenAI-compatible endpoint understands. Everything else in a decoding block is "extra".
STANDARD_SAMPLING_KEYS = {"temperature", "top_p", "seed", "presence_penalty", "frequency_penalty", "stop", "n"}


class RegistryError(ValueError):
    pass


# --------------------------------------------------------------------------- datasets

@dataclass
class DatasetSpec:
    id: str
    task_type: str
    domain: str
    rubric: str | None
    adapter: str | None = None
    data_path: str | None = None


def load_datasets(root: Path) -> dict[str, DatasetSpec]:
    raw = _load_yaml(root / "registry" / "datasets.yaml")
    out = {}
    for did, d in raw.items():
        try:
            out[did] = DatasetSpec(
                id=did,
                task_type=d["task_type"],
                domain=d["domain"],
                rubric=d.get("rubric"),
                adapter=d.get("adapter"),
                data_path=d.get("data_path"),
            )
        except KeyError as e:
            raise RegistryError(f"datasets.yaml: {did} missing field {e}") from e
    return out


def get_dataset(root: Path, dataset_id: str, task_type_hint: str | None = None) -> DatasetSpec:
    ds = load_datasets(root)
    if dataset_id in ds:
        return ds[dataset_id]
    raise RegistryError(
        f"dataset_id {dataset_id!r} is not registered in registry/datasets.yaml "
        f"(known: {', '.join(sorted(ds))}). Add an entry with task_type/domain/rubric."
    )


# --------------------------------------------------------------------------- models

@dataclass
class ModelSpec:
    id: str
    backend: str
    checkpoint: str
    decoding: dict
    thinking: dict
    max_tokens: int
    context_length: int | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    checksum: str | None = None
    batch_size: int = 8
    max_retries: int = 3
    request_timeout: float = 300
    logprobs: bool = False
    extra_body: dict = field(default_factory=dict)
    mock: dict = field(default_factory=dict)


def load_models(root: Path) -> dict[str, ModelSpec]:
    raw = _load_yaml(root / "registry" / "models.yaml")
    out = {}
    for mid, m in raw.items():
        try:
            out[mid] = ModelSpec(
                id=mid,
                backend=m["backend"],
                checkpoint=m["checkpoint"],
                decoding=m.get("decoding") or {},
                thinking=m.get("thinking") or {"supported": False},
                max_tokens=int(m["max_tokens"]),
                context_length=m.get("context_length"),
                base_url=m.get("base_url"),
                api_key_env=m.get("api_key_env"),
                checksum=m.get("checksum"),
                batch_size=int(m.get("batch_size", 8)),
                max_retries=int(m.get("max_retries", 3)),
                request_timeout=float(m.get("request_timeout", 300)),
                logprobs=bool(m.get("logprobs", False)),
                extra_body=m.get("extra_body") or {},
                mock=m.get("mock") or {},
            )
        except KeyError as e:
            raise RegistryError(f"models.yaml: {mid} missing field {e}") from e
    return out


def get_model(root: Path, model_id: str, backend_override: str | None = None) -> ModelSpec:
    """`model_id` may be 'name@backend' (e.g. 'qwen3.8-27b@mock'): the registry entry `name` run on another
    backend. The suffix stays in the id, so such outputs live in their own directory and can never be
    mistaken for (or cached as) real-model outputs."""
    import dataclasses

    base_id, _, suffix = model_id.partition("@")
    override = backend_override or suffix or None
    ms = load_models(root)
    if base_id not in ms:
        raise RegistryError(f"model {base_id!r} not in registry/models.yaml (known: {', '.join(sorted(ms))})")
    spec = ms[base_id]
    if override and override != spec.backend:
        spec = dataclasses.replace(spec, id=f"{base_id}@{override}", backend=override)
    return spec


@dataclass
class ThinkingPlan:
    """How to realise thinking on/off for one model. `available=False` -> the run is n/a."""

    want: bool
    available: bool
    method: str
    reason: str = ""
    chat_template_kwargs: dict | None = None
    user_suffix: str | None = None
    checkpoint: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def resolve_thinking(model: ModelSpec, want: bool) -> ThinkingPlan:
    t = model.thinking or {}
    mode_key = "thinking_on" if want else "thinking_off"
    if not model.decoding.get(mode_key):
        return ThinkingPlan(want, False, t.get("method", "none"), f"models.yaml has no decoding.{mode_key}")
    if not t.get("supported", False):
        if want:
            return ThinkingPlan(want, False, "none", "model does not support thinking")
        return ThinkingPlan(want, True, "none")
    method = t.get("method")
    if method == "chat_template_kwargs":
        return ThinkingPlan(want, True, method, chat_template_kwargs={t.get("kwarg", "enable_thinking"): want})
    if method == "system_prompt":
        suffix = t.get("on_suffix") if want else t.get("off_suffix")
        return ThinkingPlan(want, True, method, user_suffix=suffix)
    if method == "checkpoint":
        ck = t.get("on_checkpoint") if want else t.get("off_checkpoint")
        if not ck:
            return ThinkingPlan(want, False, method, f"no {'on' if want else 'off'}_checkpoint configured")
        return ThinkingPlan(want, True, method, checkpoint=ck)
    if method == "fixed":
        fixed = str(t.get("fixed_mode", "off")).lower()
        fixed_on = fixed in ("on", "true")
        if fixed_on != want:
            return ThinkingPlan(want, False, method, f"reasoning mode is fixed to {fixed!r}")
        return ThinkingPlan(want, True, method)
    raise RegistryError(f"models.yaml: {model.id}: unknown thinking.method {method!r}")


# --------------------------------------------------------------------------- experiment configs

@dataclass
class ExperimentConfig:
    id: str
    thinking: bool
    rubric: bool
    output_mode: str  # binary | ternary
    description: str = ""
    raw: dict = field(default_factory=dict)


def load_config(root: Path, config_id: str) -> ExperimentConfig:
    path = root / "configs" / f"{config_id}.yaml"
    if not path.exists():
        raise RegistryError(f"config {config_id!r} not found at {path}")
    raw = _load_yaml(path)
    if raw.get("output_mode") not in ("binary", "ternary"):
        raise RegistryError(f"{path}: output_mode must be binary|ternary")
    return ExperimentConfig(
        id=raw.get("id", config_id),
        thinking=bool(raw["thinking"]),
        rubric=bool(raw["rubric"]),
        output_mode=raw["output_mode"],
        description=raw.get("description", ""),
        raw=raw,
    )


def list_configs(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "configs").glob("*.yaml"))


# --------------------------------------------------------------------------- prompts

SYSTEM_MARK = "<<<SYSTEM>>>"
USER_MARK = "<<<USER>>>"


def skeleton_path(root: Path, task_type: str, rubric: bool, output_mode: str) -> Path:
    return root / "prompts" / "skeletons" / f"{task_type}_{'rubric' if rubric else 'minimal'}_{output_mode}.md"


def split_skeleton(text: str) -> tuple[str, str]:
    if SYSTEM_MARK not in text or USER_MARK not in text:
        raise RegistryError(f"prompt skeleton must contain {SYSTEM_MARK} and {USER_MARK}")
    _, rest = text.split(SYSTEM_MARK, 1)
    system, user = rest.split(USER_MARK, 1)
    return system.strip("\n"), user.strip("\n")


def format_options(options: dict | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


@dataclass
class PromptPlan:
    """Everything that determines the raw generation for (dataset, model, config). Hashed into config_hash."""

    dataset_id: str
    model_id: str
    config: ExperimentConfig
    task_type: str
    skeleton_file: str
    skeleton_text: str
    rubric_file: str | None
    rubric_text: str | None
    thinking: ThinkingPlan
    generation: dict  # decoding params (+ max_tokens, checkpoint, logprobs)
    config_hash: str
    na_reason: str | None = None

    @property
    def is_na(self) -> bool:
        return self.na_reason is not None

    def render(self, case) -> list[dict]:
        system, user = split_skeleton(self.skeleton_text)
        fills = {
            "{{question}}": case.question,
            "{{options}}": format_options(case.options),
            "{{candidate_answer}}": case.candidate_answer,
            "{{rubric}}": (self.rubric_text or "").strip(),
        }
        for k, v in fills.items():
            user = user.replace(k, v)
            system = system.replace(k, v)
        if self.thinking.user_suffix:
            user = f"{user}\n{self.thinking.user_suffix}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def describe(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "config_id": self.config.id,
            "config": self.config.raw,
            "task_type": self.task_type,
            "skeleton_file": self.skeleton_file,
            "skeleton_text": self.skeleton_text,
            "rubric_file": self.rubric_file,
            "rubric_text": self.rubric_text,
            "thinking": self.thinking.as_dict(),
            "generation": self.generation,
            "config_hash": self.config_hash,
            "na_reason": self.na_reason,
        }

    def component_hashes(self) -> dict:
        return {
            "config": hash_obj(self.config.raw, 16),
            "prompt": hash_obj({"skeleton": self.skeleton_text, "rubric": self.rubric_text}, 16),
            "decoding": hash_obj({"generation": self.generation, "thinking": self.thinking.as_dict()}, 16),
        }


def build_prompt_plan(root: Path, dataset: DatasetSpec, model: ModelSpec, config: ExperimentConfig) -> PromptPlan:
    sk_path = skeleton_path(root, dataset.task_type, config.rubric, config.output_mode)
    if not sk_path.exists():
        raise RegistryError(f"missing prompt skeleton {sk_path}")
    skeleton_text = read_text_normalized(sk_path)
    rubric_text = rubric_file = None
    if config.rubric:
        if not dataset.rubric:
            raise RegistryError(f"dataset {dataset.id} has no rubric fragment but config {config.id} needs one")
        rubric_file = dataset.rubric
        rubric_text = read_text_normalized(root / dataset.rubric)

    thinking = resolve_thinking(model, config.thinking)
    mode_key = "thinking_on" if config.thinking else "thinking_off"
    decoding = dict(model.decoding.get(mode_key) or {})
    generation = {
        "decoding": decoding,
        "max_tokens": model.max_tokens,
        "checkpoint": thinking.checkpoint or model.checkpoint,
        "logprobs": model.logprobs,
    }
    payload = {
        "config": config.raw,
        "skeleton": skeleton_text,
        "rubric": rubric_text,
        "generation": generation,
        "thinking": thinking.as_dict(),
    }
    return PromptPlan(
        dataset_id=dataset.id,
        model_id=model.id,
        config=config,
        task_type=dataset.task_type,
        skeleton_file=str(sk_path.relative_to(root)).replace("\\", "/"),
        skeleton_text=skeleton_text,
        rubric_file=rubric_file,
        rubric_text=rubric_text,
        thinking=thinking,
        generation=generation,
        config_hash=hash_obj(payload, 12),
        na_reason=None if thinking.available else thinking.reason,
    )


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise RegistryError(f"missing file {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
