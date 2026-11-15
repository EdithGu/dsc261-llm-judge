"""[B] Inference runner: caching by (case_id, config_hash), resume, execution-only retries, lock check, manifest.

Layout:  {out}/raw/{dataset_id}/{model_id}/{config_hash}[-{tag}]/
            {case_id}.json        one record per case (never overwritten once written)
            config.json           resolved config, prompt texts, generation params, hashes
            run_manifest.json     git commit, model checksum, backend/vLLM version, batch size, data hash, sessions
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llm_judge.backends import Backend, ExecutionError
from llm_judge.registry import ModelSpec, PromptPlan
from llm_judge.schema import Case
from llm_judge.utils import env_info, git_info, now_iso, read_json, safe_filename, sha256_file, write_json

MODES = ("smoke", "probe", "screen", "run")


class LockError(RuntimeError):
    pass


def run_dir_for(out_root: Path, plan: PromptPlan, tag: str = "main") -> Path:
    name = plan.config_hash if tag in ("", "main") else f"{plan.config_hash}-{tag}"
    return Path(out_root) / "raw" / plan.dataset_id / plan.model_id / name


def record_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / f"{safe_filename(case_id)}.json"


def load_records(run_dir: Path) -> dict[str, dict]:
    out = {}
    for p in Path(run_dir).glob("*.json"):
        if p.name in ("config.json", "run_manifest.json", "pilot_report.json"):
            continue
        rec = read_json(p)
        if isinstance(rec, dict) and "case_id" in rec and "config_hash" in rec:
            out[rec["case_id"]] = rec
    return out


@dataclass
class RunSummary:
    run_dir: Path
    config_id: str
    config_hash: str
    na_reason: str | None = None
    n_cases: int = 0
    n_cached: int = 0
    n_generated: int = 0
    n_exec_failure: int = 0
    n_retried: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        if self.na_reason:
            return f"  {self.config_id:<4} n/a ({self.na_reason})"
        return (f"  {self.config_id:<4} {self.config_hash}  cases={self.n_cases} cached={self.n_cached} "
                f"new={self.n_generated} exec_fail={self.n_exec_failure} retried={self.n_retried} "
                f"({self.seconds:.1f}s)  -> {self.run_dir}")


def build_params(plan: PromptPlan, case: Case, run_tag: str) -> dict:
    return {
        "model": plan.generation["checkpoint"],
        "max_tokens": plan.generation["max_tokens"],
        "decoding": dict(plan.generation["decoding"]),
        "chat_template_kwargs": plan.thinking.chat_template_kwargs,
        "logprobs": plan.generation.get("logprobs", False),
        "_mock_hint": {
            "case_id": case.case_id,
            "label": case.label,
            "thinking": plan.config.thinking,
            "rubric": plan.config.rubric,
            "output_mode": plan.config.output_mode,
            "config_key": plan.config.id,
            "run_tag": run_tag,
        },
    }


def _generate_one(backend: Backend, plan: PromptPlan, case: Case, max_retries: int, run_tag: str,
                  sleep: Callable[[float], None]) -> dict:
    messages = plan.render(case)
    params = build_params(plan, case, run_tag)
    attempts = []
    for attempt in range(1, max_retries + 2):
        params["_attempt"] = attempt
        t0 = time.perf_counter()
        try:
            res = backend.generate(messages, params)
        except ExecutionError as e:
            attempts.append({"attempt": attempt, "error": str(e)[:500], "retryable": e.retryable, "at": now_iso()})
            if not e.retryable or attempt > max_retries:
                break
            sleep(min(backend.retry_backoff * (2 ** (attempt - 1)), 60.0))
            continue
        latency = time.perf_counter() - t0
        return {
            "status": "ok",
            "text": res.text,
            "reasoning": res.reasoning,
            "usage": res.usage,
            "finish_reason": res.finish_reason,
            "latency": res.latency if res.latency is not None else latency,
            "logprobs": res.logprobs,
            "model_version": res.model_version,
            "attempts": attempts,  # failed attempts before success (empty if first try worked)
        }
    return {
        "status": "exec_failure",
        "text": None,
        "reasoning": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None},
        "finish_reason": None,
        "latency": None,
        "logprobs": None,
        "model_version": None,
        "attempts": attempts,
    }


def check_lock(lock_path: Path, plan: PromptPlan) -> None:
    """Formal `run` refuses to start if prompt / parser / decoding differ from the frozen versions."""
    from llm_judge.parse import parser_hash

    if not lock_path.exists():
        raise LockError(f"no freeze lock at {lock_path}. Run `freeze` at the end of the pilot first (section 6.5).")
    lock = read_json(lock_path)
    problems = []
    if lock.get("parser_hash") != parser_hash():
        problems.append(f"parser hash {parser_hash()} != frozen {lock.get('parser_hash')}")
    key = f"{plan.dataset_id}/{plan.model_id}/{plan.config.id}"
    entry = (lock.get("entries") or {}).get(key)
    if entry is None:
        problems.append(f"{key} was not frozen")
    elif entry.get("config_hash") != plan.config_hash:
        now = plan.component_hashes()
        diffs = [k for k, v in (entry.get("components") or {}).items() if now.get(k) != v]
        problems.append(f"{key}: config_hash {plan.config_hash} != frozen {entry.get('config_hash')} "
                        f"(changed: {', '.join(diffs) or 'unknown'})")
    if problems:
        raise LockError("lock check failed — formal run refused:\n  - " + "\n  - ".join(problems))


def run_inference(
    cases: list[Case],
    plan: PromptPlan,
    backend: Backend,
    model: ModelSpec,
    out_root: Path,
    mode: str,
    data_path: Path | None = None,
    run_tag: str = "main",
    batch_size: int | None = None,
    max_retries: int | None = None,
    retry_failed: bool = False,
    lock_path: Path | None = None,
    root: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    run_dir = run_dir_for(out_root, plan, run_tag)
    summary = RunSummary(run_dir=run_dir, config_id=plan.config.id, config_hash=plan.config_hash, n_cases=len(cases))

    if mode == "run":
        check_lock(lock_path or (Path(out_root) / "frozen" / "lock.json"), plan)

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = plan.describe() | {"run_tag": run_tag, "data_path": str(data_path) if data_path else None}
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        write_json(cfg_path, cfg)

    if plan.is_na:
        summary.na_reason = plan.na_reason
        return summary

    batch_size = int(batch_size or model.batch_size)
    max_retries = int(model.max_retries if max_retries is None else max_retries)

    pending: list[Case] = []
    for c in cases:
        p = record_path(run_dir, c.case_id)
        if p.exists():
            if retry_failed and read_json(p).get("status") == "exec_failure":
                pending.append(c)
            else:
                summary.n_cached += 1
        else:
            pending.append(c)

    man_path = run_dir / "run_manifest.json"
    manifest = read_json(man_path) if man_path.exists() else {
        "dataset_id": plan.dataset_id,
        "model_id": plan.model_id,
        "config_id": plan.config.id,
        "config_hash": plan.config_hash,
        "run_tag": run_tag,
        "model_checkpoint": plan.generation["checkpoint"],
        "model_checksum": model.checksum,
        "sessions": [],
    }
    session = {
        "mode": mode,
        "started_at": now_iso(),
        "git": git_info(root) if root else None,
        "env": env_info(),
        "server": backend.server_info(),
        "batch_size": batch_size,
        "max_retries": max_retries,
        "data_path": str(data_path) if data_path else None,
        "data_sha256": sha256_file(data_path) if data_path and Path(data_path).exists() else None,
        "n_requested": len(cases),
        "n_pending": len(pending),
    }
    manifest["sessions"].append(session)
    write_json(man_path, manifest)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for b in range(0, len(pending), batch_size):  # fixed-size batches (D1)
            batch = pending[b : b + batch_size]
            results = list(pool.map(lambda c: _generate_one(backend, plan, c, max_retries, run_tag, sleep), batch))
            for c, res in zip(batch, results):
                rec = {
                    "case_id": c.case_id,
                    "question_id": c.question_id,
                    "dataset_id": c.dataset_id,
                    "split": c.split,
                    "model_id": plan.model_id,
                    "config_id": plan.config.id,
                    "config_hash": plan.config_hash,
                    "run_tag": run_tag,
                    "mode": mode,
                    "timestamp": now_iso(),
                    **res,
                }
                write_json(record_path(run_dir, c.case_id), rec)
                summary.n_generated += 1
                summary.n_exec_failure += res["status"] == "exec_failure"
                summary.n_retried += bool(res["attempts"])
            if progress:
                progress(f"    {plan.config.id}: {min(b + batch_size, len(pending))}/{len(pending)}")
    summary.seconds = time.perf_counter() - t0

    session.update({"ended_at": now_iso(), "seconds": round(summary.seconds, 3),
                    "n_generated": summary.n_generated, "n_exec_failure": summary.n_exec_failure,
                    "n_retried": summary.n_retried})
    write_json(man_path, manifest)
    return summary
