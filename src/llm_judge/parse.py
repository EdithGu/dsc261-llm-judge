"""[C] Parsing and validity classification.

Status values (section 5 [C]):
    valid           JSON parses, verdict legal for the output mode, confidence in [0, 1]
    invalid_format  no parsable JSON object with a verdict, or verdict not legal
    invalid_conf    confidence missing / non-numeric / out of range
    truncated       finish_reason == "length"
    exec_failure    the request still failed after retries (no text)
    abstain         verdict == "abstain" (ternary mode / E5 only)

Bump PARSER_VERSION whenever the rules change; the parser hash is part of the freeze lock.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_judge.utils import read_text_normalized, sha256_text

PARSER_VERSION = "1.0.0"

STATUSES = ("valid", "invalid_format", "invalid_conf", "truncated", "exec_failure", "abstain")
INVALID_STATUSES = ("invalid_format", "invalid_conf", "truncated", "exec_failure")
VERDICTS = {"binary": ("correct", "incorrect"), "ternary": ("correct", "incorrect", "abstain")}

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Judgment:
    status: str
    verdict: str | None = None       # "correct" | "incorrect" | "abstain" | None
    confidence: float | None = None
    error: str | None = None

    @property
    def verdict_bool(self) -> bool | None:
        return {"correct": True, "incorrect": False}.get(self.verdict or "")

    def to_dict(self) -> dict:
        return asdict(self)


def parser_hash() -> str:
    src = read_text_normalized(Path(__file__))
    return sha256_text(f"{PARSER_VERSION}\n{src}")[:16]


def _json_objects(text: str) -> list[dict]:
    """All top-level {...} substrings that parse as JSON objects, in order of appearance."""
    out = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                frag = text[start : i + 1]
                try:
                    obj = json.loads(frag)
                    if isinstance(obj, dict):
                        out.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def _find_payload(text: str) -> dict | None:
    """The judgment object: prefer fenced blocks, then any object; take the LAST one containing 'verdict'."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # opening tag was in the prompt template
        text = text.split("</think>", 1)[1]
    candidates: list[dict] = []
    for block in _FENCE_RE.findall(text):
        candidates.extend(_json_objects(block))
    if not candidates:
        candidates = _json_objects(text)
    with_verdict = [c for c in candidates if any(k.lower() == "verdict" for k in c)]
    return with_verdict[-1] if with_verdict else None


def _get_ci(d: dict, key: str):
    for k, v in d.items():
        if k.lower() == key:
            return v
    return None


def parse_output(text: str | None, finish_reason: str | None, output_mode: str, exec_failed: bool = False) -> Judgment:
    if output_mode not in VERDICTS:
        raise ValueError(f"output_mode must be binary|ternary, got {output_mode!r}")
    if exec_failed or text is None:
        return Judgment("exec_failure", error="no output (execution failure)")
    if finish_reason == "length":
        return Judgment("truncated", error="finish_reason == length")

    payload = _find_payload(text)
    if payload is None:
        return Judgment("invalid_format", error="no JSON object with a 'verdict' key")
    raw_verdict = _get_ci(payload, "verdict")
    if not isinstance(raw_verdict, str):
        return Judgment("invalid_format", error=f"verdict is not a string: {raw_verdict!r}")
    verdict = raw_verdict.strip().lower()
    if verdict not in VERDICTS[output_mode]:
        return Judgment("invalid_format", error=f"illegal verdict {raw_verdict!r} for {output_mode} mode")

    conf_raw = _get_ci(payload, "confidence")
    conf: float | None = None
    conf_err = None
    if conf_raw is None:
        conf_err = "confidence missing"
    elif isinstance(conf_raw, bool):
        conf_err = "confidence is a boolean"
    elif isinstance(conf_raw, (int, float)):
        conf = float(conf_raw)
    elif isinstance(conf_raw, str):
        try:
            conf = float(conf_raw.strip())
        except ValueError:
            conf_err = f"confidence not numeric: {conf_raw!r}"
    else:
        conf_err = f"confidence has type {type(conf_raw).__name__}"
    if conf is not None and (math.isnan(conf) or not 0.0 <= conf <= 1.0):
        conf_err = f"confidence {conf_raw!r} outside [0, 1]"
        conf = None

    if verdict == "abstain":  # intentional abstention; confidence optional
        return Judgment("abstain", verdict="abstain", confidence=conf)
    if conf_err:
        return Judgment("invalid_conf", verdict=verdict, error=conf_err)
    return Judgment("valid", verdict=verdict, confidence=conf)


def parse_record(record: dict, output_mode: str) -> Judgment:
    """Parse a raw output record written by the runner."""
    return parse_output(
        record.get("text"),
        record.get("finish_reason"),
        output_mode,
        exec_failed=record.get("status") == "exec_failure",
    )
