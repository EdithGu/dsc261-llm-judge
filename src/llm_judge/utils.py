"""Small shared helpers: hashing, JSON I/O, paths, git info."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------- paths

def project_root() -> Path:
    """Repo root: $LLM_JUDGE_ROOT, else the checkout containing this package, else cwd."""
    env = os.environ.get("LLM_JUDGE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in (here.parents[2], Path.cwd()):
        if (cand / "registry").is_dir():
            return cand
    return Path.cwd()


# --------------------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_text_normalized(path: str | Path) -> str:
    """Read text with CRLF -> LF so hashes of prompts/source are OS-independent."""
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_obj(obj: Any, n: int | None = None) -> str:
    h = sha256_text(canonical_json(obj))
    return h[:n] if n else h


def stable_int(text: str) -> int:
    """Deterministic 64-bit integer from a string (independent of PYTHONHASHSEED)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def stable_uniform(text: str) -> float:
    return stable_int(text) / float(1 << 64)


# --------------------------------------------------------------------------- JSON I/O

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Atomic write (tmp + rename) so an interrupted run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if o != o else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield (line_number, parsed) for each non-empty line. Raises ValueError with line number."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e.msg})") from e


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- misc

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(name: str) -> str:
    """Filesystem-safe version of a case_id; appends a short hash if anything was replaced."""
    cleaned = _SAFE.sub("_", name)
    if cleaned != name or not cleaned:
        cleaned = f"{cleaned}__{sha256_text(name)[:8]}"
    return cleaned


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info(root: str | Path) -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def parse_kv_list(items: list[str] | None) -> dict:
    """['a=1', 'b=x'] -> {'a': 1, 'b': 'x'} (values parsed as YAML scalars)."""
    import yaml

    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def split_csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]
