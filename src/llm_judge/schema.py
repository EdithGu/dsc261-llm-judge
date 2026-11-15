"""Generic case schema (section 3.1) and `validate-data` checks (section 3.2)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_judge.utils import iter_jsonl, read_json, sha256_file

SPLITS = ("pilot", "probe", "calib", "test")
TASK_TYPES = ("numeric", "multiple_choice", "short_answer")
JUDGE_VISIBLE = ("question", "options", "candidate_answer")
CORRECT_TYPE = "correct"

# field -> (types, required, nullable)
FIELDS: dict[str, tuple[tuple[type, ...], bool, bool]] = {
    "case_id": ((str,), True, False),
    "question_id": ((str,), True, False),
    "dataset_id": ((str,), True, False),
    "split": ((str,), True, False),
    "task_type": ((str,), True, False),
    "question": ((str,), True, False),
    "options": ((dict,), True, True),
    "candidate_answer": ((str,), True, False),
    "reference_answer": ((str,), True, False),
    "label": ((bool,), True, False),
    "candidate_type": ((str,), True, False),
    "meta": ((dict,), False, False),
}


@dataclass
class Case:
    case_id: str
    question_id: str
    dataset_id: str
    split: str
    task_type: str
    question: str
    options: dict | None
    candidate_answer: str
    reference_answer: str
    label: bool
    candidate_type: str
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(**{k: d.get(k, {} if k == "meta" else None) for k in FIELDS})

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- answer parsing helpers

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def parse_number(s: str) -> float | None:
    """Parse a bare numeric string like '1,000', '$12.50', '3/4', '-7', '45%'."""
    if s is None:
        return None
    t = s.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    t = t.rstrip(".")
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return float(a) / float(b)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def extract_final_number(text: str) -> float | None:
    """Final answer of a (possibly worked) numeric solution: after '####' / 'answer is', else the last number."""
    if text is None:
        return None
    direct = parse_number(text)
    if direct is not None:
        return direct
    low = text.lower()
    for marker in ("####", "answer is", "answer:"):
        pos = low.rfind(marker)
        if pos >= 0:
            nums = _NUM_RE.findall(text[pos + len(marker):])
            if nums:
                return parse_number(nums[0])
    nums = _NUM_RE.findall(text)
    return parse_number(nums[-1]) if nums else None


def mc_choice(candidate: str, options: dict) -> str | None:
    """Map a candidate answer to an option key: 'B', 'B.', '(B)', 'B. text', or the exact option text."""
    if not options:
        return None
    c = candidate.strip()
    keys = {str(k).strip().upper(): k for k in options}
    m = re.match(r"^\(?([A-Za-z])\)?[.):]?(\s|$)", c)
    if m and m.group(1).upper() in keys:
        return keys[m.group(1).upper()]
    for k, v in options.items():
        if c.lower() == str(v).strip().lower():
            return k
    return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# --------------------------------------------------------------------------- validation

@dataclass
class Issue:
    level: str  # error | warning
    check: str
    message: str
    line: int | None = None
    case_id: str | None = None

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line else ""
        if self.case_id:
            loc = f"{loc} ({self.case_id})" if loc else self.case_id
        return f"[{self.level.upper():7}] {self.check:<18} {loc + ': ' if loc else ''}{self.message}"


@dataclass
class ValidationReport:
    path: str
    n_rows: int = 0
    issues: list[Issue] = field(default_factory=list)
    split_counts: dict = field(default_factory=dict)
    label_balance: dict = field(default_factory=dict)
    sha256: str | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, check: str, message: str, line: int | None = None, case_id: str | None = None):
        self.issues.append(Issue(level, check, message, line, case_id))

    def summary(self, max_issues: int = 200) -> str:
        lines = [f"validate-data: {self.path}", f"  rows: {self.n_rows}   sha256: {self.sha256}"]
        if self.split_counts:
            lines.append("  splits: " + ", ".join(f"{k}={v}" for k, v in self.split_counts.items()))
        for s, b in self.label_balance.items():
            lines.append(f"  {s:<6} correct={b['correct']} incorrect={b['incorrect']} (correct share {b['share']:.2f})")
        for i in self.issues[:max_issues]:
            lines.append("  " + str(i))
        if len(self.issues) > max_issues:
            lines.append(f"  ... {len(self.issues) - max_issues} more issues")
        lines.append(f"  RESULT: {'PASS' if self.ok else 'FAIL'} ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        return "\n".join(lines)


_LEAK_RE = re.compile(
    r"(?:\b(?:ground[ _-]?truth|reference[ _-]answer|gold[ _-]?answer|is_correct|candidate[ _-]type)\b"
    r"|\blabel\s*[:=]"
    r"|[\[(](?:in)?correct[\])])",
    re.IGNORECASE,
)


def validate_records(
    rows: list[tuple[int, Any]],
    path: str = "<memory>",
    expected_splits: dict | None = None,
    balance_tol: float = 0.15,
    min_n_for_balance: int = 10,
) -> ValidationReport:
    rep = ValidationReport(path=path, n_rows=len(rows))
    cases: list[tuple[int, Case]] = []

    # ---- schema / types
    for line, d in rows:
        if not isinstance(d, dict):
            rep.add("error", "schema", "row is not a JSON object", line)
            continue
        cid = d.get("case_id") if isinstance(d.get("case_id"), str) else None
        bad = False
        for name, (types, required, nullable) in FIELDS.items():
            if name not in d:
                if required:
                    rep.add("error", "schema", f"missing field '{name}'", line, cid)
                    bad = True
                continue
            v = d[name]
            if v is None:
                if not nullable:
                    rep.add("error", "schema", f"field '{name}' is null", line, cid)
                    bad = True
                continue
            if isinstance(v, bool) and bool not in types:
                ok_type = False
            else:
                ok_type = isinstance(v, types)
            if not ok_type:
                rep.add("error", "schema", f"field '{name}' has type {type(v).__name__}, expected {types[0].__name__}", line, cid)
                bad = True
        extra = set(d) - set(FIELDS)
        if extra:
            rep.add("warning", "schema", f"unknown fields {sorted(extra)} (put free-form data in 'meta')", line, cid)
        if bad:
            continue
        if d["split"] not in SPLITS:
            rep.add("error", "schema", f"split {d['split']!r} not in {SPLITS}", line, cid)
            continue
        if d["task_type"] not in TASK_TYPES:
            rep.add("error", "schema", f"task_type {d['task_type']!r} not in {TASK_TYPES}", line, cid)
            continue
        if not d["question"].strip() or not d["candidate_answer"].strip():
            rep.add("error", "schema", "empty question or candidate_answer", line, cid)
        cases.append((line, Case.from_dict(d)))

    if not cases:
        if not rows:
            rep.add("error", "schema", "file contains no cases")
        return rep

    # ---- single dataset / task type per file
    ds_ids = Counter(c.dataset_id for _, c in cases)
    if len(ds_ids) > 1:
        rep.add("error", "schema", f"one file must hold one dataset_id, found {dict(ds_ids)}")
    tts = Counter(c.task_type for _, c in cases)
    if len(tts) > 1:
        rep.add("error", "schema", f"mixed task_type in one file: {dict(tts)}")

    # ---- label <-> candidate_type consistency
    for line, c in cases:
        if c.label != (c.candidate_type == CORRECT_TYPE):
            rep.add("error", "label", f"label={c.label} but candidate_type={c.candidate_type!r} "
                    f"(label must be true iff candidate_type == 'correct')", line, c.case_id)

    # ---- split counts and label balance
    by_split: dict[str, list[Case]] = defaultdict(list)
    for _, c in cases:
        by_split[c.split].append(c)
    rep.split_counts = {s: len(by_split.get(s, [])) for s in SPLITS}
    for s in SPLITS:
        cs = by_split.get(s, [])
        if not cs:
            continue
        n_c = sum(c.label for c in cs)
        share = n_c / len(cs)
        rep.label_balance[s] = {"correct": n_c, "incorrect": len(cs) - n_c, "share": share}
        if len(cs) >= min_n_for_balance and abs(share - 0.5) > balance_tol:
            rep.add("warning", "balance", f"split '{s}': correct share {share:.2f} is not roughly 50/50")
    if expected_splits:
        for s, n in expected_splits.items():
            if rep.split_counts.get(s, 0) != n:
                rep.add("error", "split_counts", f"split '{s}' has {rep.split_counts.get(s, 0)} cases, manifest says {n}")

    # ---- duplicates
    seen_ids: dict[str, int] = {}
    for line, c in cases:
        if c.case_id in seen_ids:
            rep.add("error", "duplicate", f"case_id repeated (first at line {seen_ids[c.case_id]})", line, c.case_id)
        else:
            seen_ids[c.case_id] = line
    pair_seen: dict[tuple, int] = {}
    for line, c in cases:
        key = (c.question_id, normalize_text(c.candidate_answer))
        if key in pair_seen:
            rep.add("error", "duplicate", f"same question_id + candidate_answer as line {pair_seen[key]}", line, c.case_id)
        else:
            pair_seen[key] = line
    text_to_qids: dict[str, set] = defaultdict(set)
    for _, c in cases:
        text_to_qids[normalize_text(c.question)].add(c.question_id)
    for txt, qids in text_to_qids.items():
        if len(qids) > 1:
            rep.add("error", "duplicate", f"identical question text under different question_ids {sorted(qids)}: "
                    f"{txt[:60]!r}...")

    # ---- question crossing splits
    q_splits: dict[str, set] = defaultdict(set)
    for _, c in cases:
        q_splits[c.question_id].add(c.split)
    for qid, ss in q_splits.items():
        if len(ss) > 1:
            rep.add("error", "split_leak", f"question_id {qid!r} appears in several splits {sorted(ss)}")

    # ---- probe must not overlap formal calib/test (by id AND by question text)
    formal_text = {normalize_text(c.question) for c in by_split.get("calib", []) + by_split.get("test", [])}
    for line, c in cases:
        if c.split == "probe" and normalize_text(c.question) in formal_text:
            rep.add("error", "probe_overlap", "probe question text also appears in calib/test", line, c.case_id)

    # ---- label leakage into judge-visible fields
    for line, c in cases:
        visible = [c.question, c.candidate_answer] + [str(v) for v in (c.options or {}).values()]
        blob = "\n".join(visible)
        m = _LEAK_RE.search(blob)
        if m:
            rep.add("error", "leakage", f"judge-visible text contains {m.group(0)!r}", line, c.case_id)
        if c.candidate_type != CORRECT_TYPE and "_" in c.candidate_type and \
                re.search(rf"\b{re.escape(c.candidate_type)}\b", blob, re.IGNORECASE):
            # only snake_case type tags (e.g. omitted_step) — plain words like "arithmetic" occur naturally
            rep.add("error", "leakage", f"judge-visible text mentions candidate_type tag {c.candidate_type!r}", line, c.case_id)

    # ---- task-type specific answer checks
    for line, c in cases:
        if c.task_type == "numeric":
            if parse_number(c.reference_answer) is None:
                rep.add("error", "numeric", f"reference_answer {c.reference_answer!r} is not a number", line, c.case_id)
            cand = extract_final_number(c.candidate_answer)
            if cand is None:
                rep.add("error", "numeric", "cannot extract a final number from candidate_answer", line, c.case_id)
            ref = parse_number(c.reference_answer)
            if cand is not None and ref is not None:
                same = abs(cand - ref) < 1e-6 * max(1.0, abs(ref))
                if same != c.label:
                    rep.add("warning", "numeric", f"label={c.label} but candidate final number {cand:g} "
                            f"{'==' if same else '!='} reference {ref:g}", line, c.case_id)
            if c.options:
                rep.add("warning", "schema", "numeric case has options", line, c.case_id)
        elif c.task_type == "multiple_choice":
            if not c.options:
                rep.add("error", "multiple_choice", "multiple_choice case without options", line, c.case_id)
                continue
            ck = mc_choice(c.candidate_answer, c.options)
            if ck is None:
                rep.add("error", "multiple_choice", f"candidate_answer {c.candidate_answer!r} is not one of the options", line, c.case_id)
            rk = mc_choice(c.reference_answer, c.options)
            if rk is None:
                rep.add("error", "multiple_choice", f"reference_answer {c.reference_answer!r} is not one of the options", line, c.case_id)
            if ck is not None and rk is not None and (ck == rk) != c.label:
                rep.add("error", "label", f"label={c.label} but candidate option {ck} vs reference option {rk}", line, c.case_id)
    return rep


# --------------------------------------------------------------------------- loading + manifest

class DataError(ValueError):
    pass


def find_manifest(data_path: Path, manifest: str | Path | None = None) -> Path | None:
    if manifest:
        return Path(manifest)
    data_path = Path(data_path)
    for cand in (data_path.parent / "manifest.json", data_path.with_suffix(".manifest.json")):
        if cand.exists():
            return cand
    return None


def check_manifest(data_path: Path, manifest_path: Path | None, rep: ValidationReport | None = None,
                   required: bool = False) -> dict | None:
    """Verify SHA256 + counts. With `rep`: record issues (errors if `required`). Without `rep`: raise DataError
    if `required` and anything fails."""
    data_path = Path(data_path)
    problems: list[tuple[bool, str]] = []  # (always_error, message)
    man = None
    if manifest_path is None or not Path(manifest_path).exists():
        problems.append((False, f"no manifest.json found next to {data_path.name}"))
    else:
        man = read_json(manifest_path)
        actual = sha256_file(data_path)
        if man.get("sha256") != actual:
            problems.append((True, f"SHA256 mismatch: manifest {man.get('sha256')} vs file {actual}"))
        if man.get("file") and man["file"] != data_path.name:
            problems.append((True, f"manifest is for {man['file']!r}, not {data_path.name!r}"))
        if "sampling_seed" not in man:
            problems.append((False, "manifest has no sampling_seed"))
    if rep is not None:
        for always, p in problems:
            rep.add("error" if (always or required) else "warning", "manifest", p)
    if required and problems and rep is None:  # with a report, problems are recorded as errors instead
        raise DataError("; ".join(p for _, p in problems))
    return man


def validate_file(path: str | Path, manifest: str | Path | None = None, require_manifest: bool = False) -> ValidationReport:
    path = Path(path)
    try:
        rows = list(iter_jsonl(path))
    except ValueError as e:
        rep = ValidationReport(path=str(path))
        rep.add("error", "json", str(e))
        return rep
    man_path = find_manifest(path, manifest)
    man = None
    if man_path is not None and man_path.exists():
        man = read_json(man_path)
    rep = validate_records(rows, str(path), expected_splits=(man or {}).get("split_counts"))
    rep.sha256 = sha256_file(path)
    check_manifest(path, man_path, rep, required=require_manifest)
    return rep


def load_cases(path: str | Path, verify_manifest: bool = False, manifest: str | Path | None = None) -> list[Case]:
    """Load + validate. Raises DataError on validation errors (or manifest problems when verify_manifest)."""
    rep = validate_file(path, manifest, require_manifest=False)
    if verify_manifest:
        check_manifest(Path(path), find_manifest(Path(path), manifest), required=True)
    if not rep.ok:
        raise DataError(f"{path} failed validation:\n{rep.summary(max_issues=20)}")
    return [Case.from_dict(d) for _, d in iter_jsonl(path)]


def make_manifest(path: str | Path, sampling_seed: int | None, extra: dict | None = None) -> dict:
    from llm_judge.utils import now_iso

    path = Path(path)
    rows = [d for _, d in iter_jsonl(path)]
    counts = Counter(r.get("split") for r in rows)
    ds = sorted({r.get("dataset_id") for r in rows})
    return {
        "file": path.name,
        "dataset_id": ds[0] if len(ds) == 1 else ds,
        "sha256": sha256_file(path),
        "sampling_seed": sampling_seed,
        "n_cases": len(rows),
        "split_counts": {s: counts.get(s, 0) for s in SPLITS if counts.get(s, 0)},
        "created_at": now_iso(),
        **(extra or {}),
    }
