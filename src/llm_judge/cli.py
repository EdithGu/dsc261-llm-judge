"""Command line interface.

Data group (no GPU):   validate-data, adapt, manifest, smoke, probe, pilot-report, screen
Experiment group:      freeze, run, evaluate, report

    llm-judge validate-data --data cases.jsonl
    llm-judge smoke         --data cases.jsonl --n 20 --backend mock
    llm-judge probe         --data cases.jsonl --n 100 --backend api --model cheap-api
    llm-judge pilot-report  --run outputs/raw/<dataset>/<model>/<config_hash>/
    llm-judge screen        --datasets a,b --models x,y --configs E1,E2 --split probe
    llm-judge freeze        --datasets gsm8k,medqa --models qwen3.8-27b --configs E1,E2,E3,E4,E5
    llm-judge run           --data cases_gsm8k.jsonl --model qwen3.8-27b --configs E1,E2,E3,E4,E5
    llm-judge evaluate      --data cases_gsm8k.jsonl --model qwen3.8-27b --configs E1,E2,E3,E4,E5
    llm-judge report        --model qwen3.8-27b --datasets gsm8k,medqa
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from llm_judge import gates as G
from llm_judge.backends import make_backend
from llm_judge.calibrate import DEFAULT_SALT
from llm_judge.evaluate import EvaluationError, build_rows, evaluate_dataset
from llm_judge.parse import parser_hash, PARSER_VERSION
from llm_judge.registry import (RegistryError, build_prompt_plan, get_dataset, get_model, load_config,
                                load_datasets)
from llm_judge.report import build_report
from llm_judge.runner import LockError, load_records, run_dir_for, run_inference
from llm_judge.schema import Case, DataError, load_cases, make_manifest, validate_file
from llm_judge.utils import (now_iso, parse_kv_list, project_root, read_json, split_csv, stable_int,
                             write_json, write_jsonl, iter_jsonl)

ALL_CONFIGS = "E1,E2,E3,E4,E5"


def _print(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- helpers

def select_cases(cases: list[Case], splits: list[str] | None, n: int | None, seed: int = 0) -> list[Case]:
    """Deterministic, label-balanced subset: filter splits, order by hash, interleave correct/incorrect."""
    pool = [c for c in cases if not splits or c.split in splits]
    if n is None or n >= len(pool):
        return pool
    key = lambda c: stable_int(f"{seed}|{c.case_id}")
    pos = sorted((c for c in pool if c.label), key=key)
    neg = sorted((c for c in pool if not c.label), key=key)
    out = []
    while len(out) < n and (pos or neg):
        for lst in (pos, neg):
            if lst and len(out) < n:
                out.append(lst.pop(0))
    return out


def pseudo_roles(cases: list[Case], calib_share: float = 0.3, seed: int = 0) -> dict[str, str]:
    """Smoke only: hash-assign calib/test within each label stratum, ignoring the data's split."""
    roles = {}
    for lab in (True, False):
        grp = sorted((c for c in cases if c.label == lab), key=lambda c: stable_int(f"role|{seed}|{c.case_id}"))
        n_cal = max(1, round(calib_share * len(grp))) if len(grp) > 1 else 0
        for i, c in enumerate(grp):
            roles[c.case_id] = "calib" if i < n_cal else "test"
    if not any(r == "calib" for r in roles.values()) or not any(r == "test" for r in roles.values()):
        raise DataError("smoke needs at least 2 cases of one label to form calib + test")
    return roles


def _infer(root, cases, dataset, model, config_ids, out_root, mode, args, data_path, tag="main", lock_path=None):
    backend = make_backend(model, mock_opts=parse_kv_list(getattr(args, "mock_opt", None)))
    summaries = []
    for cid in config_ids:
        plan = build_prompt_plan(root, dataset, model, load_config(root, cid))
        s = run_inference(cases, plan, backend, model, out_root, mode, data_path=data_path, run_tag=tag,
                          batch_size=getattr(args, "batch_size", None), retry_failed=getattr(args, "retry_failed", False),
                          lock_path=lock_path, root=root, progress=_print if getattr(args, "verbose", False) else None)
        _print(s.line())
        summaries.append((cid, plan, s))
    return summaries


def _run_rows(run_dir: Path, cases: list[Case]) -> tuple[list[dict], dict, list[dict]]:
    cfg = read_json(run_dir / "config.json")
    recs = load_records(run_dir)
    have = [c for c in cases if c.case_id in recs]
    rows = build_rows(have, recs, cfg["config"]["output_mode"])
    return rows, cfg, [recs[c.case_id] for c in have]


# --------------------------------------------------------------------------- commands

def cmd_validate(args, root) -> int:
    rep = validate_file(args.data, args.manifest, require_manifest=args.require_manifest)
    _print(rep.summary())
    if args.json:
        write_json(args.json, {"ok": rep.ok, "issues": [i.__dict__ for i in rep.issues],
                               "split_counts": rep.split_counts, "sha256": rep.sha256})
    return 0 if rep.ok else 1


def cmd_manifest(args, root) -> int:
    man = make_manifest(args.data, args.seed, parse_kv_list(args.extra))
    out = Path(args.out) if args.out else Path(args.data).parent / "manifest.json"
    write_json(out, man)
    _print(f"wrote {out}\n  sha256 {man['sha256']}\n  splits {man['split_counts']}")
    return 0


def cmd_adapt(args, root) -> int:
    ds = get_dataset(root, args.dataset)
    if not ds.adapter:
        raise RegistryError(f"dataset {args.dataset} has no adapter registered")
    path = root / ds.adapter
    spec = importlib.util.spec_from_file_location(f"adapter_{args.dataset}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = [d for _, d in iter_jsonl(args.raw)]
    rows = mod.convert(raw, seed=args.seed)
    out = Path(args.out)
    write_jsonl(out, rows)
    man = make_manifest(out, args.seed, {"adapter": ds.adapter, "raw_file": str(args.raw)})
    write_json(out.parent / "manifest.json", man)
    _print(f"adapter {ds.adapter}: {len(raw)} raw rows -> {len(rows)} cases -> {out} (+ manifest.json)")
    rep = validate_file(out)
    _print(rep.summary())
    return 0 if rep.ok else 1


def cmd_smoke(args, root) -> int:
    out_root = Path(args.out)
    rep = validate_file(args.data)
    if not rep.ok:
        _print(rep.summary())
        _print("smoke: fix validate-data errors first.")
        return 1
    cases = load_cases(args.data)
    subset = select_cases(cases, None, args.n, args.seed)
    roles = pseudo_roles(subset, seed=args.seed)
    dataset = get_dataset(root, subset[0].dataset_id)
    model = get_model(root, args.model, args.backend)
    configs = split_csv(args.configs)
    _print(f"smoke: {len(subset)} cases from {args.data} ({sum(v == 'calib' for v in roles.values())} pseudo-calib / "
           f"{sum(v == 'test' for v in roles.values())} pseudo-test), model {model.id}, backend {model.backend}")
    _print("[1/4] inference")
    _infer(root, subset, dataset, model, configs, out_root, "smoke", args, Path(args.data))
    _print("[2/4] parse + calibrate + policies + metrics + bootstrap")
    evaluate_dataset(root, subset, model.id, configs, out_root, role_override=roles, B=args.B, R=args.R,
                     force_thresholds=True, data_path=str(args.data))
    _print("[3/4] report")
    path = build_report(out_root / "eval", model.id, [dataset.id], out_root / "reports" / model.id, B=args.B)
    _print(f"[4/4] done -> {path}")
    _print("smoke validates that the data flows through the pipeline; mock numbers say nothing about difficulty.")
    return 0


def cmd_probe(args, root) -> int:
    cases = load_cases(args.data)
    splits = split_csv(args.split)
    subset = select_cases(cases, splits, args.n, args.seed)
    if not subset and "probe" in splits:
        _print("probe: no 'probe' split in the data — falling back to 'pilot'")
        subset = select_cases(cases, ["pilot"], args.n, args.seed)
    if not subset:
        _print("probe: no cases selected")
        return 1
    if len(subset) < G.GATES["min_probe_n_for_error_gate"]:
        _print(f"note: {len(subset)} cases < {G.GATES['min_probe_n_for_error_gate']} — error-rate CI will be wide")
    dataset = get_dataset(root, subset[0].dataset_id)
    model = get_model(root, args.model, args.backend)
    out_root = Path(args.out)
    _print(f"probe: {len(subset)} cases, model {model.id} ({model.backend}), configs {args.configs}, tag {args.tag}")
    summaries = _infer(root, subset, dataset, model, split_csv(args.configs), out_root, "probe", args,
                       Path(args.data), tag=args.tag)
    for cid, plan, s in summaries:
        if s.na_reason:
            continue
        rows, _, _ = _run_rows(s.run_dir, subset)
        md = G.difficulty_report(rows, f"{dataset.id} / {model.id} / {cid} ({len(rows)} cases)")
        (s.run_dir / "difficulty_report.md").write_text(md + "\n", encoding="utf-8")
        _print("\n" + md + f"\n\n(saved to {s.run_dir / 'difficulty_report.md'})")
    return 0


def _pick_runs(runs: list[tuple[Path, dict]]) -> tuple:
    prim = think = None
    for d, cfg in runs:
        c = cfg["config"]
        if c["output_mode"] == "binary" and not c["thinking"] and not c["rubric"] and prim is None:
            prim = (d, cfg)
        elif c["output_mode"] == "binary" and c["thinking"] and not c["rubric"] and think is None:
            think = (d, cfg)
    if prim is None:
        prim = runs[0]
    return prim, think


def cmd_pilot_report(args, root) -> int:
    run_dirs = [Path(r) for r in args.run]
    runs = []
    for d in run_dirs:
        if not (d / "config.json").exists():
            _print(f"pilot-report: {d} is not a run directory (no config.json)")
            return 1
        runs.append((d, read_json(d / "config.json")))
    data_path = args.data or runs[0][1].get("data_path")
    if not data_path:
        _print("pilot-report: pass --data (run config has no data_path)")
        return 1
    cases = load_cases(data_path)
    (pdir, pcfg), think = _pick_runs(runs)
    prim_rows, _, prim_recs = _run_rows(pdir, cases)
    prim_stats = G.run_stats(prim_rows)
    think_stats = think_recs = None
    if think:
        t_rows, _, think_recs = _run_rows(think[0], cases)
        think_stats = G.run_stats(t_rows)

    rep_dir = Path(args.replicate) if args.replicate else None
    if rep_dir is None:
        auto = pdir.parent / f"{pdir.name}-r2"
        rep_dir = auto if auto.exists() else None
    agreement = None
    if rep_dir is not None:
        r_rows, _, _ = _run_rows(rep_dir, cases)
        agreement = G.replicate_agreement(prim_rows, r_rows)

    man = read_json(pdir / "run_manifest.json") if (pdir / "run_manifest.json").exists() else {}
    batch = (man.get("sessions") or [{}])[-1].get("batch_size") or 1
    cost = G.cost_estimate(prim_recs + (think_recs or []), batch, args.formal_cases, args.formal_runs,
                           args.budget_gpu_hours, args.gpus)
    checks = G.gate_checks(prim_stats, think_stats, agreement, cost)

    title = f"{pcfg['dataset_id']} / {pcfg['model_id']}"
    md = [f"# Pilot report — {title}", "", f"_generated {now_iso()}_", "",
          f"- primary (E1-type) run: `{pdir}` ({pcfg['config_id']}, {prim_stats['n']} cases, splits {prim_stats['splits']})",
          f"- thinking (E2-type) run: `{think[0]}`" if think else "- thinking (E2-type) run: not provided",
          f"- replicate run: `{rep_dir}`" if rep_dir else "- replicate run: not provided", "",
          G.checks_markdown(checks), "",
          G.difficulty_report(prim_rows, f"{pcfg['config_id']}")]
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else pdir / "pilot_report.md"
    out.write_text(text, encoding="utf-8")
    write_json(out.with_suffix(".json"), {"checks": checks, "primary": prim_stats, "thinking": think_stats,
                                          "replicate": agreement, "cost": cost})
    _print(text + f"\n(saved to {out})")
    return 1 if args.strict and any(c["status"] == G.FAIL for c in checks) else 0


def cmd_screen(args, root) -> int:
    data_map = {k: v for k, v in (x.split("=", 1) for x in (args.data or []))}
    registry = load_datasets(root)
    out_root = Path(args.out)
    rows_md = ["| dataset | model | config | n | error rate [95% CI] | valid | truncated | max verdict share | conf mode share | gates |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    results = []
    for did in split_csv(args.datasets):
        raw = data_map.get(did) or (registry[did].data_path if did in registry else None)
        path = Path(raw) if raw else None
        if path is not None and not path.is_absolute() and not path.exists():
            path = root / path
        if path is None or not path.exists():
            _print(f"screen: no data for {did} (use --data {did}=path)")
            return 1
        cases = load_cases(path)
        subset = select_cases(cases, split_csv(args.split), args.n, args.seed)
        if not subset:
            _print(f"screen: {did} has no cases in split {args.split}")
            return 1
        dataset = get_dataset(root, did)
        for mid in split_csv(args.models):
            model = get_model(root, mid, args.backend)
            _print(f"screen: {did} x {model.id} ({len(subset)} cases)")
            summaries = _infer(root, subset, dataset, model, split_csv(args.configs), out_root, "screen", args, path)
            stats_by_cfg = {}
            for cid, plan, s in summaries:
                if s.na_reason:
                    rows_md.append(f"| {did} | {model.id} | {cid} | — | n/a ({s.na_reason}) | | | | | |")
                    continue
                rows, _, _ = _run_rows(s.run_dir, subset)
                st = G.run_stats(rows)
                stats_by_cfg[cid] = (plan, st)
            prim = next((st for cid, (p, st) in stats_by_cfg.items()
                         if not p.config.thinking and not p.config.rubric and p.config.output_mode == "binary"), None)
            think = next((st for cid, (p, st) in stats_by_cfg.items()
                          if p.config.thinking and not p.config.rubric and p.config.output_mode == "binary"), None)
            overall = "—"
            if prim:
                checks = G.gate_checks(prim, think)
                overall = G.FAIL if any(c["status"] == G.FAIL for c in checks) else (
                    G.BORDER if any(c["status"] == G.BORDER for c in checks) else G.PASS)
            for cid, (p, st) in stats_by_cfg.items():
                lo, hi = st["error_rate_ci"]
                f = lambda x: "—" if x is None else f"{100 * x:.1f}%"
                rows_md.append(f"| {did} | {model.id} | {cid} | {st['n']} | {f(st['error_rate'])} [{f(lo)}, {f(hi)}] | "
                               f"{f(st['valid_rate'])} | {f(st['truncation_rate'])} | {f(st['verdict_max_share'])} | "
                               f"{f(st['conf_mode_share'])} | {overall} |")
                results.append({"dataset": did, "model": model.id, "config": cid, "stats": st, "overall": overall})
    stamp = now_iso().replace(":", "").replace("-", "")[:15]
    out = out_root / "screen" / f"screen_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "# Screen — hard-gate comparison (section 6.3)\n\n" + "\n".join(rows_md) + \
           "\n\nGates column = overall section 6.1 status from the E1/E2-type runs of that combination " \
           "(reproducibility and cost need `pilot-report`).\n"
    out.write_text(text, encoding="utf-8")
    write_json(out.with_suffix(".json"), results)
    _print("\n" + text + f"\n(saved to {out})")
    return 0


def cmd_freeze(args, root) -> int:
    out_root = Path(args.out)
    lock_path = out_root / "frozen" / "lock.json"
    if lock_path.exists() and not args.force:
        _print(f"freeze: {lock_path} already exists. Re-freezing changes the locked setup — log it in "
               f"decision_log.md and pass --force.")
        return 1
    entries = {}
    prompts_dir = out_root / "frozen" / "prompts"
    for did in split_csv(args.datasets):
        ds = get_dataset(root, did)
        for mid in split_csv(args.models):
            model = get_model(root, mid)
            for cid in split_csv(args.configs):
                plan = build_prompt_plan(root, ds, model, load_config(root, cid))
                entries[f"{did}/{model.id}/{cid}"] = {
                    "config_hash": plan.config_hash,
                    "components": plan.component_hashes(),
                    "na_reason": plan.na_reason,
                    "generation": plan.generation,
                    "thinking": plan.thinking.as_dict(),
                    "skeleton_file": plan.skeleton_file,
                    "rubric_file": plan.rubric_file,
                }
                write_json(prompts_dir / did / model.id / f"{cid}.json", plan.describe())
    lock = {"frozen_at": now_iso(), "parser_version": PARSER_VERSION, "parser_hash": parser_hash(),
            "tie_salt": DEFAULT_SALT, "entries": entries}
    write_json(lock_path, lock)
    _print(f"froze {len(entries)} (dataset, model, config) entries + parser {parser_hash()} -> {lock_path}")
    for k, e in entries.items():
        _print(f"  {k:<40} {e['config_hash']}" + (f"  (n/a: {e['na_reason']})" if e["na_reason"] else ""))
    return 0


def cmd_run(args, root) -> int:
    cases = load_cases(args.data, verify_manifest=True, manifest=args.manifest)
    formal = [c for c in cases if c.split in ("calib", "test")]
    if not formal:
        _print("run: data has no calib/test cases")
        return 1
    dataset = get_dataset(root, formal[0].dataset_id)
    model = get_model(root, args.model, args.backend)
    out_root = Path(args.out)
    _print(f"run (formal): {dataset.id} x {model.id} — {len(formal)} calib+test cases, configs {args.configs}")
    try:
        _infer(root, formal, dataset, model, split_csv(args.configs), out_root, "run", args, Path(args.data),
               tag=args.tag, lock_path=out_root / "frozen" / "lock.json")
    except LockError as e:
        _print(str(e))
        return 2
    return 0


def cmd_evaluate(args, root) -> int:
    cases = load_cases(args.data, verify_manifest=not args.no_manifest_check, manifest=args.manifest)
    model = get_model(root, args.model, args.backend)
    ev = evaluate_dataset(root, cases, model.id, split_csv(args.configs), Path(args.out), run_tag=args.tag,
                          B=args.B, R=args.R, force_thresholds=args.force_thresholds,
                          allow_missing=args.allow_missing, data_path=str(args.data))
    m = ev["meta"]
    _print(f"evaluated {m['dataset_id']} x {m['model_id']}: calib {m['n_calib']}, test {m['n_test']} -> "
           f"{Path(args.out) / 'eval' / m['dataset_id'] / m['model_id'] / 'evaluation.json'}")
    for c, blk in ev["comparisons"].get("primary_endpoint", {}).items():
        if blk.get("reachable"):
            f = lambda x, spec: "—" if x is None else format(x, spec)
            lo, hi = blk["ci95"]
            _print(f"  {c}: risk diff @75% = {f(blk['estimate'], '+.4f')}  95% CI [{f(lo, '+.4f')}, {f(hi, '+.4f')}]"
                   f"  p_boot={f(blk['p_boot'], '.3g')}"
                   + ("  (no test case retained)" if blk["estimate"] is None else ""))
        else:
            _print(f"  {c}: 75% coverage unreachable")
    return 0


def cmd_report(args, root) -> int:
    model = get_model(root, args.model, args.backend)
    out_dir = Path(args.reports_dir) / model.id
    path = build_report(Path(args.out) / "eval", model.id, split_csv(args.datasets), out_dir, B=args.B)
    _print(f"report -> {path}")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llm-judge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_infer(p, default_out="outputs"):
        p.add_argument("--out", default=default_out, help="output root (raw/, frozen/, eval/ live under it)")
        p.add_argument("--backend", choices=["vllm", "api", "mock"], default=None,
                       help="override the model's backend (a different backend gets its own '<model>@<backend>' id)")
        p.add_argument("--batch-size", type=int, default=None, help="override models.yaml batch_size")
        p.add_argument("--mock-opt", action="append", help="mock backend knob, e.g. --mock-opt accuracy=0.9")
        p.add_argument("--retry-failed", action="store_true", help="re-attempt cached exec_failure records")
        p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("validate-data", help="static checks of a cases file")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--require-manifest", action="store_true")
    p.add_argument("--json", default=None, help="also write the report as JSON")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("manifest", help="write manifest.json (sha256, seed, split counts) for a cases file")
    p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, required=True, help="sampling seed used to build the file")
    p.add_argument("--out", default=None)
    p.add_argument("--extra", action="append", help="extra key=value fields")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("adapt", help="run a dataset adapter: raw jsonl -> cases jsonl + manifest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_adapt)

    p = sub.add_parser("smoke", help="mock end-to-end run on a few cases (no GPU)")
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--model", default="mock")
    p.add_argument("--configs", default=ALL_CONFIGS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--R", type=int, default=1000)
    common_infer(p, default_out="outputs/smoke")
    p.set_defaults(fn=cmd_smoke, backend_default="mock")

    p = sub.add_parser("probe", help="difficulty report on a probe/pilot sample")
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--model", required=True)
    p.add_argument("--configs", default="E1")
    p.add_argument("--split", default="probe", help="comma list of splits to sample from")
    p.add_argument("--tag", default="main", help="run tag; use r2 for the reproducibility replicate")
    p.add_argument("--seed", type=int, default=0)
    common_infer(p)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("pilot-report", help="section 6.1 gate checklist from pilot/probe run dirs")
    p.add_argument("--run", action="append", required=True, help="run dir; repeat for E1 and E2")
    p.add_argument("--replicate", default=None, help="replicate run dir (default: <run>-r2 if it exists)")
    p.add_argument("--data", default=None)
    p.add_argument("--budget-gpu-hours", type=float, default=None)
    p.add_argument("--formal-cases", type=int, default=200)
    p.add_argument("--formal-runs", type=int, default=10)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--strict", action="store_true", help="exit 1 if any gate FAILs")
    p.set_defaults(fn=cmd_pilot_report)

    p = sub.add_parser("screen", help="compare dataset x model combinations on hard gates")
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--configs", default="E1,E2")
    p.add_argument("--split", default="probe")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--data", action="append", help="dataset_id=path (default: data_path in datasets.yaml)")
    p.add_argument("--seed", type=int, default=0)
    common_infer(p)
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("freeze", help="end of pilot: lock prompts/configs/parser/decoding hashes")
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--configs", default=ALL_CONFIGS)
    p.add_argument("--out", default="outputs")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("run", help="formal inference on calib+test (lock-checked)")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--configs", default=ALL_CONFIGS)
    p.add_argument("--tag", default="main")
    common_infer(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("evaluate", help="parse + calibrate + policies + metrics + paired bootstrap")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--no-manifest-check", action="store_true")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["vllm", "api", "mock"], default=None)
    p.add_argument("--configs", default=ALL_CONFIGS)
    p.add_argument("--tag", default="main")
    p.add_argument("--out", default="outputs")
    p.add_argument("--B", type=int, default=10_000)
    p.add_argument("--R", type=int, default=10_000)
    p.add_argument("--force-thresholds", action="store_true", help="overwrite frozen thresholds (log it!)")
    p.add_argument("--allow-missing", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("report", help="tables, figures and report.md across datasets (Holm here)")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["vllm", "api", "mock"], default=None)
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--B", type=int, default=10_000)
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles (cp936/cp1252) must never crash on a symbol
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = build_parser()
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else project_root()
    if getattr(args, "backend", None) is None and getattr(args, "backend_default", None):
        args.backend = args.backend_default
    try:
        return int(args.fn(args, root) or 0)
    except (DataError, RegistryError, EvaluationError, LockError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
