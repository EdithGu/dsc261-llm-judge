# Experiment Pipeline Design: When Should an LLM Judge Abstain?

> DSC 261 experiment pipeline plan (v3, final)
> Based on: `DSC261_Project_Proposal.pdf`
> Scope: the experiment pipeline only. Data construction belongs to the data leads (M1, M2); this document only defines the pipeline's interface to their data.
> Chinese version: [`experiment_pipeline_plan.md`](experiment_pipeline_plan.md)

---

## Contents

1. [Research Questions and Overview](#1-research-questions-and-overview)
2. [Key Decisions](#2-key-decisions)
3. [Data Interface (for the Data Team)](#3-data-interface-for-the-data-team)
4. [Swappable Architecture: Datasets and Models](#4-swappable-architecture-datasets-and-models)
5. [Pipeline Modules](#5-pipeline-modules)
6. [Pilot: Screening and Go/No-Go Gate](#6-pilot-screening-and-gono-go-gate)
7. [Statistical Analysis and Pre-registration](#7-statistical-analysis-and-pre-registration)
8. [Code Structure](#8-code-structure)
9. [Deliverables and Handoff](#9-deliverables-and-handoff)

---

## 1. Research Questions and Overview

**Primary question:** At the same coverage, does confidence-based abstention reduce judgment errors more effectively than random abstention?

**Secondary questions:**
- Do thinking and explicit rubrics improve accuracy and selective performance?
- Do the findings hold across the math and medical domains?

### 1.1 Experiment Matrix

**Primary judge: Qwen3.8-27B**

| Run | Thinking | Rubric | Output space | Domain |
|---|---|---|---|---|
| E1 Minimal baseline | Off | No | Correct/Incorrect + confidence | GSM8K, MedQA |
| E2 Thinking only | On | No | Same | Both |
| E3 Rubric only | Off | Yes | Same | Both |
| E4 Thinking + rubric | On | Yes | Same | Both |
| E5 Direct abstention | Off | No | Correct/Incorrect/Abstain + confidence | Both |

- That is 5 runs × 2 domains = 10 inference batches. Each batch covers all calibration and test cases (200 cases per domain).
- E1–E4 form a 2×2 factorial design, so the thinking × rubric interaction can be analyzed.
- **Optional replication with DeepSeek-V4-Flash-0731:** run E1, E3, and E5 only, under one fixed inference mode. This reuses the same runner with a different backend adapter.

### 1.2 Pipeline Flow

```
Frozen cases ──► [A] Config/Prompt registry ──► [B] Inference runner ──► raw outputs (cached)
                                                                │
                                                                ▼
                                                     [C] Parsing & validity classification
                                                                │
                         ┌──────────────────────────────────────┼─────────────────────┐
                         ▼                                      ▼                     ▼
              [D] Threshold calibration (calib 60)   [F] Metrics (test 140)   [H] Error analysis
                         │ thresholds.json (frozen)              ▲
                         ▼                                      │
              [E] Policy application (test) ─────────────────────┘
                 No abstention / Confidence / Random / Direct abstention
                                                                ▼
                                                     [G] Inference (paired bootstrap)
                                                                ▼
                                                     Figures + report + reproducibility package
```

---

## 2. Key Decisions

| # | Decision | Implementation |
|---|---|---|
| D1 | **Decoding:** temperature 0 when thinking is off; the officially recommended sampling parameters with a fixed seed when thinking is on; one sample per case | Both decoding profiles are stored in `models.yaml`. The pilot includes a **reproducibility check**: run the same cases twice, report verdict agreement (target ≥ 98%) and confidence differences. vLLM can produce different outputs at temperature 0 when batch sizes change, so formal runs fix the batch size and the vLLM version, and record both in the run manifest. |
| D2 | **Invalid outputs:** in the primary analysis, they are "not retained, and not abstentions" | Coverage = retained valid judgments ÷ **all cases**, so the maximum achievable coverage is 1 minus the invalid rate. If a target coverage exceeds this ceiling, that level is marked "unreachable" and reported; the target is not lowered. A sensitivity analysis treats invalid outputs as **incorrect judgments with confidence 0**, ranked last. |
| D3 | **Add AURC** as the curve summary metric | Primary version: the curve runs only up to the maximum achievable coverage, and AURC is computed over that range. Sensitivity version: the curve is extended to 100% coverage using the D2 rule. Both versions are reported. |
| D4 | **Primary endpoint:** confidence-based vs. random abstention at 75% coverage | It covers all 4 configurations × 2 domains, so 8 tests in total, with **Holm correction** across them. The statistic is the paired-bootstrap CI of the selective-risk difference. All other comparisons are secondary (see Section 7). |
| D5 | **Confidence source:** verbalized confidence only | The backend interface reserves an optional `logprobs` field, off by default. Any later logprob analysis is exploratory and goes in a separate `exploratory/` directory, outside the main tables and conclusions. |
| D6 | **Lower bound on judge error rate:** 10% | Estimated on an independent probe sample of at least 100 questions; see Section 6. |
| D7 | **The data team runs the pilot** | Once the smoke test passes, the skeleton goes to the data team. They run the pilot and screen backup datasets and models themselves. |

---

## 3. Data Interface (for the Data Team)

### 3.1 Generic Case Schema

The schema is described by **task type** rather than tied to a specific dataset, so swapping datasets does not require changing it. Each dataset is delivered as one `cases_{dataset_id}.jsonl` file, one case per line:

| Field | Type | Description | Shown to judge? |
|---|---|---|---|
| `case_id` | str | Unique ID | No |
| `question_id` | str | Original question ID, used for bootstrap resampling and duplicate checks | No |
| `dataset_id` | str | e.g. `gsm8k`, `medqa` | No |
| `split` | enum | `pilot` / `probe` / `calib` / `test` | No |
| `task_type` | enum | `numeric` / `multiple_choice`; can later be extended with `short_answer` | No |
| `question` | str | Question text | **Yes** |
| `options` | dict \| null | Answer options for multiple-choice questions | **Yes** |
| `candidate_answer` | str | Answer to be judged | **Yes** |
| `reference_answer` | str | Reference answer | No |
| `label` | bool | Whether the candidate is correct (ground truth) | No |
| `candidate_type` | str | `correct` or an error type such as `omitted_step`, `arithmetic`, `distractor` | No |
| `meta` | dict | Free-form field: construction record, difficulty, subject, etc. | No |

Each data file comes with a `manifest.json` containing its SHA256 hash, the sampling seed, and the count of each split. The pipeline checks the hash at startup and refuses to run on a mismatch.

The data team only needs to write an **adapter** (`adapters/<dataset>.py`) that converts raw data into this format. They do not need to touch any other part of the pipeline.

### 3.2 Commands for the Data Team (no GPU required)

```
validate-data  --data cases.jsonl
smoke          --data cases.jsonl --n 20 --backend mock
probe          --data cases.jsonl --n 100 --backend api --model <cheap-model>
pilot-report   --run outputs/raw/<dataset>/<model>/<config_hash>/
```

| Command | Purpose |
|---|---|
| `validate-data` | Static checks that finish in seconds. See the list of checks below. |
| `smoke` | Runs a small sample through the whole pipeline using a **mock backend** that produces rule-based or random valid outputs: inference, parsing, calibration, policies, metrics, and plots. It checks that the data flows end to end; it does not measure performance. |
| `probe` | Runs a small sample through a cheap API model, a small model, or the real model, and produces a **difficulty report**: judge error rates on correct and incorrect candidates separately, error rates by `candidate_type`, and the confidence distribution. This lets the data team see for themselves whether the constructed errors are too easy or too hard. |
| `pilot-report` | Reads pilot or probe outputs and produces the **gate checklist** (Section 6.1), marking each criterion pass, borderline, or fail, with values and CIs. |

Checks run by `validate-data`:
- schema and field types;
- count of each split, and whether correct and incorrect candidates are roughly balanced;
- whether any question appears in more than one split;
- duplicate questions;
- whether any judge-visible field leaks the label;
- whether numeric answers can be parsed as numbers;
- whether multiple-choice candidates are among the options;
- **whether probe questions overlap with the formal calibration or test questions**.

### 3.3 Bundled Examples and Docs

- `tests/fixtures/`: 10 well-formed examples each for GSM8K and MedQA, plus a set of deliberately malformed examples that show what `validate-data` errors look like.
- `docs/adapter_guide.md`: how to write an adapter.
- `docs/pilot_guide.md`: the pilot runbook (see Section 9.2).

---

## 4. Swappable Architecture: Datasets and Models

### 4.1 Registries

**`registry/datasets.yaml`** records, for each dataset:
- the adapter path;
- the `task_type`;
- the domain label (math / medical);
- the matching rubric snippet.

**`registry/models.yaml`** records, for each model:
- the checkpoint and backend type;
- two decoding profiles, one for thinking on and one for thinking off (see D1);
- the context length and `max_tokens`;
- **thinking capability**: whether it is supported, and how it is toggled (chat-template parameter, system prompt, or a separate checkpoint).

If a replacement model has no thinking toggle, E2 and E4 are automatically marked `n/a` and this is stated in the report. The pipeline does not fail.

### 4.2 Unified Model Backend Interface

```
generate(messages, params) → {
    text, reasoning, usage, finish_reason, latency,
    logprobs   # optional, off by default (see D5)
}
```

| Backend | Use |
|---|---|
| `vllm` | OpenAI-compatible server, used for the main experiments |
| `api` | Hosted API (OpenAI-compatible), used for the DeepSeek replication and the data team's `probe` runs |
| `mock` | Used for unit tests and `smoke`; needs no GPU |

Swapping models only requires changing the YAML; the runner code stays the same.

### 4.3 Prompts Organized by Task Type

- Prompt skeletons are combinations of `task_type` × {minimal, rubric} × {binary, ternary}, stored in `prompts/skeletons/`.
- Domain-specific rubric content is filled into slots and stored in `prompts/rubrics/`:
  - math: numerical equivalence and any required units;
  - medical: consistency with the question and the answer options.
- Moving to another dataset of the same task type only requires adding a rubric snippet.

### 4.4 Output Isolation

- Outputs are written to `outputs/raw/{dataset_id}/{model_id}/{config_hash}/{case_id}.json`.
- Results for different combinations never overwrite each other, and every combination uses the same analysis scripts.

---

## 5. Pipeline Modules

### [A] Config and Prompt Registry

- Each experiment configuration (E1–E5) is one YAML file specifying thinking, rubric, and output mode (binary / ternary). Configurations are independent of datasets and models.
- Every template requires strict JSON output: `{"verdict": "...", "confidence": 0.xx}`.
- The prompt must state what confidence means: **the judge's probability that its own verdict is correct, not the probability that the candidate is correct.**
- Each configuration gets a `config_hash` computed from the YAML plus the full template text, and the hash is written into every output.

### [B] Inference Runner

- vLLM serves an OpenAI-compatible endpoint. Thinking is toggled using the method recorded in the model registry.
- Each case is sampled once, with decoding set according to D1.
- The cache key is `(case_id, config_hash)`. Interrupted runs can resume, and existing results are never overwritten by a re-run.
- Each record stores: the raw output, the reasoning trace, prompt/completion/reasoning token counts, latency, `finish_reason`, a timestamp, and the model version.
- **Retries:** only execution failures (network errors, OOM) are retried, up to N times, and every retry is logged. Formatting errors are not retried, because retrying them would amount to picking favorable results.
- **Freeze check:** if the hashes of the prompts, parser, or decoding parameters differ from the frozen versions, the formal `run` command refuses to execute. `probe` and `screen` are exempt, so the pilot phase can iterate freely.

### [C] Parsing and Validity Classification

| Status | Condition |
|---|---|
| `valid` | JSON parses, verdict is legal, confidence ∈ [0,1] |
| `invalid_format` | Parsing fails or the verdict is illegal |
| `invalid_conf` | Confidence is missing or out of range |
| `truncated` | `finish_reason == length` (watch this closely with thinking on) |
| `exec_failure` | Still failing after retries |
| `abstain` | E5 only; an intentional abstention |

- Parsing rules are frozen after the pilot and covered by unit tests built from real pilot outputs.
- Invalid outputs are handled according to D2.

### [D] Threshold Calibration (calibration split only; 60 cases per model × config × domain)

1. Sort valid judgments by confidence, highest first. **Break ties** with a deterministic, label-independent rule: order by `hash(case_id + fixed salt)`. Verbalized LLM confidence tends to cluster on values like 0.9 and 0.95, so this step is necessary.
2. For target coverage levels of 50%, 75%, and 90% (with all cases as the denominator, per D2), find the corresponding `(τ, hash_cutoff)`.
   - Retention rule: `conf > τ`, or `conf == τ` and `hash < cutoff`.
   - If a target exceeds the achievable ceiling, that level is marked "unreachable".
3. Write the result to `outputs/frozen/thresholds.json`, hash it, and freeze it. It is never modified afterwards.
4. With 60 cases, coverage moves in steps of about 1.7%, so actual test coverage will differ from the targets. This is reported as observed.

### [E] Policy Application (test split only; 140 cases per domain)

| Policy | Procedure |
|---|---|
| No abstention | Retain all valid judgments |
| Confidence-based | Apply the frozen threshold to get the actual coverage `c` and retained count `k` |
| Random | Randomly retain `k` cases from the same test set, repeated R = 10,000 times with fixed seeds; report the expected selective risk and its distribution. In theory its expected value equals the full-coverage error rate, and this analytical result is used to cross-check the code. |
| Direct abstention (E5) | Reported at its observed coverage, with two comparisons: (1) against random abstention for E1 at the same coverage; (2) against the point on E1's confidence-based curve at the same coverage. The second comparison is labeled descriptive, because the coverage match uses the test set. |

### [F] Metrics

| Category | Metrics |
|---|---|
| Primary | Coverage, selective risk, risk–coverage curves (full threshold sweep on test) |
| Curve summary | AURC, in primary and sensitivity versions (see D3); E-AURC optional |
| Secondary | Full-coverage accuracy, Brier score (over all binary judgments, target = whether the verdict is correct), reliability diagrams |
| Operational | Referral volume (cases sent to human review), error-capture rate (judge errors removed by abstention ÷ all judge errors) |
| Bias breakdown | Reported separately for correct and incorrect candidates: false accepts (wrong judged right) and false rejects (right judged wrong) |
| Cost | Mean tokens, reasoning tokens, runtime |
| Validity | Proportion of each status, one table per run |

**Sensitivity analysis:** recompute the primary endpoint and AURC with invalid outputs treated as incorrect judgments at confidence 0.

### [G] Statistical Inference

See Section 7.

### [H] Error Analysis (after real outputs are available)

- Cross-tabulate judge errors by `candidate_type`, e.g. omitted steps and arithmetic mistakes in GSM8K, distractors in MedQA.
- List **high-confidence errors** (conf ≥ 0.9 and wrong) separately; these are where confidence-based abstention fails.
- Sample reasoning traces from the thinking configurations and summarize failure modes, such as re-solving the problem while judging or being swayed by the candidate.

---

## 6. Pilot: Screening and Go/No-Go Gate

The pilot is run by **the data team** (D7). Its goals are to confirm that the inference setup works, and to decide whether the current dataset and model combination can produce meaningful results.

### 6.1 Hard Criteria (checked automatically by `pilot-report`)

Judged on E1 results over the pilot/probe samples; thinking-related items use E2:

| Problem | Failing condition | Why it is a problem |
|---|---|---|
| Judge too accurate | E1 error rate: ≥ 10% passes; 6%–10% is **borderline**, decided by the group; < 6% fails | Too few errors in the test set; risk–coverage comparisons have no statistical power |
| Judge too weak | E1 error rate > 40% | Close to guessing; abstention analysis is meaningless |
| One-sided verdicts | A single verdict accounts for > 85% of outputs | Nearly all errors are of one type |
| Degenerate confidence | More than 70% of outputs share one value | The ranking carries no information |
| Unstable output | After prompt fixes, validity is still < 95% or truncation > 2% | Too many invalid outputs |
| Not reproducible | Verdict agreement between two runs < 98% | Violates the D1 requirement |
| Too expensive | Projected formal-run cost exceeds the compute budget | The runs cannot be completed |

**On sample size:** the 40-question pilot gives only about 20 questions per domain. An error rate of 10% then means just 2 errors, with a 95% CI of roughly [3%, 30%], which cannot settle the gate. The **error-rate gate is therefore estimated on a `probe` sample of at least 100 questions**:
- these questions are drawn from outside the 200 formal questions, with `split = probe`;
- construction standards can be looser than for the formal data;
- `validate-data` confirms they do not overlap with the formal data.

The 40-question pilot sample is still used to check the serving interface, output format, truncation rate, and reproducibility.

### 6.2 Escalation Ladder: Smallest Change First

1. **Adjust data difficulty** without changing datasets: the data team makes errors subtler, e.g. smaller numerical errors or more plausible distractors, then verifies with `probe`.
2. **Swap to another dataset in the same domain**: a harder math dataset for math, another medical-exam multiple-choice dataset for medicine. This keeps the math + medical structure and the research question unchanged. The new dataset needs its own pilot and probe samples first.
3. **Change the model or its size**: a smaller model from the same family if the judge is too accurate; another open model with a thinking toggle if output or thinking behavior is unstable.
4. **Narrow the scope**: as a last resort, e.g. keep only one domain, and explain why in the report.

### 6.3 Screening Mode

```
screen --datasets <a,b> --models <x,y> --configs E1,E2 --split probe
```

Runs several dataset × model combinations at once, each on probe/pilot samples with configurations E1 and E2 only, and outputs a comparison table of the hard criteria so the group can choose a combination.

### 6.4 Rules Against Cherry-Picking

- **Swaps are allowed only during the pilot.** Once the formal calibration and test runs start, the dataset, model, prompts, parser, and decoding parameters are all locked, enforced by the freeze check in [B].
- If a swap turns out to be necessary after formal runs have started, **rerun all calibration and test cases** on the new combination; partial swaps are not allowed.
- Every swap is recorded in `decision_log.md`: date, triggering criterion and value, what was swapped in, and who approved it. The report includes the failed results of the original combination.

### 6.5 What Gets Frozen After the Pilot

Prompts, configs, parser version, decoding parameters, and `max_tokens`, all hashed and stored in `outputs/frozen/`.

---

## 7. Statistical Analysis and Pre-registration

### 7.1 Primary Endpoint (pre-registered)

- **Selective-risk difference between confidence-based and random abstention at 75% coverage.**
- Scope: E1–E4 × {GSM8K, MedQA}, 8 tests in total.
- Method: paired bootstrap with B = 10,000, resampling test questions stratified by candidate correctness; report two-sided 95% CIs, with **Holm correction** across the 8 tests.
- Written into `preregistration.md` and timestamped before the formal runs.

### 7.2 Secondary Comparisons (CIs only; no correction and no significance claims)

- Confidence-based vs. random abstention at the 50% and 90% coverage levels.
- Both versions of AURC, and AURC differences between configurations (bootstrap CIs).
- Full-coverage accuracy of E2, E3, and E4 relative to E1: exact McNemar test plus bootstrap CIs.
- The thinking × rubric interaction.
- Differences between domains.
- The two E5 direct-abstention comparisons (see [E]).
- Brier score and calibration.

### 7.3 Sensitivity and Exploratory Analyses

- **Sensitivity:** invalid outputs treated as incorrect judgments at confidence 0 (D2, D3).
- **Exploratory** (optional, resources permitting): verdict-token logprobs as confidence (D5), with results kept in `exploratory/`.

### 7.4 Implementation Requirements

- All configurations compute their differences on **the same set of resampling indices**, so comparisons stay paired.
- Every bootstrap and random-baseline seed is fixed and recorded.

---

## 8. Code Structure

```
dsc261-llm-judge/
├── registry/
│   ├── datasets.yaml
│   └── models.yaml            # two decoding profiles, thinking capability, logprobs switch
├── adapters/                  # written by the data team: gsm8k.py, medqa.py, <new_dataset>.py
├── configs/                   # E1–E5, independent of dataset and model
├── prompts/
│   ├── skeletons/             # task_type × minimal/rubric × binary/ternary
│   └── rubrics/               # per-domain slot snippets
├── src/
│   ├── schema.py              # Case definition + validation
│   ├── backends/              # vllm.py, api.py, mock.py
│   ├── runner.py              # [B] inference + cache + resume + freeze check
│   ├── parse.py               # [C]
│   ├── calibrate.py           # [D] thresholds + tie-breaking + unreachable detection
│   ├── policies.py            # [E] four policies
│   ├── metrics.py             # [F] including both AURC versions
│   ├── stats.py               # [G] paired bootstrap, Holm, McNemar
│   ├── gates.py               # Section 6.1 hard-criteria checks
│   └── cli.py                 # validate-data / smoke / probe / screen / pilot-report / run / evaluate / report
├── tests/
│   ├── fixtures/              # well-formed + deliberately malformed examples
│   └── test_*.py              # parser, tie-breaking, random-baseline analytical check, AURC, invalid-output handling
├── outputs/
│   ├── raw/{dataset}/{model}/{config_hash}/
│   ├── frozen/                # thresholds.json, frozen prompts and config hashes
│   └── exploratory/           # logprob and other exploratory results
├── reports/                   # figures, tables
├── docs/
│   ├── experiment_pipeline_plan.md      # Chinese version
│   ├── experiment_pipeline_plan_en.md   # this document
│   ├── adapter_guide.md
│   └── pilot_guide.md
├── preregistration.md
├── decision_log.md
└── Makefile
```

Every run automatically produces a `run_manifest.json` recording the git commit, model checksum, vLLM version, batch size, config_hash, data hash, and start and end times.

---

## 9. Deliverables and Handoff

### 9.1 Skeleton-Phase Deliverables

**In scope:**
- all CLI commands listed in Sections 3.2 and 6.3;
- the mock, vLLM, and API backends, with the reserved `logprobs` field;
- example GSM8K and MedQA adapters, plus fixtures;
- parsing, calibration, the four policies, metrics (including both AURC versions), paired bootstrap with Holm correction, and gate checks;
- `pilot-report` and the freeze check;
- unit tests for all of the above;
- `adapter_guide.md`, `pilot_guide.md`, and templates for `decision_log.md` and `preregistration.md`.

**Acceptance criteria:** using the mock backend on the fixtures, the full flow `validate-data → smoke → pilot-report → run → evaluate → report` runs end to end, and all unit tests pass.

**Out of scope:**
- real model inference, which belongs to the data team's pilot;
- formal data construction;
- error-analysis notebooks, which wait for real outputs.

### 9.2 Pilot Workflow After Handoff (`docs/pilot_guide.md`)

1. Write the adapter and generate `cases_*.jsonl`, including the pilot and probe splits.
2. Run `validate-data` and fix every error.
3. Run `smoke --backend mock` to confirm the data runs end to end.
4. Configure a real backend from the templates: either the vLLM launch example for Qwen, or an API backend.
5. Run `probe` and the pilot with E1 and E2, running the reproducibility check twice.
6. Run `pilot-report` and review the gate checklist.
7. If every criterion passes, freeze the settings and hand off to the experiment leads for the formal runs. If any fails, escalate following Section 6.2, compare candidate combinations with `screen`, and fill in `decision_log.md`.
