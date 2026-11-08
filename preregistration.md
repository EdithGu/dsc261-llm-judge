# Pre-registration — When Should an LLM Judge Abstain?

> Fill in and commit BEFORE the first formal `run`. The commit timestamp is the registration timestamp.
> After that, this file is append-only (amendments go at the bottom, dated, with a reason).

- Registered on (date / commit): `____`
- Authors: `____`
- Frozen lock: `outputs/frozen/lock.json` sha256 `____`
- Data manifests: `gsm8k` sha256 `____` ; `medqa` sha256 `____`

## 1. Primary endpoint

- Quantity: selective risk (confidence-based abstention) − selective risk (random abstention), at **75 % target coverage**
  (coverage denominator = all cases, D2), thresholds calibrated on calib (60) and frozen before test (140) is touched.
- Random-abstention selective risk = its exact expectation (error rate of the valid pool); the 10 000-draw
  simulation is reported as a cross-check.
- Scope: E1–E4 × {GSM8K, MedQA} = **8 tests**.
- Inference: paired bootstrap, B = 10 000, resampling test questions (by `question_id`) stratified by candidate
  correctness, same index matrix for all configs of a domain; two-sided 95 % percentile CI; bootstrap p-value;
  **Holm** correction across the 8 tests at α = 0.05. An unreachable target enters Holm as p = 1.
- Seeds: bootstrap `20261107`, random baseline `12345`, tie-break salt `dsc261-tiebreak-v1`.

## 2. Secondary comparisons (CIs only; no correction; no significance claims)

- Same contrast at 50 % and 90 % coverage.
- AURC (primary: valid judgments up to max coverage; sensitivity: invalid = confidence-0 errors, ranked last) and
  differences vs E1.
- Full-coverage accuracy of E2, E3, E4 vs E1: McNemar exact + bootstrap CI.
- Thinking × rubric interaction: (E4 − E3) − (E2 − E1), for accuracy and AURC.
- Domain differences (unpaired bootstrap).
- E5 direct abstention: vs E1 random abstention at the same k; vs E1 confidence curve at the same coverage
  (descriptive — coverage is matched on test).
- Brier score and reliability diagrams.

## 3. Sensitivity and exploratory analyses

- Sensitivity: invalid outputs treated as wrong judgments with confidence 0 ranked last; primary endpoint and AURC recomputed.
- Exploratory (optional): verdict-token logprobs as confidence → `outputs/exploratory/` only, never in main tables.

## 4. Decision rules fixed in advance

- Invalid outputs are neither retained nor abstentions (D2). Unreachable coverages are reported, never relaxed.
- One sample per case; execution failures retried ≤ N times; format errors never retried.
- Any change of dataset / model / prompt / parser / decoding after formal runs start ⇒ full re-run of calib + test
  and an entry in `decision_log.md`; original results are still reported.

## Amendments

<!-- YYYY-MM-DD — what changed — why — who approved -->
