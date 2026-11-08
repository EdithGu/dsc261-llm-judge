"""Example adapter: GSM8K-style rows (+ constructed candidates) -> generic case schema.

Expected raw row (one per candidate; the data group builds the candidates):
    {
      "id": "train-00042",                      # original GSM8K item id
      "question": "...",
      "answer": "... worked solution ... #### 18",  # original GSM8K answer field
      "candidate_answer": "... #### 17",        # the answer the judge will see
      "candidate_type": "correct" | "arithmetic" | "omitted_step" | ...,
      "split": "pilot" | "probe" | "calib" | "test",
      "candidate_id": "c1",                     # optional, distinguishes several candidates per question
      "meta": {...}                             # optional
    }

Run:  llm-judge adapt --dataset gsm8k --raw raw_gsm8k.jsonl --out data/gsm8k/cases_gsm8k.jsonl --seed 0
"""

DATASET_ID = "gsm8k"
TASK_TYPE = "numeric"


def reference_from_solution(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"GSM8K answer has no '####' final answer: {answer[:80]!r}")
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def convert(raw_rows: list[dict], seed: int = 0) -> list[dict]:
    out = []
    for i, r in enumerate(raw_rows):
        qid = f"{DATASET_ID}-{r['id']}"
        cand_id = r.get("candidate_id") or f"c{i}"
        ctype = r["candidate_type"]
        out.append({
            "case_id": f"{qid}-{cand_id}",
            "question_id": qid,
            "dataset_id": DATASET_ID,
            "split": r["split"],
            "task_type": TASK_TYPE,
            "question": r["question"].strip(),
            "options": None,
            "candidate_answer": r["candidate_answer"].strip(),
            "reference_answer": reference_from_solution(r["answer"]),
            "label": ctype == "correct",
            "candidate_type": ctype,
            "meta": {"source_id": r["id"], **(r.get("meta") or {})},
        })
    return out
