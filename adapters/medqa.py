"""Example adapter: MedQA (USMLE, 4/5 options) rows (+ constructed candidates) -> generic case schema.

Expected raw row (one per candidate):
    {
      "id": "test-0137",
      "question": "A 45-year-old man presents with ...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer_idx": "C",                        # original MedQA field
      "candidate_answer": "B",                  # option letter the judge will see
      "candidate_type": "correct" | "distractor" | ...,
      "split": "pilot" | "probe" | "calib" | "test",
      "candidate_id": "c1",                     # optional
      "meta_info": "step1"                      # optional original field
    }

Run:  llm-judge adapt --dataset medqa --raw raw_medqa.jsonl --out data/medqa/cases_medqa.jsonl --seed 0
"""

DATASET_ID = "medqa"
TASK_TYPE = "multiple_choice"


def convert(raw_rows: list[dict], seed: int = 0) -> list[dict]:
    out = []
    for i, r in enumerate(raw_rows):
        qid = f"{DATASET_ID}-{r['id']}"
        cand_id = r.get("candidate_id") or f"c{i}"
        ctype = r["candidate_type"]
        opts = {str(k).strip().upper(): str(v).strip() for k, v in r["options"].items()}
        cand = str(r["candidate_answer"]).strip().upper()
        ref = str(r["answer_idx"]).strip().upper()
        out.append({
            "case_id": f"{qid}-{cand_id}",
            "question_id": qid,
            "dataset_id": DATASET_ID,
            "split": r["split"],
            "task_type": TASK_TYPE,
            "question": r["question"].strip(),
            "options": opts,
            "candidate_answer": cand,
            "reference_answer": ref,
            "label": ctype == "correct",
            "candidate_type": ctype,
            "meta": {"source_id": r["id"], "meta_info": r.get("meta_info")},
        })
    return out
