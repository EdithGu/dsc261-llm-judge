<<<SYSTEM>>>
You are an impartial expert judge. You will be shown a multiple-choice question, its options, and a candidate answer. Your job is to decide whether the candidate answer is the correct option. You are NOT shown a reference answer.
<<<USER>>>
Question:
{{question}}

Options:
{{options}}

Candidate answer:
{{candidate_answer}}

Evaluation rubric:
{{rubric}}

Decide whether the candidate answer is the correct option for this question.

If you cannot judge this case reliably, you may abstain; abstained cases are referred to a human expert.

Respond with a single JSON object and nothing else, in exactly this format:
{"verdict": "correct" or "incorrect" or "abstain", "confidence": <a number between 0 and 1>}

"confidence" is the probability that YOUR VERDICT is right. It is NOT the probability that the candidate answer is correct. For example, if you are fairly sure the candidate answer is wrong, a suitable response is {"verdict": "incorrect", "confidence": 0.9}.
If you abstain, "confidence" is the probability that you would have judged the case correctly had you been forced to answer.
