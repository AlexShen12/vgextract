"""Explicit observer and solver contracts. No source metadata is a model input."""
import json

SOLVER = """Answer a retrospective visual-memory question using only this ordered video
sequence, its source timestamps, and permitted patch-specific static references. Do not
infer unseen events from game strategy or remembered mechanics. You have no construction
logs or external memory. If evidence is missing, abstain. Return JSON only after any thinking:
{"answer": <typed value requested>, "abstain": false,
"evidence":[{"start": <source seconds>, "end": <source seconds>, "claim":"visible fact"}]}
Evidence intervals must refer to supplied source-video times. Use numbers for numeric/time
answers, strings for identities, lists/objects as requested. A static reference establishes
mechanics or identity only; it never establishes a match event.
"""


def solver_messages(question: dict, frames: list[dict]) -> list[dict]:
    """Allowlist only: intentionally cannot serialize the reference answer or its evidence."""
    payload = {"question": question["prompt"], "answer_type": question["answer_type"],
               "interval": question["interval"], "knowledge_category": question["knowledge_category"],
               "frames": [{"time": f["time"], "id": f["id"]} for f in frames]}
    return [{"role": "system", "content": SOLVER},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}]
