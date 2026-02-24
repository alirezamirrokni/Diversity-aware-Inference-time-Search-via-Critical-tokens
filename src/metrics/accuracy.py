import os
import json
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

from utils import extract_tag, stringify


def get_correct_string(correct: Any) -> str:
    # Extract a canonical gold string from the TriviaQA 'answer' object.
    if isinstance(correct, dict):
        for k in ["normalized_value", "value", "text", "answer"]:
            if k in correct and correct[k] is not None:
                return str(correct[k])
        if "aliases" in correct and isinstance(correct["aliases"], list) and correct["aliases"]:
            return str(correct["aliases"][0])
        if "normalized_aliases" in correct and isinstance(correct["normalized_aliases"], list) and correct["normalized_aliases"]:
            return str(correct["normalized_aliases"][0])
        return ""
    return str(correct) if correct is not None else ""


def gold_aliases_from_correct(correct: Any) -> List[str]:
    # Extract a de-duplicated list of gold aliases.
    if not isinstance(correct, dict):
        return []
    aliases: List[str] = []

    if isinstance(correct.get("aliases"), list):
        aliases.extend([stringify(a).strip() for a in correct["aliases"] if stringify(a).strip()])

    if isinstance(correct.get("normalized_aliases"), list):
        aliases.extend([stringify(a).strip() for a in correct["normalized_aliases"] if stringify(a).strip()])

    seen = set()
    out: List[str] = []
    for a in aliases:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def openai_call(client: OpenAI, model: str, prompt: str, max_output_tokens: int) -> str:
    # Call OpenAI Responses API with retry/backoff.
    delay = 1.0
    for _ in range(8):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            return (resp.output_text or "").strip()
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    raise RuntimeError("OpenAI API failed repeatedly.")


def build_judge_prompt(
    question: str,
    predicted: str,
    gold: str,
    gold_aliases: List[str],
    allow_external_knowledge: bool,
) -> str:
    # Build an LLM-as-a-judge prompt for correctness.
    aliases_block = "\n".join(f"- {a}" for a in gold_aliases[:30]) if gold_aliases else ""
    policy = "You may use your own world knowledge." if allow_external_knowledge else "You must judge using only the provided gold answer and aliases."
    return f"""
You are an LLM-as-a-judge. Determine whether the model's predicted answer is correct for the question.

{policy}

You must output EXACTLY:
<verdict>correct</verdict> or <verdict>incorrect</verdict>

Rules:
- Be strict: the predicted answer must refer to the same entity/value as the gold answer.
- Accept aliases, common abbreviations, punctuation/case differences, and minor formatting differences.
- If the prediction is a sentence, consider whether it clearly states the correct answer.

Question: {question}

Gold answer (canonical): {gold}

Gold aliases (non-exhaustive):
{aliases_block}

Predicted answer: {predicted}
""".strip()


def judge_verdict(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
    allow_external_knowledge: bool,
    question: str,
    predicted: str,
    correct_obj: Any,
) -> str:
    # Use OpenAI to judge whether predicted answer is correct.
    gold = stringify(get_correct_string(correct_obj)).strip()
    aliases = gold_aliases_from_correct(correct_obj)
    prompt = build_judge_prompt(question, predicted, gold, aliases, allow_external_knowledge)
    text = openai_call(client, model, prompt, max_output_tokens=max_output_tokens)
    verdict = extract_tag(text, "verdict").strip().lower()
    if verdict not in {"correct", "incorrect"}:
        verdict = "incorrect"
    return verdict


def evaluate_file(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
    allow_external_knowledge: bool,
    in_path: str,
    out_path: str,
) -> Dict[str, Any]:
    # Judge accuracy for a single verified JSON file and write *_judge_accuracy.json.
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {"error": "JSON is not a list"}

    n = 0
    hits = 0
    judgments: List[Dict[str, Any]] = []

    for i, item in enumerate(tqdm(data, desc=os.path.basename(in_path), leave=False)):
        if not isinstance(item, dict):
            continue
        n += 1

        qid = item.get("id", i)
        question = stringify(item.get("question", "")).strip()
        predicted = stringify(item.get("final_answer", item.get("best", ""))).strip()
        if not predicted:
            predicted = stringify(item.get("best", "")).strip()

        verdict = judge_verdict(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            allow_external_knowledge=allow_external_knowledge,
            question=question,
            predicted=predicted,
            correct_obj=item.get("correct", None),
        )
        if verdict == "correct":
            hits += 1

        judgments.append(
            {
                "id": qid,
                "question": question,
                "predicted": predicted,
                "verdict": verdict,
                "correct": item.get("correct", None),
            }
        )

    if n == 0:
        return {"error": "No valid items"}

    metrics = {"n": n, "accuracy": hits / n}
    out_json = {"file": os.path.basename(in_path), "metrics": metrics, "judgments": judgments}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return metrics
