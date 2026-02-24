import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils import extract_tag, stringify


def build_verified_filename(json_file_name: str, greedy: bool = False) -> str:
    # Build output filename next to input, appending 'verified.json' (and optional greedy tag).
    dirpath = os.path.dirname(json_file_name) or "."
    base = json_file_name[:-5] if json_file_name.lower().endswith(".json") else json_file_name

    if greedy:
        fname = os.path.basename(base) + "_greedy_verified.json"
        return os.path.join(dirpath, fname)

    return os.path.join(dirpath, os.path.basename(base) + "verified.json")


def normalize_for_match(s: str) -> str:
    # Normalize a string for matching (lower + whitespace normalization).
    s = stringify(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_answer_from_option_text(opt_text: str) -> str:
    # Extract an 'Answer:' field if present; else fallback to a compact answer string.
    t = stringify(opt_text).strip()
    if not t:
        return ""
    m = re.search(r"(?is)\banswer\s*:\s*(.*?)(?:\bexplanation\s*:|$)", t)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r"\s+", " ", ans).strip(" \n\t,;")
        return ans
    if "," in t:
        return t.split(",", 1)[0].strip()
    return t.strip()


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


def verifier_choose_best(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
    question: str,
    options: List[Any],
    greedy: bool,
) -> Tuple[str, Optional[int], List[Any]]:
    # Ask OpenAI to select the best final answer from provided options.
    if not isinstance(options, list):
        options = []

    if greedy:
        options = options[:1]

    formatted = []
    norm_option_texts = []
    norm_option_answers = []

    for i, opt in enumerate(options):
        block = stringify(opt).strip()
        if not block:
            continue
        formatted.append(f"{i+1}. {block}")
        norm_option_texts.append((i + 1, normalize_for_match(block)))
        norm_option_answers.append((i + 1, normalize_for_match(extract_answer_from_option_text(block))))

    if not formatted:
        formatted = ["1. "]
        norm_option_texts = [(1, "")]
        norm_option_answers = [(1, "")]

    options_block = "\n\n".join(formatted)

    prompt = f'''
You are a verifier. Your job: choose the best final answer for the question using the candidate options below.
Each option contains an Answer and an Explanation. The Explanation may be wrong; use your own judgment.

You MUST choose your final answer ONLY from the provided options.
Do NOT invent a new answer that is not explicitly present in the options.

You may reason privately, but your output MUST be exactly:
<final>...</final>

Rules (STRICT):
- Inside <final>, output ONLY the final answer as ONE short expression (e.g., a name, a year, a number, a place).
- If none of the options are correct, still pick the closest/best option Answer by copying ONLY the shortest correct-looking expression from one option (do not create a new answer).
- Do NOT output a sentence.
- Do NOT add extra words like "The answer is", "It is", or any explanation.
- Your <final> must match exactly one of the option Answers (copy it verbatim).

Now do it:

Question: {question}

Options:
{options_block}

Output:
'''.strip()

    text = openai_call(client, model, prompt, max_output_tokens=max_output_tokens)

    final = extract_tag(text, "final")
    if not final:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        final = (lines[0] if lines else "").strip()
    final = stringify(final).strip()

    final_norm = normalize_for_match(final)
    chosen_option_num = None

    for num, ans_norm in norm_option_answers:
        if ans_norm and ans_norm == final_norm:
            chosen_option_num = num
            break

    if chosen_option_num is None:
        for num, opt_norm in norm_option_texts:
            if opt_norm and final_norm and final_norm in opt_norm:
                chosen_option_num = num
                break

    return final, chosen_option_num, options


def verify_file(
    in_path: str,
    out_path: str,
    openai_api_key: str,
    verifier_model: str = "gpt-5.2",
    max_output_tokens: int = 200,
    greedy: bool = False,
) -> None:
    # Verify a generation JSON file and write a verified JSON file.
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in env var OPENAI_API_KEY.")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples")

    client = OpenAI(api_key=openai_api_key)

    verified: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        qid = item.get("id", i)
        try:
            qid = int(qid)
        except Exception:
            qid = i

        question = stringify(item.get("question", "")).strip()
        correct = item.get("correct", None)
        options = item.get("generated", item.get("options", []))

        final_answer, chosen_option_num, used_options = verifier_choose_best(
            client=client,
            model=verifier_model,
            max_output_tokens=max_output_tokens,
            question=question,
            options=options,
            greedy=greedy,
        )

        options_out = used_options[:1] if greedy else used_options

        verified.append(
            {
                "id": qid,
                "question": question,
                "options": options_out,
                "final_answer": final_answer,
                "chosen_option_num": chosen_option_num,
                "correct": correct,
            }
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verified, f, ensure_ascii=False, indent=2)
