import numpy as np
import torch
from typing import Any, Dict, List

from hf import build_chat_text


@torch.inference_mode()
def beam_search_record(
    tokenizer,
    model,
    is_gemma: bool,
    question: str,
    correct: Any,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    qid: int,
) -> Dict[str, Any]:
    # Generate num_beams candidates using deterministic beam search and pick best by sequence score.
    if is_gemma:
        user_prompt = "Give a short accurate answer (without additional text).\n\nQuestion: " + question
        messages = [{"role": "user", "content": user_prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Give a short accurate answer (without additional text)."},
            {"role": "user", "content": question},
        ]

    chat_text = build_chat_text(tokenizer, messages)
    input_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = int(input_ids.shape[-1])

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        length_penalty=float(length_penalty),
        early_stopping=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    seqs = out.sequences
    seq_scores = out.sequences_scores
    if seq_scores is None:
        seq_scores = torch.zeros((seqs.shape[0],), device=seqs.device, dtype=torch.float32)

    options: List[str] = []
    scores: List[float] = []

    for j in range(seqs.shape[0]):
        txt = tokenizer.decode(seqs[j][prompt_len:], skip_special_tokens=True).strip()
        options.append(txt)
        scores.append(float(seq_scores[j].item()))

    best_idx = int(np.argmax(np.array(scores, dtype=np.float64)))

    return {
        "id": int(qid),
        "question": question,
        "options": options,
        "final_answer": options[best_idx],
        "chosen_option_num": best_idx + 1,
        "correct": correct,
    }
