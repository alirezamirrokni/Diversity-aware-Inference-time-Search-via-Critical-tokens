import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from hf import build_chat_text


def top_k_top_p_filter(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    # Apply top-k and/or nucleus (top-p) filtering to a probability vector.
    p = probs

    if top_k and top_k > 0:
        _, topk_i = torch.topk(p, min(top_k, p.numel()))
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[topk_i] = True
        p = torch.where(mask, p, torch.zeros_like(p))

    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(p, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=-1)
        keep = cumsum <= top_p
        if keep.numel() > 0:
            keep[0] = True
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[sorted_i[keep]] = True
        p = torch.where(mask, p, torch.zeros_like(p))

    s = p.sum()
    if s.item() <= 0:
        return probs
    return p / s


@torch.inference_mode()
def sample_generate_with_logprob(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[List[int], float]:
    # Sample tokens autoregressively and accumulate the log-probability of sampled tokens.
    device0 = input_ids.device
    past = None

    gen_ids: List[int] = []
    logprob = 0.0

    for _ in range(max_new_tokens):
        if past is None:
            step_inp = input_ids
        else:
            step_inp = torch.tensor([[gen_ids[-1]]], device=device0, dtype=input_ids.dtype)

        out = model(input_ids=step_inp, past_key_values=past, use_cache=True)
        past = out.past_key_values

        logits = out.logits[:, -1, :].float().squeeze(0)
        if temperature and temperature != 1.0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)
        probs = top_k_top_p_filter(probs, top_k, top_p)

        nxt = int(torch.multinomial(probs, num_samples=1).item())
        p = float(probs[nxt].item())

        gen_ids.append(nxt)
        logprob += math.log(max(p, 1e-45))

        if nxt == tokenizer.eos_token_id:
            break

    return gen_ids, logprob


@torch.inference_mode()
def bon_record(
    tokenizer,
    model,
    is_gemma: bool,
    question: str,
    correct: Any,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    qid: int,
) -> Dict[str, Any]:
    # Sample num_samples completions and pick the best by accumulated logprob.
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

    options: List[str] = []
    scores: List[float] = []

    for _ in range(num_samples):
        gen_ids, lp = sample_generate_with_logprob(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        options.append(ans)
        scores.append(float(lp))

    best_idx = int(np.argmax(np.array(scores, dtype=np.float64)))

    return {
        "id": int(qid),
        "question": question,
        "options": options,
        "final_answer": options[best_idx],
        "chosen_option_num": best_idx + 1,
        "correct": correct,
    }
