import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from hf import build_chat_text
from utils import safe_parse_list


def delete_proper_nouns_exist_in_the_question(proper_nouns: List[str], question: str) -> List[str]:
    # Remove extracted proper nouns that already occur in the question.
    return [pn for pn in proper_nouns if pn not in question]


def find_proper_noun_token_ranges(generated_tokens: List[str], proper_nouns: List[str]) -> Dict[str, List[int]]:
    # Map each proper noun span to the list of token indices that overlap it (char-based alignment).
    results: Dict[str, List[int]] = {}

    text_so_far = ""
    char_to_token: Dict[int, int] = {}
    current_pos = 0

    for token_idx, token in enumerate(generated_tokens):
        start = current_pos
        end = current_pos + len(token)

        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx

        current_pos = end
        text_so_far += token

    for proper_noun in proper_nouns:
        start_char = text_so_far.find(proper_noun)
        if start_char == -1:
            results[proper_noun] = []
            continue

        end_char = start_char + len(proper_noun)
        contributing_tokens = set()

        for char_pos in range(start_char, end_char):
            if char_pos in char_to_token:
                contributing_tokens.add(char_to_token[char_pos])

        results[proper_noun] = sorted(contributing_tokens)

    return results


def find_clustrable_token(
    proper_nouns_positions: Dict[str, List[int]],
    tokens_confidence: Dict[int, float],
) -> int | None:
    # Choose the first token of the least-confident proper noun span (sum log probs), else None.
    proper_nouns_confidence: Dict[str, float] = {}

    for pn, positions in proper_nouns_positions.items():
        if not positions:
            continue

        s = 0.0
        ok = True

        for position in positions:
            if position not in tokens_confidence:
                ok = False
                break

            p = tokens_confidence[position]
            if p <= 0:
                ok = False
                break

            s += math.log(p)

        if ok:
            proper_nouns_confidence[pn] = s

    if proper_nouns_confidence:
        min_proper_noun = min(proper_nouns_confidence.keys(), key=lambda k: proper_nouns_confidence[k])
        return proper_nouns_positions[min_proper_noun][0]

    return None


@dataclass
class OursConfig:
    # Configuration for clustered-branch generation.
    max_new_tokens: int = 100
    top_k: int = 100
    cluster_k: int = 5
    num_new_samples: int = 4
    sample_cluster: bool = True
    sample_within_cluster: bool = False
    layer_idx: int | None = None
    emb_batch: int = 4


class OursGenerator:
    def __init__(self, tokenizer, model, is_gemma: bool, cfg: OursConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.is_gemma = is_gemma
        self.cfg = cfg

        num_hidden_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise RuntimeError("model.config.num_hidden_layers is missing; cannot infer last layer index.")

        if self.cfg.layer_idx is None:
            self.cfg.layer_idx = int(num_hidden_layers)

    @torch.inference_mode()
    def extract_proper_nouns_from_sentence(self, text: str) -> str:
        # Use the model to extract proper nouns from a sentence (returned as a Python list literal).
        prompt = f'''
You are a helpful assistant. Your task is to extract all proper nouns from a given sentence. Each proper noun should be returned exactly as they appear in the sentence.
If no proper nouns are found, return an empty list ([]).
Do not return additional text (just a list of proper nouns or empty list (if no proper nouns are found)).

Example Input:
"Elon Musk visited NASA headquarters in Washington last week."

Example Output:
["Elon Musk", "NASA", "Washington"]

Input Sentence:
{text}

Your Output:
'''.strip()

        if self.is_gemma:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

        chat_text = build_chat_text(self.tokenizer, messages)
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=128)
        gen_only = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def greedy_generate_with_cache(self, input_ids: torch.Tensor) -> Tuple[List[int], Dict[int, float], float]:
        # Greedy decode while tracking per-step confidence and total logprob.
        device0 = input_ids.device
        past = None

        gen_token_ids: List[int] = []
        conf: Dict[int, float] = {}

        for t in range(self.cfg.max_new_tokens):
            if past is None:
                step_inp = input_ids
            else:
                step_inp = torch.tensor([[gen_token_ids[-1]]], device=device0, dtype=input_ids.dtype)

            out = self.model(input_ids=step_inp, past_key_values=past, use_cache=True)
            past = out.past_key_values

            logits = out.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.argmax(probs, dim=-1).item())

            conf[t] = float(probs[0, nxt].item())
            gen_token_ids.append(nxt)

            if nxt == self.tokenizer.eos_token_id:
                break

        logprob = 0.0
        for t in range(len(gen_token_ids)):
            logprob += math.log(max(conf[t], 1e-45))

        return gen_token_ids, conf, logprob

    @torch.inference_mode()
    def prepare_branch_state(
        self,
        input_ids: torch.Tensor,
        branch_index: int,
    ) -> Tuple[Any, torch.Tensor, List[int], float, torch.Tensor]:
        # Recompute cache up to branch_index and return state at branching position.
        device0 = input_ids.device

        prefix_ids: List[int] = []
        prefix_logprob = 0.0

        out = self.model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        if branch_index == 0:
            return past, logits, prefix_ids, prefix_logprob, input_ids

        probs = torch.softmax(logits, dim=-1)
        last_token = int(torch.argmax(probs, dim=-1).item())

        prefix_ids.append(last_token)
        prefix_logprob += math.log(max(probs[0, last_token].item(), 1e-45))

        for t in range(1, branch_index + 1):
            step_inp = torch.tensor([[last_token]], device=device0, dtype=input_ids.dtype)
            out = self.model(input_ids=step_inp, past_key_values=past, use_cache=True)

            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

            if t == branch_index:
                context_ids = torch.cat(
                    [input_ids, torch.tensor([prefix_ids], device=device0, dtype=input_ids.dtype)],
                    dim=-1,
                )
                return past, logits, prefix_ids, prefix_logprob, context_ids

            probs = torch.softmax(logits, dim=-1)
            last_token = int(torch.argmax(probs, dim=-1).item())

            prefix_ids.append(last_token)
            prefix_logprob += math.log(max(probs[0, last_token].item(), 1e-45))

        context_ids = torch.cat(
            [input_ids, torch.tensor([prefix_ids], device=device0, dtype=input_ids.dtype)],
            dim=-1,
        )
        return past, logits, prefix_ids, prefix_logprob, context_ids

    @torch.inference_mode()
    def compute_topk_clusters(
        self,
        context_ids: torch.Tensor,
        logits: torch.Tensor,
        greedy_token_id: int,
    ) -> Dict[str, Any]:
        # Cluster top-k next-token embeddings at cfg.layer_idx and return cluster stats.
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        topk_probs, topk_indices = torch.topk(probs, self.cfg.top_k)

        device0 = context_ids.device
        topk_indices_dev = topk_indices.to(device0)

        embeddings_parts = []
        for start in range(0, self.cfg.top_k, self.cfg.emb_batch):
            end = min(self.cfg.top_k, start + self.cfg.emb_batch)

            cand = topk_indices_dev[start:end].view(-1, 1)
            base = context_ids.repeat(cand.size(0), 1)
            batch_inp = torch.cat([base, cand], dim=1)

            out = self.model(input_ids=batch_inp, output_hidden_states=True)
            hs = out.hidden_states[self.cfg.layer_idx][:, -1, :].to(torch.float32).detach().cpu().numpy()
            embeddings_parts.append(hs)

        embeddings = np.concatenate(embeddings_parts, axis=0)

        kmeans = KMeans(n_clusters=self.cfg.cluster_k, random_state=0, n_init=10).fit(embeddings)
        cluster_labels = kmeans.labels_

        greedy_pos = (topk_indices == greedy_token_id).nonzero(as_tuple=False)
        greedy_cluster = int(cluster_labels[0]) if greedy_pos.numel() == 0 else int(cluster_labels[int(greedy_pos[0].item())])

        topk_probs_cpu = topk_probs.to(torch.float32).detach().cpu().numpy().astype(np.float64)

        cluster_prob_sums = np.zeros(self.cfg.cluster_k, dtype=np.float64)
        for c in range(self.cfg.cluster_k):
            cluster_prob_sums[c] = float(np.sum(topk_probs_cpu[cluster_labels == c]))

        return {
            "topk_indices": topk_indices.detach().cpu().numpy().astype(np.int64),
            "topk_probs": topk_probs_cpu,
            "cluster_labels": cluster_labels.astype(np.int64),
            "cluster_prob_sums": cluster_prob_sums,
            "greedy_cluster": greedy_cluster,
        }

    def choose_cluster_and_token(self, cluster_info: Dict[str, Any]) -> Tuple[int, float]:
        # Choose an alternative token from a non-greedy cluster based on cluster mass.
        topk_indices = cluster_info["topk_indices"]
        topk_probs = cluster_info["topk_probs"]
        cluster_labels = cluster_info["cluster_labels"]
        cluster_prob_sums = cluster_info["cluster_prob_sums"]
        greedy_cluster = cluster_info["greedy_cluster"]

        cluster_list = [c for c in range(len(cluster_prob_sums)) if c != greedy_cluster]
        if not cluster_list:
            return int(topk_indices[0]), float(topk_probs[0])

        weights = cluster_prob_sums[cluster_list].astype(np.float64)

        if self.cfg.sample_cluster:
            s = float(weights.sum())
            chosen_cluster = int(cluster_list[0]) if s <= 0 else int(np.random.choice(cluster_list, p=weights / s))
        else:
            chosen_cluster = int(cluster_list[0]) if float(weights.sum()) <= 0 else int(cluster_list[int(np.argmax(weights))])

        idxs = np.where(cluster_labels == chosen_cluster)[0]
        if idxs.size == 0:
            return int(topk_indices[0]), float(topk_probs[0])

        probs_in = topk_probs[idxs].astype(np.float64)

        if self.cfg.sample_within_cluster:
            s = float(probs_in.sum())
            chosen_local = int(idxs[0]) if s <= 0 else int(np.random.choice(idxs, p=probs_in / s))
        else:
            chosen_local = int(idxs[int(np.argmax(probs_in))])

        return int(topk_indices[chosen_local]), float(topk_probs[chosen_local])

    @torch.inference_mode()
    def continue_greedy_from_past(
        self,
        input_ids: torch.Tensor,
        prefix_ids: List[int],
        past: Any,
        start_token_id: int,
        start_token_prob: float,
        prefix_logprob: float,
        branch_index: int,
    ) -> Tuple[List[int], float]:
        # Continue greedy generation after forcing an alternative start token at the branch.
        device0 = input_ids.device

        gen_ids = list(prefix_ids)
        logprob = float(prefix_logprob)

        gen_ids.append(int(start_token_id))
        logprob += math.log(max(float(start_token_prob), 1e-45))

        if int(start_token_id) == self.tokenizer.eos_token_id:
            return gen_ids, logprob

        remaining = self.cfg.max_new_tokens - (branch_index + 1)
        last = int(start_token_id)

        for _ in range(max(0, remaining)):
            step_inp = torch.tensor([[last]], device=device0, dtype=input_ids.dtype)
            out = self.model(input_ids=step_inp, past_key_values=past, use_cache=True)

            past = out.past_key_values

            logits = out.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.argmax(probs, dim=-1).item())

            logprob += math.log(max(probs[0, nxt].item(), 1e-45))
            gen_ids.append(nxt)

            last = nxt
            if last == self.tokenizer.eos_token_id:
                break

        return gen_ids, logprob

    @torch.inference_mode()
    def generate_many(self, question: str) -> Tuple[List[str], List[float], str, float, int]:
        # Generate greedy answer + clustered-branch continuations; pick best by logprob.
        if self.is_gemma:
            user_prompt = "Give a short accurate answer (without additional text).\n\nQuestion: " + question
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Give a short accurate answer (without additional text)."},
                {"role": "user", "content": question},
            ]

        chat_text = build_chat_text(self.tokenizer, messages)
        input_ids = self.tokenizer(chat_text, return_tensors="pt").input_ids.to(self.model.device)

        gen0_ids, conf0, logprob0 = self.greedy_generate_with_cache(input_ids)
        response0 = self.tokenizer.decode(gen0_ids, skip_special_tokens=True).strip()

        proper_nouns_raw = self.extract_proper_nouns_from_sentence(text=response0)
        proper_nouns = safe_parse_list(proper_nouns_raw)
        proper_nouns = delete_proper_nouns_exist_in_the_question(proper_nouns=proper_nouns, question=question)

        gen0_token_strs = [self.tokenizer.decode([tid]) for tid in gen0_ids]
        pn_positions = find_proper_noun_token_ranges(generated_tokens=gen0_token_strs, proper_nouns=proper_nouns)

        branch_index = find_clustrable_token(proper_nouns_positions=pn_positions, tokens_confidence=conf0)
        if branch_index is None:
            branch_index = 0
        branch_index = int(max(0, min(branch_index, self.cfg.max_new_tokens - 1)))

        past, logits_at_branch, prefix_ids, prefix_logprob, context_ids = self.prepare_branch_state(input_ids, branch_index)
        greedy_token_id = int(torch.argmax(torch.softmax(logits_at_branch, dim=-1), dim=-1).item())

        cluster_info = self.compute_topk_clusters(
            context_ids=context_ids,
            logits=logits_at_branch,
            greedy_token_id=greedy_token_id,
        )

        answers = [response0]
        scores = [float(logprob0)]

        for _ in range(self.cfg.num_new_samples):
            token_id, token_prob = self.choose_cluster_and_token(cluster_info)

            gen_ids_k, logprob_k = self.continue_greedy_from_past(
                input_ids=input_ids,
                prefix_ids=prefix_ids,
                past=past,
                start_token_id=token_id,
                start_token_prob=token_prob,
                prefix_logprob=prefix_logprob,
                branch_index=branch_index,
            )

            response_k = self.tokenizer.decode(gen_ids_k, skip_special_tokens=True).strip()
            answers.append(response_k)
            scores.append(float(logprob_k))

        best_idx = int(np.argmax(np.array(scores, dtype=np.float64)))
        return answers, scores, answers[best_idx], float(scores[best_idx]), best_idx
