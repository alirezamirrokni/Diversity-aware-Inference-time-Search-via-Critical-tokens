import os
import re
import json
import math
import hashlib
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from sacrebleu.metrics import BLEU

from utils import mild_clean


SELF_BLEU_N = 2
bleu = BLEU(effective_order=True)


def extract_answer_from_text(opt_text: str) -> str:
    # Extract an 'Answer:' field if present; otherwise return the text.
    t = (opt_text or "").strip()
    if not t:
        return ""
    m = re.search(r"(?is)\banswer\s*:\s*(.*?)(?:\bexplanation\s*:|$)", t)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r"\s+", " ", ans).strip(" \n\t,;")
        return ans
    return t


def extract_answer(candidate: Any) -> str:
    # Extract a normalized answer string from either a string or a dict-like candidate.
    if candidate is None:
        return ""
    if isinstance(candidate, str):
        return extract_answer_from_text(candidate)
    if isinstance(candidate, dict):
        for k in ["answer", "final_answer", "final", "prediction", "text", "output", "response", "generated", "content"]:
            if k in candidate and candidate[k] is not None:
                return extract_answer_from_text(str(candidate[k]))
        return extract_answer_from_text(json.dumps(candidate, ensure_ascii=False))
    return extract_answer_from_text(str(candidate))


def get_candidates(item: Dict[str, Any]) -> List[Any]:
    # Get a list of candidate strings from common keys.
    if isinstance(item.get("generated", None), list):
        return item["generated"]
    if isinstance(item.get("options", None), list):
        return item["options"]
    return []


def distinct_n(texts: List[str], n: int) -> float:
    # Compute Distinct-n for a list of texts (tokenized by whitespace).
    ngrams = []
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ngrams.append(tuple(toks[i : i + n]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def self_bleu_score(texts: List[str]) -> float:
    # Compute average sentence-level Self-BLEU over a set of candidates.
    if len(texts) <= 1:
        return float("nan")
    scores = []
    for i in range(len(texts)):
        hyp = texts[i]
        refs = [texts[j] for j in range(len(texts)) if j != i]
        if not refs:
            continue
        sc = bleu.sentence_score(hyp, refs).score / 100.0
        scores.append(sc)
    return sum(scores) / len(scores) if scores else float("nan")


def fmt_num(x: float, digits: int = 3) -> str:
    # Format possibly-NaN floats for reporting.
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{digits}f}"


def file_signature(path: str) -> Dict[str, Any]:
    # Basic file signature used for caching results.
    st = os.stat(path)
    return {"mtime": st.st_mtime, "size": st.st_size}


def config_hash(diversity_k: int) -> str:
    # Hash configuration to invalidate cached results when parameters change.
    cfg = {
        "SELF_BLEU_N": SELF_BLEU_N,
        "DIVERSITY_K": diversity_k,
        "cleaning": "mild_clean_v1",
        "metrics": ["distinct2", "self_bleu"],
    }
    s = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def load_cached(out_path: str, in_sig: Dict[str, Any], cfg_hash: str) -> Optional[Dict[str, Any]]:
    # Load cached diversity metrics if config and input signature match.
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return None
        meta = cached.get("meta", {})
        if not isinstance(meta, dict):
            return None
        if meta.get("config_hash") != cfg_hash:
            return None
        if meta.get("input_mtime") != in_sig["mtime"]:
            return None
        if meta.get("input_size") != in_sig["size"]:
            return None
        if "metrics" not in cached:
            return None
        return cached
    except Exception:
        return None


def evaluate_file(in_path: str, out_path: str, diversity_k: int = 5) -> Dict[str, Any]:
    # Compute Distinct-2 and Self-BLEU over first K candidates per item; write *_diversity.json.
    in_sig = file_signature(in_path)
    cfg_h = config_hash(diversity_k)
    cached = load_cached(out_path, in_sig, cfg_h)
    if cached is not None:
        m = cached.get("metrics", {})
        return {
            "distinct2": float(m.get("distinct2", float("nan"))),
            "self_bleu": float(m.get("self_bleu", float("nan"))),
            "cached": True,
        }

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {"error": "JSON is not a list"}

    n = 0
    k_used_sum = 0.0

    d2_sum = 0.0
    sb_sum = 0.0
    d2_cnt = 0
    sb_cnt = 0

    for item in tqdm(data, desc=os.path.basename(in_path), leave=False):
        if not isinstance(item, dict):
            continue
        n += 1

        candidates = get_candidates(item)

        cand_raw = [extract_answer(c) for c in candidates]
        cand = [mild_clean(t) for t in cand_raw]
        cand = [t for t in cand if t]

        k_used = min(diversity_k, len(cand))
        k_used_sum += float(k_used)

        k_texts = cand[:k_used]

        if k_used >= 2:
            d2_sum += distinct_n(k_texts, 2)
            sb_sum += self_bleu_score(k_texts)
            d2_cnt += 1
            sb_cnt += 1

    if n == 0:
        return {"error": "No valid items"}

    k_used_avg = k_used_sum / n

    raw_distinct2 = (d2_sum / d2_cnt) if d2_cnt > 0 else float("nan")
    raw_self_bleu = (sb_sum / sb_cnt) if sb_cnt > 0 else float("nan")

    if isinstance(raw_distinct2, float) and math.isnan(raw_distinct2):
        distinct2 = float("nan")
    else:
        distinct2 = raw_distinct2 * (k_used_avg / float(diversity_k)) if diversity_k > 0 else float("nan")

    if isinstance(raw_self_bleu, float) and math.isnan(raw_self_bleu):
        self_bleu = float("nan")
    else:
        self_bleu = raw_self_bleu * (float(diversity_k) / k_used_avg) if k_used_avg > 0 else float("nan")

    metrics = {"distinct2": distinct2, "self_bleu": self_bleu}

    out_json = {
        "file": os.path.basename(in_path),
        "meta": {"config_hash": cfg_h, "input_mtime": in_sig["mtime"], "input_size": in_sig["size"]},
        "metrics": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return {**metrics, "cached": False}
