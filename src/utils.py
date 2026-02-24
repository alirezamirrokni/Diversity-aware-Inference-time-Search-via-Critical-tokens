import os
import re
import json
import ast
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def slugify(s: str) -> str:
    # Convert a string into a filesystem-safe slug.
    s = s.replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def stringify(x: Any) -> str:
    # Safely convert any object to a string.
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def safe_parse_list(s: str) -> List[str]:
    # Safely parse a Python list literal; return [] on failure.
    try:
        x = ast.literal_eval(s)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def set_seed(seed: int) -> None:
    # Set Python/NumPy/Torch seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def ensure_pad_token(tokenizer) -> None:
    # Ensure tokenizer has a pad_token_id to avoid generate() warnings/errors.
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def model_family(model_name: str) -> str:
    # Map HF model name to a stable family folder name.
    s = (model_name or "").lower()

    if "gemma" in s:
        return "gemma"
    if "qwen" in s:
        return "qwen"
    if "mistral" in s or "mixtral" in s:
        return "mistral"
    if "phi-3" in s or "phi3" in s or "/phi" in s:
        return "phi"
    if "zephyr" in s:
        return "zephyr"
    if "falcon" in s:
        return "falcon"
    if "olmo" in s:
        return "olmo"

    return slugify(model_name.split("/")[-1] if "/" in model_name else model_name)


def make_results_path(results_dir: str, model_name: str, filename: str) -> str:
    # Build a results path under results/<family>/filename and create the folder.
    fam = model_family(model_name)
    out_dir = os.path.join(results_dir, fam)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def load_cache_list(path: str) -> Tuple[List[Dict[str, Any]], set]:
    # Load a list-of-dicts JSON file as a cache and return (cleaned_list, done_ids).
    if not os.path.exists(path):
        return [], set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return [], set()

        done = set()
        cleaned: List[Dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue
            try:
                qid = int(item["id"])
            except Exception:
                continue
            if qid in done:
                continue
            done.add(qid)
            cleaned.append(item)

        return cleaned, done

    except Exception:
        return [], set()


def save_list_sorted_by_id(path: str, rows: List[Dict[str, Any]]) -> None:
    # Sort list by integer id and write it to disk.
    rows.sort(key=lambda x: int(x.get("id", 10**18)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def extract_tag(text: str, tag: str) -> str:
    # Extract content inside <tag>...</tag>.
    if not isinstance(text, str):
        return ""
    m = re.search(
        rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def mild_clean(s: str) -> str:
    # Lowercase and normalize whitespace.
    if s is None:
        return ""
    s = str(s).lower()
    return " ".join(s.split())
