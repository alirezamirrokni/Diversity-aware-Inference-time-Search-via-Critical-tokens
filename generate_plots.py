import os
import re
import json
import math
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Global plotting style (bigger + bold everywhere)
# -----------------------------
sns.set_style("darkgrid")
sns.set_palette("bright")
sns.set_context("talk", font_scale=1.1)  # slightly larger overall

# A more "paper-like" font; usually available with matplotlib
FONT_FAMILY = "STIXGeneral"

# Bump sizes
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 16

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "axes.linewidth": 1.3,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,  # better embedding in PDFs
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def _bold_ticks(ax: plt.Axes) -> None:
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")


def _style_legend_outside(ax: plt.Axes, ncol: int = 1) -> None:
    """
    Place legend outside to the right (prevents overlap with bars/lines).
    """
    leg = ax.legend(
        title="",
        frameon=False,
        ncol=ncol,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        prop={"size": LEGEND_SIZE, "weight": "bold"},
    )
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontweight("bold")


RESULTS_ROOT = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Tokenization + diversity metrics
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(cands: List[str], n: int = 2) -> float:
    """Distinct-n over a set of candidates: |union ngrams| / total ngrams."""
    union = set()
    total = 0
    for y in cands:
        toks = _tokenize(y)
        ng = _ngrams(toks, n)
        total += len(ng)
        union.update(ng)
    return float(len(union) / total) if total > 0 else 0.0


def _bleu2(candidate: List[str], references: List[List[str]], eps: float = 1e-9) -> float:
    """Minimal BLEU-2 with BP and smoothing; weights (0.5, 0.5)."""
    if not candidate:
        return 0.0

    cand_len = len(candidate)
    ref_lens = [len(r) for r in references if r is not None]
    if not ref_lens:
        return 0.0
    ref_len = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))

    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / max(cand_len, 1)))

    precisions = []
    for n in (1, 2):
        cand_ngrams = _ngrams(candidate, n)
        if not cand_ngrams:
            precisions.append(eps)
            continue

        cand_counts = Counter(cand_ngrams)

        max_ref_counts: Counter = Counter()
        for ref in references:
            ref_counts = Counter(_ngrams(ref, n))
            for ng, ct in ref_counts.items():
                if ct > max_ref_counts.get(ng, 0):
                    max_ref_counts[ng] = ct

        clipped = 0
        total = 0
        for ng, ct in cand_counts.items():
            total += ct
            clipped += min(ct, max_ref_counts.get(ng, 0))

        p_n = (clipped / total) if total > 0 else 0.0
        precisions.append(max(p_n, eps))

    return float(bp * math.exp(0.5 * (math.log(precisions[0]) + math.log(precisions[1]))))


def self_bleu_2(cands: List[str]) -> float:
    """SelfBLEU-2 over a set of candidates: average BLEU-2 of each vs the rest."""
    if len(cands) < 2:
        return float("nan")
    tok = [_tokenize(y) for y in cands]
    scores = []
    for i in range(len(tok)):
        refs = [tok[j] for j in range(len(tok)) if j != i]
        scores.append(_bleu2(tok[i], refs))
    return float(np.mean(scores))


def extract_candidates(item: Dict[str, Any]) -> List[str]:
    """
    Robust candidate extraction across your schemas:
      - item["options"] (baselines)
      - item["generated"] (some DISC runs)
      - item["candidates"]
      - fallback: ["final_answer"] or ["best"]
    """
    for k in ("options", "generated", "candidates"):
        if k in item and isinstance(item[k], list):
            c = [str(x) for x in item[k] if isinstance(x, str) and x.strip()]
            if c:
                return c
    for k in ("final_answer", "best"):
        if k in item and isinstance(item[k], str) and item[k].strip():
            return [item[k].strip()]
    return []


def compute_diversity_from_json(path: str, k: int = 5) -> Dict[str, float]:
    """
    Compute mean Distinct-2@k and SelfBLEU-2@k across examples in a JSON file,
    using up to the first k candidates per example.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {"distinct2": float("nan"), "selfbleu2": float("nan")}

    d_vals = []
    s_vals = []
    for ex in data:
        if not isinstance(ex, dict):
            continue
        cands = extract_candidates(ex)[:k]
        if not cands:
            continue
        d_vals.append(distinct_n(cands, n=2))
        s_vals.append(self_bleu_2(cands))

    distinct2 = float(np.mean(d_vals)) if d_vals else float("nan")
    s_clean = [v for v in s_vals if not (isinstance(v, float) and math.isnan(v))]
    selfbleu2 = float(np.mean(s_clean)) if s_clean else float("nan")
    return {"distinct2": distinct2, "selfbleu2": selfbleu2}


# -----------------------------
# File helpers
# -----------------------------
def list_jsons(model_dir: str) -> List[str]:
    return sorted(
        [
            os.path.join(model_dir, fn)
            for fn in os.listdir(model_dir)
            if fn.endswith(".json")
        ]
    )


def pick_first_matching(paths: List[str], include_patterns: List[str]) -> Optional[str]:
    """
    Pick first file whose basename matches ALL regex substrings in include_patterns.
    """
    for p in paths:
        base = os.path.basename(p)
        ok = True
        for pat in include_patterns:
            if re.search(pat, base) is None:
                ok = False
                break
        if ok:
            return p
    return None


def detect_model_key(folder_name: str) -> str:
    return folder_name.lower().strip()


MODEL_DISPLAY = {
    "qwen": "Qwen2.5-7B-Instruct",
    "gemma": "Gemma-7B-It",
    "mistral": "Mistral-7B-Instruct-v0.3",
    "phi-3": "Phi-3-mini-4k-instruct",
}
MODEL_ORDER = ["qwen", "gemma", "mistral", "phi-3"]


# -----------------------------
# Discover model folders
# -----------------------------
if not os.path.isdir(RESULTS_ROOT):
    raise FileNotFoundError(f"Cannot find '{RESULTS_ROOT}/' directory.")

model_dirs = [
    os.path.join(RESULTS_ROOT, d)
    for d in os.listdir(RESULTS_ROOT)
    if os.path.isdir(os.path.join(RESULTS_ROOT, d))
]
model_keys = [detect_model_key(os.path.basename(d)) for d in model_dirs]
model_map = dict(zip(model_keys, model_dirs))


# ============================================================
# Part A) Across-model line plots (DISC / BoN / Beam)
# ============================================================
MAIN_METHOD_FILES: Dict[Tuple[str, str], str] = {}

for mk, mdir in model_map.items():
    paths = list_jsons(mdir)

    disc = pick_first_matching(paths, [r"^Ours_", r"_sc1_sw0_"]) or pick_first_matching(paths, [r"^Ours_"])
    bon = pick_first_matching(paths, [r"^BoN_"])
    beam = pick_first_matching(paths, [r"^BeamSearch_"])

    if disc:
        MAIN_METHOD_FILES[(mk, "DISC")] = disc
    if bon:
        MAIN_METHOD_FILES[(mk, "BoN")] = bon
    if beam:
        MAIN_METHOD_FILES[(mk, "Beam")] = beam

main_rows = []
for (mk, method), path in MAIN_METHOD_FILES.items():
    met = compute_diversity_from_json(path, k=5)
    main_rows.append(
        {
            "model_key": mk,
            "model": MODEL_DISPLAY.get(mk, mk),
            "method": method,
            "Distinct-2": met["distinct2"],
            "SelfBLEU-2": met["selfbleu2"],
        }
    )
df_main = pd.DataFrame(main_rows)

df_main["model_key"] = pd.Categorical(
    df_main["model_key"],
    categories=[m for m in MODEL_ORDER if m in model_map],
    ordered=True,
)
df_main = df_main.sort_values(["model_key", "method"])


def lineplot_across_models(metric_col: str, out_pdf: str) -> None:
    plt.figure(figsize=(10.8, 3.6))
    ax = sns.lineplot(
        data=df_main,
        x="model",
        y=metric_col,
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        linewidth=2.3,
        marker="o",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.3)

    _bold_ticks(ax)
    _style_legend_outside(ax, ncol=1)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


lineplot_across_models("Distinct-2", os.path.join(PLOTS_DIR, "distinct2_across_models.pdf"))
lineplot_across_models("SelfBLEU-2", os.path.join(PLOTS_DIR, "selfbleu2_across_models.pdf"))


# ============================================================
# Part B) Ablation BAR plots (no spider/radar)
# ============================================================

ABLATION_VARIANTS = [
    ("DISC",              [r"^Ours_", r"_sc1_sw0_"]),
    ("sc=1, sw=0",        [r"_sc1_sw0_"]),
    ("sc=0, sw=0",        [r"_sc0_sw0_"]),
    ("sc=1, sw=1",        [r"_sc1_sw1_"]),
]

ABLATION_MODELS = [mk for mk in ["qwen", "gemma"] if mk in model_map]

ab_rows = []
for mk in ABLATION_MODELS:
    paths = list_jsons(model_map[mk])

    for label, pats in ABLATION_VARIANTS:
        p = pick_first_matching(paths, pats)
        if p is None:
            ab_rows.append(
                {
                    "model_key": mk,
                    "model": MODEL_DISPLAY.get(mk, mk),
                    "variant": label,
                    "Distinct-2": float("nan"),
                    "SelfBLEU-2": float("nan"),
                }
            )
            continue

        met = compute_diversity_from_json(p, k=5)
        ab_rows.append(
            {
                "model_key": mk,
                "model": MODEL_DISPLAY.get(mk, mk),
                "variant": label,
                "Distinct-2": met["distinct2"],
                "SelfBLEU-2": met["selfbleu2"],
            }
        )

df_ab = pd.DataFrame(ab_rows)
df_ab["model_key"] = pd.Categorical(df_ab["model_key"], categories=ABLATION_MODELS, ordered=True)
df_ab["variant"] = pd.Categorical(
    df_ab["variant"],
    categories=[v[0] for v in ABLATION_VARIANTS],
    ordered=True,
)
df_ab = df_ab.sort_values(["model_key", "variant"])


def barplot_ablation(metric_col: str, out_pdf: str) -> None:
    plt.figure(figsize=(10.8, 3.8))
    ax = sns.barplot(
        data=df_ab,
        x="model",
        y=metric_col,
        hue="variant",
        edgecolor="none",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.3)

    _bold_ticks(ax)
    _style_legend_outside(ax, ncol=1)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


barplot_ablation("Distinct-2", os.path.join(PLOTS_DIR, "ablation_distinct2_bar.pdf"))
barplot_ablation("SelfBLEU-2", os.path.join(PLOTS_DIR, "ablation_selfbleu2_bar.pdf"))

print(f"Saved PDFs to: {PLOTS_DIR}/")
print("Main plots:",
      "distinct2_across_models.pdf, selfbleu2_across_models.pdf")
print("Ablation bar plots:",
      "ablation_distinct2_bar.pdf, ablation_selfbleu2_bar.pdf")