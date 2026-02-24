# Decision-Aware Sparse Diversification for Inference-Time Search

---
## Abstract 

> Inference-time scaling for large language models typically relies on generating multiple candidate completions and then selecting one. In practice, however, common strategies such as repeated sampling and beam search often spend much of their budget on near-duplicate candidates, limiting the amount of new information gained per unit compute. We introduce DISC, a lightweight, decision-aware search procedure that promotes diversity only at a small number of high-impact generation steps. DISC first produces a greedy draft, identifies a low-confidence decision point along the draft, and branches by drawing an alternative from a probability-weighted cluster of top candidates in token-embedding space, followed by greedy continuation. A verifier model then selects the final output from the candidate set. Across four instruction-tuned models on the first 100 TriviaQA validation questions, DISC consistently improves accuracy over greedy decoding, best-of-$N$ sampling, and beam search under comparable budgets. These results indicate that sparse, uncertainty-guided diversification can improve inference-time search without modifying model weights.

---

## Main figure 


![DISC main figure](assets/main_figure.png)


---

## Project structure

```
DISC/
  scripts/               # runnable entry points (generation / verify / metrics)
  src/                   # core implementation (HF wrapper, DISC, metrics utils)
  results/               # generated outputs (auto-created)
  plots/                 # generated plots (auto-created)
  requirements.txt
  generate_plots.py
  .env.example
```

---

## Requirements

- Python **>= 3.10**
- GPU recommended (HF causal LMs), CPU works but is slow
- Internet access for:
  - Hugging Face model downloads (and optional gated model access)
  - dataset download: `mandarjoshi/trivia_qa` (config: `rc`)
  - OpenAI API calls (verifier + judge accuracy)

---

## Setup

### 1) Create an environment & install dependencies

#### Linux / macOS (bash/zsh)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### 2) (Optional) Plotting dependencies

`generate_plots.py` uses pandas + seaborn + matplotlib.

```bash
pip install -U pandas seaborn matplotlib
```

---

## Environment variables (API keys)

The code expects keys via environment variables (it does **not** auto-load `.env` unless you add a dotenv loader).

- `OPENAI_API_KEY` (**required**) for:
  - `scripts/run_verify.py`
  - `scripts/run_accuracy.py`
- `HF_TOKEN` (**optional**) only if you use gated/private HF models

### Linux / macOS
```bash
export HF_TOKEN="hf_..."          # optional
export OPENAI_API_KEY="sk-..."    # required for verify/accuracy
```

### Windows PowerShell
```powershell
$env:HF_TOKEN="hf_..."            # optional
$env:OPENAI_API_KEY="sk-..."      # required for verify/accuracy
```

---

## Hugging Face login

If you need gated/private models:

```bash
python scripts/hf_login.py
```

This reads `HF_TOKEN` and logs in via `huggingface_hub`.

---

## How outputs are stored

Runs write JSON files under:

```
results/<model_family>/
```

The “model family” is derived from the HF model id (see `src/utils.py:model_family()`), e.g.
- `microsoft/Phi-3-mini-4k-instruct` → `results/phi/`
- other models have their own family keys

All generation scripts **resume** if a results file exists: they skip IDs already processed.

---

## Quickstart (end-to-end smoke test)

This runs DISC on 3 TriviaQA examples, verifies with OpenAI, then computes diversity and judge accuracy.

### Linux / macOS
```bash
python scripts/run_ours.py --n 3
python scripts/run_verify.py --in results/phi/Ours_TriviaQA_*.json
python scripts/run_diversity.py --results_dir results
python scripts/run_accuracy.py --results_dir results
```

### Windows (PowerShell)
```powershell
python scripts\run_ours.py --n 3
python scripts\run_verify.py --in results\phi\Ours_TriviaQA_*.json
python scripts\run_diversity.py --results_dir results
python scripts\run_accuracy.py --results_dir results
```

---

## Running experiments

All generation scripts load:
- dataset: `mandarjoshi/trivia_qa` with config `rc`
- default split: validation
- default `--n 100` (first 100 validation questions)

To see all options for any script:
```bash
python scripts/run_ours.py -h
python scripts/run_bon.py -h
python scripts/run_beamsearch.py -h
python scripts/run_verify.py -h
python scripts/run_diversity.py -h
python scripts/run_accuracy.py -h
```

### 1) Beam Search baseline

Beam search output is already “verified-format” (options + chosen index), so you generally do **not** need `run_verify.py`.

```bash
python scripts/run_beamsearch.py \
  --model "microsoft/Phi-3-mini-4k-instruct" \
  --n 100 \
  --max_new_tokens 100 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --seed 42
```

### 2) Best-of-N sampling (BoN) baseline

BoN output is also “verified-format”.

```bash
python scripts/run_bon.py \
  --model "microsoft/Phi-3-mini-4k-instruct" \
  --n 100 \
  --max_new_tokens 100 \
  --num_samples 5 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --seed 42
```

### 3) DISC (Ours)

DISC produces a greedy draft, diversifies at a **single** low-confidence decision, then continues greedily for each branch.

```bash
python scripts/run_ours.py \
  --model "microsoft/Phi-3-mini-4k-instruct" \
  --n 100 \
  --max_new_tokens 100 \
  --top_k 100 \
  --cluster_k 5 \
  --num_new_samples 4 \
  --sample_cluster 1 \
  --sample_within_cluster 0 \
  --layer_idx -1 \
  --emb_batch 4 \
  --seed 42
```

**Flag meanings:**
- `--top_k`: how many next-token candidates to consider at the branching point
- `--cluster_k`: number of clusters in token-embedding space
- `--num_new_samples`: how many alternative candidates to branch into (in addition to the greedy draft)
- `--layer_idx`: which token-embedding layer is used for clustering (`-1` = last)
- `--emb_batch`: batching for embedding extraction (reduce if you OOM)

---

## Verification

DISC outputs are *not* automatically in verified-format, so to evaluate them with downstream scripts you typically run verification.

```bash
python scripts/run_verify.py \
  --in results/<model_family>/Ours_TriviaQA_....json \
  --model gpt-5.2 \
  --max_output_tokens 200
```

This produces a sibling file ending in `...verified.json`, containing (per example):
- `options`: candidate answers
- `final_answer`: verifier-selected text
- `chosen_option_num`: 1-indexed chosen option

---

## Diversity metrics

Computes Distinct-2 and Self-BLEU-2 on all `*verified*.json` files under `results/`.

```bash
python scripts/run_diversity.py --results_dir results --k 5
```

Outputs `*_diversity.json` next to each verified file.

---

## Judge accuracy

Uses an LLM judge to compare your final answer to the TriviaQA ground truth.

```bash
python scripts/run_accuracy.py \
  --results_dir results \
  --model gpt-5.2 \
  --max_output_tokens 180 \
  --allow_external_knowledge 1
```

Outputs `*_judge_accuracy.json` next to each verified file.

---

## Plotting

After running experiments and metrics:

```bash
python generate_plots.py
```

---

## Reproducibility notes

- Use `--seed` for deterministic shuffling / sampling where applicable.
- Exact reproducibility can still vary across GPUs / CUDA versions / PyTorch versions.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{disc2026,
  title     = {DISC: Decision-Aware Sparse Diversification for Inference-Time Search},
  author    = {Alireza Mirrokni},
  year      = {2026}
}
```
