import argparse

from _bootstrap import bootstrap
bootstrap()

from data import load_triviaqa_rc
from hf import detect_flags, load_tokenizer_and_model
from methods.ours import OursConfig, OursGenerator
from utils import ensure_pad_token, load_cache_list, make_results_path, save_list_sorted_by_id, set_seed, slugify


def main() -> None:
    # Run DISC "Ours" generation method on TriviaQA.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")

    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--cluster_k", type=int, default=5)
    ap.add_argument("--num_new_samples", type=int, default=4)
    ap.add_argument("--sample_cluster", type=int, default=1)
    ap.add_argument("--sample_within_cluster", type=int, default=0)
    ap.add_argument("--layer_idx", type=int, default=-1)
    ap.add_argument("--emb_batch", type=int, default=4)

    args = ap.parse_args()

    tokenizer, model = load_tokenizer_and_model(args.model)
    ensure_pad_token(tokenizer)

    is_gemma, _, _ = detect_flags(args.model)

    cfg = OursConfig(
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        cluster_k=args.cluster_k,
        num_new_samples=args.num_new_samples,
        sample_cluster=bool(args.sample_cluster),
        sample_within_cluster=bool(args.sample_within_cluster),
        layer_idx=None if args.layer_idx < 0 else int(args.layer_idx),
        emb_batch=args.emb_batch,
    )
    gen = OursGenerator(tokenizer=tokenizer, model=model, is_gemma=is_gemma, cfg=cfg)

    ds = load_triviaqa_rc("validation")

    model_slug = slugify(args.model.split("/")[-1])
    out_name = (
        f"Ours_TriviaQA_{model_slug}"
        f"_mnt{cfg.max_new_tokens}"
        f"_topk{cfg.top_k}"
        f"_ck{cfg.cluster_k}"
        f"_ns{cfg.num_new_samples}"
        f"_sc{int(bool(cfg.sample_cluster))}"
        f"_sw{int(bool(cfg.sample_within_cluster))}"
        f"_layer{cfg.layer_idx}"
        f"_eb{cfg.emb_batch}"
        f"_seed{args.seed}.json"
    )

    out_path = make_results_path(args.results_dir, args.model, out_name)
    results, done_ids = load_cache_list(out_path)

    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        if i in done_ids:
            continue

        set_seed(args.seed + i)

        question = ex["question"]
        correct = ex["answer"]

        answers, scores, best, best_score, best_idx = gen.generate_many(question)

        results.append(
            {
                "id": i,
                "question": question,
                "generated": answers,
                "scores": scores,
                "best": best,
                "best_score": best_score,
                "best_index": best_idx,
                "correct": correct,
            }
        )

        done_ids.add(i)
        save_list_sorted_by_id(out_path, results)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
