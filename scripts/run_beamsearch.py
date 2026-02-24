import argparse

from _bootstrap import bootstrap
bootstrap()

from data import load_triviaqa_rc
from hf import detect_flags, load_tokenizer_and_model
from methods.beamsearch import beam_search_record
from utils import ensure_pad_token, load_cache_list, make_results_path, save_list_sorted_by_id, set_seed, slugify, stringify


def main() -> None:
    # Run BeamSearch baseline on TriviaQA (outputs verified-format JSON).
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")

    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.0)

    args = ap.parse_args()

    tokenizer, model = load_tokenizer_and_model(args.model)
    ensure_pad_token(tokenizer)

    is_gemma, _, _ = detect_flags(args.model)

    ds = load_triviaqa_rc("validation")

    model_slug = slugify(args.model.split("/")[-1])
    out_name = (
        f"BeamSearch_TriviaQA_{model_slug}"
        f"_mnt{args.max_new_tokens}"
        f"_beams{args.num_beams}"
        f"_lp{args.length_penalty}"
        f"_seed{args.seed}verified.json"
    )
    out_path = make_results_path(args.results_dir, args.model, out_name)

    results, done_ids = load_cache_list(out_path)

    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        if i in done_ids:
            continue

        set_seed(args.seed + i)

        question = stringify(ex.get("question", "")).strip()
        correct = ex.get("answer", None)

        rec = beam_search_record(
            tokenizer=tokenizer,
            model=model,
            is_gemma=is_gemma,
            question=question,
            correct=correct,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            qid=i,
        )

        results.append(rec)
        done_ids.add(i)
        save_list_sorted_by_id(out_path, results)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
