import argparse
import os

from _bootstrap import bootstrap
bootstrap()

from openai import OpenAI
from metrics.accuracy import evaluate_file


def iter_verified_files(results_dir: str):
    # Yield verified JSON paths recursively under results_dir.
    for root, _, files in os.walk(results_dir):
        for fn in files:
            fn_l = fn.lower()
            if not fn_l.endswith(".json"):
                continue
            if "verified" not in fn_l:
                continue
            if fn_l.endswith("_judge_accuracy.json") or fn_l.endswith("_diversity.json"):
                continue
            yield os.path.join(root, fn)


def main() -> None:
    # Compute accuracy for all verified files under results_dir.
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--model", type=str, default="gpt-5.2")
    ap.add_argument("--max_output_tokens", type=int, default=180)
    ap.add_argument("--allow_external_knowledge", type=int, default=1)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in your environment.")

    client = OpenAI(api_key=api_key)

    for in_path in sorted(iter_verified_files(args.results_dir)):
        out_path = in_path[:-5] + "_judge_accuracy.json"
        res = evaluate_file(
            client=client,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            allow_external_knowledge=bool(args.allow_external_knowledge),
            in_path=in_path,
            out_path=out_path,
        )
        if "error" in res:
            print(f"{os.path.basename(in_path)}: ERROR: {res['error']}")
        else:
            print(f"{os.path.basename(in_path)}: acc={res['accuracy']*100:.2f}%  N={res['n']}")

    print("Done.")


if __name__ == "__main__":
    main()
