import argparse
import os

from _bootstrap import bootstrap
bootstrap()

from metrics.diversity import evaluate_file, fmt_num


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
    # Compute diversity for all verified files under results_dir.
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    for in_path in sorted(iter_verified_files(args.results_dir)):
        out_path = in_path[:-5] + "_diversity.json"
        res = evaluate_file(in_path, out_path, diversity_k=args.k)
        if "error" in res:
            print(f"{os.path.basename(in_path)}: ERROR: {res['error']}")
        else:
            tag = " (cached)" if res.get("cached") else ""
            print(
                f"{os.path.basename(in_path)}: "
                f"Distinct-2@{args.k}={fmt_num(res['distinct2'])}  "
                f"SelfBLEU-2@{args.k}={fmt_num(res['self_bleu'])}{tag}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
