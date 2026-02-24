import argparse
import os

from _bootstrap import bootstrap
bootstrap()

from methods.verifier import build_verified_filename, verify_file


def main() -> None:
    # Run verifier to select the best answer among candidates using OpenAI.
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=str)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--model", type=str, default="gpt-5.2")
    ap.add_argument("--max_output_tokens", type=int, default=200)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    out_path = build_verified_filename(args.in_path, greedy=args.greedy)

    verify_file(
        in_path=args.in_path,
        out_path=out_path,
        openai_api_key=api_key,
        verifier_model=args.model,
        max_output_tokens=args.max_output_tokens,
        greedy=args.greedy,
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
