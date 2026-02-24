import os
from huggingface_hub import login


def main() -> None:
    # Login to HuggingFace using HF_TOKEN environment variable.
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is empty. Set HF_TOKEN env var if you need gated/private models.")
        return
    login(token=token)
    print("HuggingFace login done.")


if __name__ == "__main__":
    main()
