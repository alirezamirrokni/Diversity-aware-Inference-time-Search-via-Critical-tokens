from datasets import load_dataset


def load_triviaqa_rc(split: str = "validation"):
    # Load TriviaQA (rc) split.
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    return ds[split]
