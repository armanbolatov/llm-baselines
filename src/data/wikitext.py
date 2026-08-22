import os

import numpy as np
import tiktoken
from datasets import load_dataset


def get_wikitext_data(datasets_base_dir):
    """Prepare WikiText-103 raw dataset as train/val token bins."""
    WIKITEXT_DATA_PATH = os.path.join(datasets_base_dir, "wikitext/")
    train_bin = os.path.join(WIKITEXT_DATA_PATH, "train.bin")
    val_bin = os.path.join(WIKITEXT_DATA_PATH, "val.bin")

    if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("downloading WikiText-103 and tokenizing")

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        raw_train_data = "\n".join(ds["train"]["text"])
        raw_eval_data = "\n".join(ds["validation"]["text"])

        tokenizer = tiktoken.get_encoding("gpt2")
        train_tokenized = np.array(
            tokenizer.encode_ordinary(raw_train_data), dtype=np.uint16
        )
        eval_tokenized = np.array(
            tokenizer.encode_ordinary(raw_eval_data), dtype=np.uint16
        )

        train_tokenized.tofile(train_bin)
        eval_tokenized.tofile(val_bin)
        print("completed the tokenization process")

    return {
        "train": train_bin,
        "val": val_bin,
    }
