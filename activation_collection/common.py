import pickle
import random

import numpy as np
import pandas as pd
import torch


def load_csv_dataset(path, seed):
    df = pd.read_csv(path).reset_index(drop=True).sample(frac=1, random_state=seed)
    examples = []
    for _, row in df.iterrows():
        ex = dict(row)
        ex["__text__"] = str(row["statement"])
        ex["is_correct"] = bool(row["label"])
        examples.append(ex)
    return examples


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_batch(filename, buffer):
    if not buffer:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "ab") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    buffer.clear()


def collapse_activations(acts, token_mode):
    if token_mode == "all":
        return acts
    if token_mode == "last":
        return {k: v[-1].copy() for k, v in acts.items()}
    raise ValueError("token_mode must be 'all' or 'last'")
