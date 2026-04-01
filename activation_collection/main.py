from pathlib import Path

import torch
import transformer_lens.utils as utils
from tqdm import tqdm
from transformer_lens import HookedTransformer

from common import (
    collapse_activations,
    load_csv_dataset,
    save_batch,
    set_seeds,
)
from model_instructions import build_prompt, label


def run(
    model_name,
    instruction,
    dataset_path,
    save_dir,
    token_mode="all",
    batch_size=100,
    seed=42,
    tag="run",
):
    assert token_mode in {"all", "last"}
    set_seeds(seed)

    dataset_path = Path(dataset_path)
    dataset = dataset_path.stem

    D = load_csv_dataset(dataset_path, seed)
    print(f"[data] loaded {len(D)} examples")

    model_tag = model_name.rsplit("/", 1)[-1]
    base = f"{dataset}_{instruction}_{model_tag}_{token_mode}_{tag}"
    out_true = Path(save_dir) / f"{base}_correct.pkl"
    out_false = Path(save_dir) / f"{base}_incorrect.pkl"

    device = utils.get_device()
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, trust_remote_code=True, dtype=torch.float16,
    )
    resid_hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]

    buf_true, buf_false = [], []
    c = i = 0

    for ex in tqdm(D, desc=f"Collecting[{instruction}|{dataset}]"):
        prompt, _ = build_prompt(instruction, ex)
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True, names_filter=resid_hooks)

        resid = {n: cache[n].detach().cpu().numpy() for n in resid_hooks}
        resid = collapse_activations(resid, token_mode)

        lab, aux = label(ex)

        rec = {
            "dataset": dataset,
            "instruction": instruction,
            "prompt": prompt,
            "label": bool(lab),
            "aux": aux,
            "resid_activations": resid,
            "model_tokens": tokens.detach().cpu().numpy(),
            "final_logits": logits[0, -1].detach().cpu().numpy(),
        }

        if lab:
            c += 1
            buf_true.append(rec)
            if len(buf_true) >= batch_size:
                save_batch(out_true, buf_true)
        else:
            i += 1
            buf_false.append(rec)
            if len(buf_false) >= batch_size:
                save_batch(out_false, buf_false)

        del tokens, logits, cache, resid
        torch.cuda.empty_cache()

    save_batch(out_true, buf_true)
    save_batch(out_false, buf_false)
    total = max(1, c + i)
    print(f"[done] {out_true}\n       {out_false}\nTrue-rate={c / total:.2%} (T:{c}, F:{i})")
