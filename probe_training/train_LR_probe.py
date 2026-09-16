import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

LAYER_RE = re.compile(r"blocks\.(\d+)\.hook_resid_post")


# ── Probe ──

class LRProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=False)
        self.register_buffer("mean", torch.zeros(d_in))

    def score(self, x):
        return self.linear(x - self.mean).squeeze(-1)

    def forward(self, x):
        return torch.sigmoid(self.score(x))

    def direction(self):
        return self.linear.weight.data[0]

    @staticmethod
    def train_probe(X, y, lr=1e-3, weight_decay=0.1, epochs=1000):
        probe = LRProbe(X.shape[1])
        probe.mean.copy_(X.mean(0))

        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()

        # Pass X directly; forward() centers by itself
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(probe(X), y)
            loss.backward()
            optimizer.step()

        return probe


# ── Data loading ──

def load_pickle_stream(path):
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.extend(pickle.load(f))
            except EOFError:
                break
    return records


def load_activations(correct_path, incorrect_path):
    records = load_pickle_stream(correct_path) + load_pickle_stream(incorrect_path)
    layer_keys = sorted(records[0]["resid_activations"].keys(),
                        key=lambda k: int(LAYER_RE.search(k).group(1)))

    X_by_layer = {}
    for lk in layer_keys:
        vecs = []
        for r in records:
            a = np.asarray(r["resid_activations"][lk])
            vecs.append(a[-1] if a.ndim > 1 else a)
        X_by_layer[lk] = np.stack(vecs)

    y = np.array([int(bool(r["label"])) for r in records])
    return X_by_layer, y


def balance(X_by_layer, y, seed=42):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    rng.shuffle(idx)
    return {lk: X[idx] for lk, X in X_by_layer.items()}, y[idx]


# ── Training ──

def train_per_layer(X_by_layer, y, test_size=0.3, seed=42):
    tr_idx, te_idx = train_test_split(np.arange(len(y)),
                                      test_size=test_size, stratify=y,
                                      random_state=seed)
    directions, means, rows = {}, {}, []

    for lk, X in X_by_layer.items():
        Xtr = torch.tensor(X[tr_idx]).float()
        ytr = torch.tensor(y[tr_idx]).float()
        Xte = torch.tensor(X[te_idx]).float()
        yte = y[te_idx]

        probe = LRProbe.train_probe(Xtr, ytr)
        w = probe.direction().detach().numpy()
        directions[lk] = w
        means[lk] = probe.mean.detach().numpy()

        scores = probe.score(Xte).detach().numpy()
        acc = accuracy_score(yte, (scores > 0).astype(int))
        auroc = roc_auc_score(yte, scores)
        layer_idx = int(LAYER_RE.search(lk).group(1))
        rows.append({"layer": lk, "layer_idx": layer_idx,
                      "acc": acc, "auroc": auroc,
                      "n_train": len(tr_idx), "n_test": len(te_idx)})
        print(f"  layer {layer_idx:2d}  acc={acc:.4f}  auroc={auroc:.4f}")

    metrics = pd.DataFrame(rows).sort_values("layer_idx").reset_index(drop=True)
    return directions, means, metrics


# ── Save ──

def save_results(out_dir, run_tag, directions, means, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out_dir / f"{run_tag}_metrics.csv", index=False)

    wdir = out_dir / f"{run_tag}_weights"
    wdir.mkdir(parents=True, exist_ok=True)
    for lk, w in directions.items():
        idx = int(LAYER_RE.search(lk).group(1))
        np.save(wdir / f"layer_{idx:02d}.npy", w)

    with open(out_dir / f"{run_tag}_probes.pkl", "wb") as f:
        pickle.dump({"directions": directions, "means": means, "metrics": metrics}, f)

    print(f"[saved] {out_dir / run_tag}_*")


# ── CLI ──

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct_path", required=True)
    ap.add_argument("--incorrect_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", action="store_true")
    args = ap.parse_args()

    X_by_layer, y = load_activations(args.correct_path, args.incorrect_path)
    print(f"[loaded] {len(y)} examples  pos={int((y==1).sum())}  neg={int((y==0).sum())}")

    if args.balance:
        X_by_layer, y = balance(X_by_layer, y, seed=args.seed)
        print(f"[balanced] {len(y)} examples")

    directions, means, metrics = train_per_layer(X_by_layer, y,
                                                  test_size=args.test_size,
                                                  seed=args.seed)
    save_results(Path(args.out_dir), args.run_tag, directions, means, metrics)
