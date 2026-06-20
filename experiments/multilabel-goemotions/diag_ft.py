"""Phase-4 diagnostic: is the FT bert scorer broken, or only AutoIntent's decision threshold?

Trains BertScorer directly on a prepared dataset (visible loss), then evaluates its raw sigmoid outputs
on the test split at a sweep of fixed global thresholds. If macro-F1 at some fixed threshold is healthy,
the scorer learned and the degenerate pipeline result came from decision-layer threshold selection; if it
is ~0.1 at every threshold, the bert itself did not learn (recipe problem: epochs / lr / collapse).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from autointent.modules.scoring._bert import BertScorer

data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ge_classwise_100_s1_test.json")
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
# 4th arg: "es" keeps default early stopping; "noes" disables it (train full epochs, keep last checkpoint)
early_stop = (sys.argv[4] if len(sys.argv) > 4 else "es") == "es"
es_config = None if early_stop else {"metric": None}

data = json.loads(data_path.read_text(encoding="utf-8"))
Xtr = [r["utterance"] for r in data["train"]]
ytr = [r["label"] for r in data["train"]]
Xte = [r["utterance"] for r in data["test"]]
yte = np.array([r["label"] for r in data["test"]])

print(f"train={len(Xtr)} test={len(Xte)} classes={len(ytr[0])} epochs<= {epochs} lr={lr} early_stop={early_stop}", flush=True)
scorer = BertScorer(
    classification_model_config={"model_name": "bert-base-uncased", "device": "mps"},
    num_train_epochs=epochs,
    batch_size=32,
    learning_rate=lr,
    seed=1,
    early_stopping_config=es_config,
    print_progress=True,
)
scorer.fit(Xtr, ytr)
P = scorer.predict(Xte)
print(f"\npred sigmoid stats: min={P.min():.3f} max={P.max():.3f} mean={P.mean():.3f} std={P.std():.3f}", flush=True)
print(f"per-class mean spread: min={P.mean(0).min():.3f} max={P.mean(0).max():.3f}", flush=True)
print("macro-F1 at fixed global thresholds:")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (P > t).astype(int)
    macro = f1_score(yte, pred, average="macro", zero_division=0)
    micro = f1_score(yte, pred, average="micro", zero_division=0)
    print(f"  thr={t:.1f}  macro-F1={macro:.4f}  micro-F1={micro:.4f}  pred-positive-rate={(pred.mean()):.3f}")
