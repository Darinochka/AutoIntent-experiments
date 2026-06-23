"""Phase-4 best-shot FT diagnostic: bert-base with LR warmup + weight decay (absent from the preset).

The preset's BertScorer builds TrainingArguments with no warmup, which is the most likely specific cause
of the representation collapse seen at lr in {1e-5, 3e-5} (BCE plateaus at the base-rate floor ~0.17,
predictions near-constant across classes). This subclass adds warmup_ratio + weight_decay and trains full
epochs, then evaluates raw sigmoid separation at fixed thresholds. If this still collapses, the failure is
fundamental to bert-base FT on this sparse 28-class task at ~2k rows, not a missing-warmup artifact.

Usage:  uv run diag_ft_warmup.py <data.json> <epochs> <lr> <warmup_ratio>
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import json
import numpy as np
from sklearn.metrics import f1_score

from autointent.modules.scoring._bert import BertScorer


class WarmupBertScorer(BertScorer):
    """BertScorer whose Trainer uses LR warmup + weight decay (otherwise identical)."""

    warmup_ratio = 0.1
    weight_decay = 0.01

    def _train(self, tokenized_dataset) -> None:  # type: ignore[no-untyped-def]
        from transformers import DataCollatorWithPadding, PrinterCallback, ProgressCallback, Trainer, TrainingArguments

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                warmup_ratio=self.warmup_ratio,
                weight_decay=self.weight_decay,
                seed=self.seed,
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch",
                logging_strategy="steps",
                logging_steps=10,
                report_to=self.report_to,
                fp16=self.classification_model_config.fp16,
                bf16=self.classification_model_config.bf16,
                use_cpu=self.classification_model_config.device == "cpu",
                metric_for_best_model=self.early_stopping_config.metric,
                load_best_model_at_end=self.early_stopping_config.metric is not None,
            )
            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                processing_class=self._tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
                compute_metrics=self._get_compute_metrics(),
                callbacks=self._get_trainer_callbacks(),
            )
            if not self.print_progress:
                trainer.remove_callback(PrinterCallback)
                trainer.remove_callback(ProgressCallback)
            trainer.train()


data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ge_classwise_100_s1_test.json")
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
warmup = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

data = json.loads(data_path.read_text(encoding="utf-8"))
Xtr = [r["utterance"] for r in data["train"]]
ytr = [r["label"] for r in data["train"]]
Xte = [r["utterance"] for r in data["test"]]
yte = np.array([r["label"] for r in data["test"]])

print(f"WARMUP run: train={len(Xtr)} epochs={epochs} lr={lr} warmup_ratio={warmup}", flush=True)
scorer = WarmupBertScorer(
    classification_model_config={"model_name": "bert-base-uncased", "device": "mps"},
    num_train_epochs=epochs,
    batch_size=32,
    learning_rate=lr,
    seed=1,
    early_stopping_config={"metric": None},  # train full epochs; the scoring_f1@0.5 stop is broken here
    print_progress=True,
)
scorer.warmup_ratio = warmup
scorer.fit(Xtr, ytr)
P = scorer.predict(Xte)
print(f"\npred sigmoid stats: min={P.min():.3f} max={P.max():.3f} mean={P.mean():.3f} std={P.std():.3f}", flush=True)
print(f"per-class mean spread: min={P.mean(0).min():.3f} max={P.mean(0).max():.3f}", flush=True)
print("macro-F1 at fixed global thresholds:")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (P > t).astype(int)
    print(
        f"  thr={t:.1f}  macro-F1={f1_score(yte, pred, average='macro', zero_division=0):.4f}  "
        f"micro-F1={f1_score(yte, pred, average='micro', zero_division=0):.4f}  pos-rate={pred.mean():.3f}"
    )
