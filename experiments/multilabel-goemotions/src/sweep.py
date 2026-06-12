"""Grid sweep to find the best scoring/decision target metric across dataset sizes and balance modes.

For every (size, balance) cell a dataset is prepared once, then every (scoring_metric, decision_metric)
pair is optimized on it. Final eval metrics (on the fixed GoEmotions-validation eval set) are collected
into an incremental summary CSV so a long sweep can be interrupted and resumed.

Balance modes:
- classwise:  cap N samples/class -> balanced N-shot. Always feasible.
- stratified: floor of N samples/class -> imbalanced, but collapses to the full train once N exceeds the
              rarest class's total (~77 for GoEmotions' "grief"), so 100- and 500-shot become identical
              (the sweep dedups identical datasets so the redundant cell isn't re-run).
- natural:    proportion-preserving sample sized to match classwise(N) -> size-matched imbalanced. May be
              INFEASIBLE at small N: rare classes can drop out, and AutoIntent requires every class in every
              split (including the validation it carves from train). Such cells are skipped with a warning.

Every fit is wrapped in try/except: a cell that AutoIntent can't handle is recorded as failed and the
sweep continues.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from autointent.metrics import DICISION_METRICS_MULTILABEL, SCORING_METRICS_MULTILABEL

from src.data import (
    DEFAULT_CONFIG,
    DEFAULT_REPO,
    assemble_mapping,
    classwise_subsample,
    label_matrix,
    load_goemotions,
    natural_subsample,
    save_mapping,
    stratified_subsample,
)
from src.naming import metrics_path
from src.pipeline import run_experiment

ALL_SCORING = sorted(SCORING_METRICS_MULTILABEL)
ALL_DECISION = sorted(DICISION_METRICS_MULTILABEL)

CSV_FIELDS = [
    "size",
    "balance",
    "train_size",
    "scoring_metric",
    "decision_metric",
    "eval_f1",
    "eval_precision",
    "eval_recall",
    "eval_accuracy",
    "scorer",
    "decisioner",
    "status",
]


def subsample_for(balance: str, train_full: list[dict], n_classes: int, n: int, seed: int) -> list[dict]:
    """Return the train subsample for a (balance, size) cell."""
    if balance == "classwise":
        return classwise_subsample(train_full, n_classes, n, seed)
    if balance == "stratified":
        return stratified_subsample(train_full, n_classes, n, seed)
    if balance == "natural":
        # Size-matched to the balanced N-shot set, but drawn at natural proportions.
        total = len(classwise_subsample(train_full, n_classes, n, seed))
        return natural_subsample(train_full, n_classes, total, seed)
    raise ValueError(f"unknown balance '{balance}'")


def build_datasets(
    sizes: list[int], balances: list[str], repo: str, config: str, seed: int, data_dir: Path
) -> dict[tuple[str, int], Path]:
    """Prepare one dataset JSON per (balance, size) cell. Skips cells missing a class. Returns {(balance, size): path}."""
    ds = names = train_full = eval_rows = None
    n_classes = 0
    paths: dict[tuple[str, int], Path] = {}
    for n in sizes:
        for balance in balances:
            path = data_dir / f"ge_{balance}_{n}.json"
            if path.exists():
                paths[(balance, n)] = path
                continue
            if ds is None:
                ds, names = load_goemotions(repo, config)
                n_classes = len(names)
                train_full, eval_rows = list(ds["train"]), list(ds["validation"])
            print(f"Building dataset {path.name} ({balance}, {n}-shot) ...")
            train_rows = subsample_for(balance, train_full, n_classes, n, seed)
            present = int((label_matrix(train_rows, n_classes).sum(axis=0) > 0).sum())
            if present < n_classes:
                print(f"  SKIP: only {present}/{n_classes} classes present; AutoIntent needs every class in every split.")
                continue
            save_mapping(assemble_mapping(names, train_rows, eval_rows), path)
            paths[(balance, n)] = path
    return paths


def _train_hash(path: Path) -> str:
    """Content hash of a dataset's train split (used to dedup identical cells, e.g. collapsed stratified)."""
    train = json.loads(path.read_text(encoding="utf-8"))["train"]
    return hashlib.md5(json.dumps(train, sort_keys=True).encode()).hexdigest()


def _summarize(report: dict, balance: str, n: int, scoring: str, decision: str, status: str) -> dict[str, Any]:
    tm = report.get("test_metrics", {}) if report else {}
    modules = report.get("selected_modules", []) if report else []
    chosen = {m["node_type"]: m.get("module_name") for m in modules if "module_name" in m}
    return {
        "size": n,
        "balance": balance,
        "train_size": (report or {}).get("fed_split_sizes", {}).get("train"),
        "scoring_metric": scoring,
        "decision_metric": decision,
        "eval_f1": tm.get("decision_f1"),
        "eval_precision": tm.get("decision_precision"),
        "eval_recall": tm.get("decision_recall"),
        "eval_accuracy": tm.get("decision_accuracy"),
        "scorer": chosen.get("scoring"),
        "decisioner": chosen.get("decision"),
        "status": status,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_best(results: list[dict]) -> None:
    best: dict[tuple[int, str], dict] = {}
    for r in results:
        if r["eval_f1"] is None:
            continue
        key = (r["size"], r["balance"])
        if key not in best or r["eval_f1"] > best[key]["eval_f1"]:
            best[key] = r
    print("\n=== best (scoring, decision) per cell, by eval_f1 ===")
    for (size, balance), r in sorted(best.items()):
        print(
            f"  size={size:<4} {balance:<10} scoring={r['scoring_metric']:<24} "
            f"decision={r['decision_metric']:<20} -> f1={r['eval_f1']:.4f}"
        )


def run_sweep(
    *,
    sizes: list[int],
    balances: list[str],
    scoring_metrics: list[str],
    decision_metrics: list[str],
    preset: str,
    embedder_model: str | None,
    device: str | None,
    seed: int,
    logs_dir: Path,
    data_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> list[dict]:
    """Run the full grid and return the collected summary rows."""
    logs_dir, data_dir = Path(logs_dir), Path(data_dir)
    total = len(sizes) * len(balances) * len(scoring_metrics) * len(decision_metrics)
    print(
        f"Sweep plan: {len(sizes)} sizes x {len(balances)} balances x {len(scoring_metrics)} scoring "
        f"x {len(decision_metrics)} decision = {total} runs"
    )

    if dry_run:
        for n in sizes:
            for balance in balances:
                for scoring in scoring_metrics:
                    for decision in decision_metrics:
                        print(f"  gm-{balance}-{n}-s_{scoring}-d_{decision}")
        print("dry-run: no datasets built, no fits executed.")
        return []

    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = build_datasets(sizes, balances, DEFAULT_REPO, DEFAULT_CONFIG, seed, data_dir)
    summary_csv = logs_dir / "sweep_summary.csv"

    results: list[dict] = []
    seen: dict[tuple[str, str, str], str] = {}  # (train_hash, scoring, decision) -> exp_name already computed
    done = 0
    # Outer loop over datasets so AutoIntent's embedding cache stays warm within a (balance, size) cell.
    for (balance, n), data_path in paths.items():
        train_hash = _train_hash(data_path)
        for scoring in scoring_metrics:
            for decision in decision_metrics:
                done += 1
                exp_name = f"gm-{balance}-{n}-s_{scoring}-d_{decision}"
                out_path = metrics_path(logs_dir, exp_name)
                dedup_key = (train_hash, scoring, decision)
                print(f"\n[{done}/{total}] {exp_name}")

                status = "ok"
                report: dict | None = None
                if out_path.exists() and not overwrite:
                    print("  exists -> skip (resume)")
                    report = json.loads(out_path.read_text(encoding="utf-8"))
                elif dedup_key in seen:
                    print(f"  identical dataset to {seen[dedup_key]} -> reuse result")
                    report = json.loads(metrics_path(logs_dir, seen[dedup_key]).read_text(encoding="utf-8"))
                    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    if overwrite:
                        out_path.unlink(missing_ok=True)
                        run_dir = logs_dir / exp_name
                        if run_dir.exists():
                            shutil.rmtree(run_dir)
                    try:
                        report = run_experiment(
                            data_path=data_path,
                            preset=preset,
                            exp_name=exp_name,
                            logs_dir=logs_dir,
                            out_path=out_path,
                            embedder_model=embedder_model,
                            device=device,
                            scoring_metric=scoring,
                            decision_metric=decision,
                            seed=seed,
                            dump_modules=False,
                        )
                        seen[dedup_key] = exp_name
                    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
                        status = f"failed: {exc}"
                        print(f"  FAILED: {exc}")

                results.append(_summarize(report or {}, balance, n, scoring, decision, status))
                _write_csv(summary_csv, results)  # rewrite each iteration for crash safety

    print(f"\nWrote summary: {summary_csv}")
    _print_best(results)
    return results
