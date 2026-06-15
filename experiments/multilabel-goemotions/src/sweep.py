"""Grid sweep to find the best scoring/decision target metric across dataset sizes and balance modes.

For every (size, balance, seed) cell a dataset is prepared once, then every (scoring_metric,
decision_metric) pair is optimized on it. Each cell is run once per seed (different subsample + AutoIntent
seed); per-run results go to sweep_runs.csv and a seed-aggregated view (mean/std) to sweep_summary.csv,
so a long sweep can be interrupted and resumed.

Balance modes:
- classwise:  cap N samples/class -> balanced N-shot. Always feasible.
- stratified: floor of N samples/class -> imbalanced, but collapses to the full train once N exceeds the
              rarest class's total (~77 for GoEmotions' "grief"), so 100- and 500-shot become identical
              (identical (dataset, seed) cells are deduped and not re-fit).
- natural:    proportion-preserving sample sized to match classwise(N) -> size-matched imbalanced. May be
              INFEASIBLE at small N: rare classes can drop out, and AutoIntent requires every class in every
              split (including the validation it carves from train). Such cells are skipped with a warning.

Every fit is wrapped in try/except: a cell that AutoIntent can't handle is recorded as failed and the
sweep continues.
"""

import csv
import hashlib
import json
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from autointent.metrics import DICISION_METRICS_MULTILABEL, SCORING_METRICS_MULTILABEL
from cyclopts import Parameter
from loguru import logger

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

EXP_DIR = Path(__file__).resolve().parent.parent

ALL_SCORING = sorted(SCORING_METRICS_MULTILABEL)
ALL_DECISION = sorted(DICISION_METRICS_MULTILABEL)
BALANCES = ("classwise", "stratified", "natural")

RUN_FIELDS = [
    "seed", "size", "balance", "train_size", "scoring_metric", "decision_metric",
    "eval_f1", "eval_precision", "eval_recall", "eval_accuracy", "scorer", "decisioner", "status",
]
SUMMARY_FIELDS = [
    "size", "balance", "scoring_metric", "decision_metric", "n_seeds", "train_size",
    "f1_mean", "f1_std", "precision_mean", "recall_mean", "accuracy_mean", "scorer", "decisioner",
]


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for the target-metric sweep (also usable directly from Python)."""

    sizes: Annotated[list[int], Parameter(help="Shot counts (per-class budget).", consume_multiple=True)] = field(
        default_factory=lambda: [10, 100, 500]
    )
    balances: Annotated[
        list[str], Parameter(help="Subset of: classwise, stratified, natural.", consume_multiple=True)
    ] = field(default_factory=lambda: ["classwise", "stratified"])
    scoring_metrics: Annotated[
        list[str], Parameter(help="Scoring-node target metrics to try.", consume_multiple=True)
    ] = field(default_factory=lambda: list(ALL_SCORING))
    decision_metrics: Annotated[
        list[str], Parameter(help="Decision-node target metrics to try.", consume_multiple=True)
    ] = field(default_factory=lambda: list(ALL_DECISION))
    seeds: Annotated[
        list[int], Parameter(help="Seeds; each cell runs once per seed and is averaged.", consume_multiple=True)
    ] = field(default_factory=lambda: [42])
    preset: Annotated[str, Parameter(help="AutoIntent search-space preset.")] = "classic-light"
    embedder_model: Annotated[str | None, Parameter(help="Override the preset's embedder.")] = None
    device: Annotated[Literal["cpu", "cuda", "mps"] | None, Parameter(help="Torch device for the embedder.")] = None
    logs_dir: Annotated[Path, Parameter(help="Directory for logs/metrics/summary.")] = EXP_DIR / "logs"
    data_dir: Annotated[Path, Parameter(help="Directory for prepared datasets.")] = EXP_DIR / "data"
    overwrite: Annotated[bool, Parameter(help="Rerun cells even if their metrics already exist.")] = False
    dry_run: Annotated[bool, Parameter(help="Print the run plan and exit (no datasets, no fits).")] = False


def subsample_for(
    balance: str, train_full: list[dict[str, Any]], n_classes: int, n: int, seed: int
) -> list[dict[str, Any]]:
    """Return the train subsample for a (balance, size) cell."""
    if balance == "classwise":
        return classwise_subsample(train_full, n_classes, n, seed)
    if balance == "stratified":
        return stratified_subsample(train_full, n_classes, n, seed)
    if balance == "natural":
        total = len(classwise_subsample(train_full, n_classes, n, seed))
        return natural_subsample(train_full, n_classes, total, seed)
    msg = f"unknown balance '{balance}'"
    raise ValueError(msg)


def build_datasets(
    sizes: list[int], balances: list[str], seeds: list[int], repo: str, config: str, data_dir: Path
) -> dict[tuple[str, int, int], Path]:
    """Prepare one dataset JSON per (balance, size, seed). Skips cells missing a class."""
    cells = [(balance, n, seed) for n in sizes for balance in balances for seed in seeds]
    candidate = {cell: data_dir / f"ge_{cell[0]}_{cell[1]}_s{cell[2]}.json" for cell in cells}
    if all(path.exists() for path in candidate.values()):
        return candidate

    ds, names = load_goemotions(repo, config)
    n_classes = len(names)
    train_full, eval_rows = list(ds["train"]), list(ds["validation"])

    paths: dict[tuple[str, int, int], Path] = {}
    for cell in cells:
        balance, n, seed = cell
        path = candidate[cell]
        if path.exists():
            paths[cell] = path
            continue
        logger.info("Building dataset {} ({}, {}-shot, seed {}) ...", path.name, balance, n, seed)
        train_rows = subsample_for(balance, train_full, n_classes, n, seed)
        present = int((label_matrix(train_rows, n_classes).sum(axis=0) > 0).sum())
        if present < n_classes:
            logger.warning("SKIP: only {}/{} classes present; AutoIntent needs every class.", present, n_classes)
            continue
        save_mapping(assemble_mapping(names, train_rows, eval_rows), path)
        paths[cell] = path
    return paths


def _train_hash(path: Path) -> str:
    train = json.loads(path.read_text(encoding="utf-8"))["train"]
    return hashlib.md5(json.dumps(train, sort_keys=True).encode(), usedforsecurity=False).hexdigest()


def _summarize(
    report: dict[str, Any], balance: str, n: int, scoring: str, decision: str, seed: int, status: str
) -> dict[str, Any]:
    tm = report.get("test_metrics", {}) if report else {}
    modules = report.get("selected_modules", []) if report else []
    chosen = {m["node_type"]: m.get("module_name") for m in modules if "module_name" in m}
    return {
        "seed": seed,
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


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average eval metrics across seeds per (size, balance, scoring, decision) cell."""
    groups: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r["status"] != "ok" or r["eval_f1"] is None:
            continue
        groups[(r["size"], r["balance"], r["scoring_metric"], r["decision_metric"])].append(r)

    rows: list[dict[str, Any]] = []
    for (size, balance, sm, dm), rs in groups.items():

        def mean(name: str, rs: list[dict[str, Any]] = rs) -> float:
            return round(statistics.mean([float(r[name]) for r in rs]), 4)

        f1 = [float(r["eval_f1"]) for r in rs]
        rows.append({
            "size": size,
            "balance": balance,
            "scoring_metric": sm,
            "decision_metric": dm,
            "n_seeds": len(rs),
            "train_size": rs[0]["train_size"],
            "f1_mean": round(statistics.mean(f1), 4),
            "f1_std": round(statistics.stdev(f1), 4) if len(f1) > 1 else 0.0,
            "precision_mean": mean("eval_precision"),
            "recall_mean": mean("eval_recall"),
            "accuracy_mean": mean("eval_accuracy"),
            "scorer": Counter(r["scorer"] for r in rs).most_common(1)[0][0],
            "decisioner": Counter(r["decisioner"] for r in rs).most_common(1)[0][0],
        })
    return sorted(rows, key=lambda r: (r["size"], r["balance"], r["scoring_metric"], r["decision_metric"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_best(summary: list[dict[str, Any]]) -> None:
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for r in summary:
        key = (r["size"], r["balance"])
        if key not in best or r["f1_mean"] > best[key]["f1_mean"]:
            best[key] = r
    logger.info("Best (scoring, decision) per cell, by mean eval_f1 over seeds:")
    for (size, balance), r in sorted(best.items()):
        logger.info(
            "size={:<4} {:<10} scoring={:<24} decision={:<20} -> f1={:.4f} ± {:.4f} (n={})",
            size,
            balance,
            r["scoring_metric"],
            r["decision_metric"],
            r["f1_mean"],
            r["f1_std"],
            r["n_seeds"],
        )


def _run_cell(
    cfg: SweepConfig,
    exp_name: str,
    data_path: Path,
    scoring: str,
    decision: str,
    seed: int,
    dedup_key: tuple[str, str, str, int],
    seen: dict[tuple[str, str, str, int], str],
) -> tuple[dict[str, Any] | None, str]:
    """Run, resume, or reuse one sweep cell. Returns (report or None, status); updates ``seen``."""
    logs_dir = Path(cfg.logs_dir)
    out_path = metrics_path(logs_dir, exp_name)
    if out_path.exists() and not cfg.overwrite:
        logger.info("exists -> skip (resume)")
        return json.loads(out_path.read_text(encoding="utf-8")), "ok"
    if dedup_key in seen:
        logger.info("identical (dataset, seed) to {} -> reuse result", seen[dedup_key])
        report = json.loads(metrics_path(logs_dir, seen[dedup_key]).read_text(encoding="utf-8"))
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report, "ok"
    if cfg.overwrite:
        out_path.unlink(missing_ok=True)
        run_dir = logs_dir / exp_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
    try:
        report = run_experiment(
            data_path=data_path,
            preset=cfg.preset,
            exp_name=exp_name,
            logs_dir=logs_dir,
            embedder_model=cfg.embedder_model,
            device=cfg.device,
            scoring_metric=scoring,
            decision_metric=decision,
            seed=seed,
            dump_modules=False,
        )
    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
        logger.exception("cell failed")
        return None, f"failed: {exc}"
    seen[dedup_key] = exp_name
    return report, "ok"


def _print_plan(cfg: SweepConfig) -> None:
    for n in cfg.sizes:
        for balance in cfg.balances:
            for seed in cfg.seeds:
                for scoring in cfg.scoring_metrics:
                    for decision in cfg.decision_metrics:
                        logger.info("gm-{}-{}-s_{}-d_{}-seed{}", balance, n, scoring, decision, seed)


def run_sweep(cfg: SweepConfig) -> list[dict[str, Any]]:
    """Run the full grid and return the collected per-run summary rows."""
    bad = [b for b in cfg.balances if b not in BALANCES]
    if bad:
        msg = f"unknown balance(s) {bad}; choose from {list(BALANCES)}"
        raise SystemExit(msg)
    logs_dir, data_dir = Path(cfg.logs_dir), Path(cfg.data_dir)
    total = (
        len(cfg.sizes) * len(cfg.balances) * len(cfg.scoring_metrics) * len(cfg.decision_metrics) * len(cfg.seeds)
    )
    logger.info(
        "Sweep plan: {} sizes x {} balances x {} scoring x {} decision x {} seeds = {} runs",
        len(cfg.sizes),
        len(cfg.balances),
        len(cfg.scoring_metrics),
        len(cfg.decision_metrics),
        len(cfg.seeds),
        total,
    )

    if cfg.dry_run:
        _print_plan(cfg)
        logger.info("dry-run: no datasets built, no fits executed.")
        return []

    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = build_datasets(cfg.sizes, cfg.balances, cfg.seeds, DEFAULT_REPO, DEFAULT_CONFIG, data_dir)
    runs_csv, summary_csv = logs_dir / "sweep_runs.csv", logs_dir / "sweep_summary.csv"

    runs: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, int], str] = {}  # (train_hash, scoring, decision, seed) -> exp_name
    done = 0
    for (balance, n, seed), data_path in paths.items():
        train_hash = _train_hash(data_path)
        for scoring in cfg.scoring_metrics:
            for decision in cfg.decision_metrics:
                done += 1
                exp_name = f"gm-{balance}-{n}-s_{scoring}-d_{decision}-seed{seed}"
                dedup_key = (train_hash, scoring, decision, seed)
                logger.info("[{}/{}] {}", done, total, exp_name)
                report, status = _run_cell(cfg, exp_name, data_path, scoring, decision, seed, dedup_key, seen)
                runs.append(_summarize(report or {}, balance, n, scoring, decision, seed, status))
                _write_csv(runs_csv, runs, RUN_FIELDS)

    summary = _aggregate(runs)
    _write_csv(summary_csv, summary, SUMMARY_FIELDS)
    logger.info("Wrote {} and {}", runs_csv, summary_csv)
    _print_best(summary)
    return runs
