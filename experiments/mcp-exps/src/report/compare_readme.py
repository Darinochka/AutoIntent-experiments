"""Compare baseline `basic-fs-*` JSONL reports to CV-aggregated tool-suggest reports (README filenames)."""

from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from .constants import PASSED_EPS
from .models import CaseRow, ExperimentHeader


@dataclass(frozen=True, slots=True)
class _CaseUsageAgg:
    n: int
    mean_in: float
    mean_out: float
    mean_req: float
    mean_cost: float
    sum_in: float
    sum_out: float
    sum_req: float
    sum_cost: float


def _aggregate_case_usage(cases: list[CaseRow]) -> _CaseUsageAgg:
    """Token/cost sums and per-row means (fair across single-run vs N-trace CV merge)."""
    n = len(cases)
    if n == 0:
        return _CaseUsageAgg(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sin = sum(c.metrics.input_tokens for c in cases)
    sout = sum(c.metrics.output_tokens for c in cases)
    sreq = sum(c.metrics.requests for c in cases)
    scst = sum(c.metrics.cost for c in cases)
    return _CaseUsageAgg(
        n=n,
        mean_in=sin / n,
        mean_out=sout / n,
        mean_req=sreq / n,
        mean_cost=scst / n,
        sum_in=sin,
        sum_out=sout,
        sum_req=sreq,
        sum_cost=scst,
    )


# (label, basic stem without .jsonl, cv stem without .jsonl) — matches reports/ + README baselines / CV script names.
README_BASIC_VS_CV: list[tuple[str, str, str]] = [
    ("Haiku 4.5", "basic-fs-haiku-v2_test_0", "cv-readme-haiku-4-5"),
    ("Opus 4.6", "basic-fs-opus-4-6_true_test_0", "cv-readme-opus-4-6"),
    ("GPT-5.4", "basic-fs-gpt-5-4-true_test_0", "cv-readme-gpt-5-4"),
    ("GPT-5.4 mini", "basic-fs-gpt-5-4-mini_test_0", "cv-gpt54-mini-aggregated"),
    ("GPT-5.4 nano", "basic-fs-gpt-5-4-nano_test_0", "cv-readme-gpt-5-4-nano"),
    ("Qwen3 Coder+", "basic-fs-qwen3-coder-plus-v2_test_0", "cv-readme-qwen3-coder-plus"),
    ("DeepSeek V3.2", "basic-fs-deepseek-v-3-2_test_0", "cv-readme-deepseek-v3-2"),
]


def load_report_jsonl(path: Path) -> tuple[ExperimentHeader, list[CaseRow]]:
    """Read JSONL: first line header, remaining lines cases."""
    with path.open(encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line.strip():
            msg = f"empty report: {path}"
            raise ValueError(msg)
        header = ExperimentHeader.model_validate_json(header_line)
        cases: list[CaseRow] = []
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            cases.append(CaseRow.model_validate_json(line))
    return header, cases


def soft_pass_rate(cases: list[CaseRow]) -> float:
    """Fraction of evaluator slots that scored 1.0 (missing values skipped)."""
    passed = 0
    total = 0
    for c in cases:
        for ev in c.scores.values():
            if ev.value is None:
                continue
            total += 1
            if abs(float(ev.value) - 1.0) < PASSED_EPS:
                passed += 1
    return passed / total if total else 0.0


def hard_pass_rate(header: ExperimentHeader) -> float:
    """Share of tasks with all evaluators at 1.0 (from JSONL header)."""
    if header.total_tasks <= 0:
        return 0.0
    return header.passed_tasks / header.total_tasks


def print_basic_vs_cv_table(reports_dir: Path) -> None:
    """Print Rich tables: pass rates, then per-case mean usage (fair) and optional sanity totals.

    For CV, ``aggregate-links`` / merged JSONL **headers** sum `chat` metrics over **every** merged
    ``trace_id``, while a single `basic` run is one trace — comparing raw header tokens to a baseline
    is misleading. Here we compare **means (and sums) of per-case rows**, which are comparable when
    both files list one row per task execution (e.g. 25 rows).
    """
    console = Console(width=120, soft_wrap=True)
    missing: list[str] = []

    pass_tbl = Table(
        title="Pass rates: basic-fs (single run) vs tool-suggest OOS CV (5 folds)",
        box=box.SIMPLE,
        width=120,
    )
    pass_tbl.add_column("Model", max_width=16)
    pass_tbl.add_column("hard_B", justify="right", no_wrap=True)
    pass_tbl.add_column("hard_CV", justify="right", no_wrap=True)
    pass_tbl.add_column("soft_B", justify="right", no_wrap=True)
    pass_tbl.add_column("soft_CV", justify="right", no_wrap=True)

    use_mean_tbl = Table(
        title="Usage per case (mean over case rows) — use for model comparison; both sides: N = row count",
        box=box.SIMPLE,
        width=120,
    )
    use_mean_tbl.add_column("Model", overflow="fold", max_width=12)
    use_mean_tbl.add_column("N_b", justify="right", no_wrap=True)
    use_mean_tbl.add_column("N_c", justify="right", no_wrap=True)
    use_mean_tbl.add_column("in_mB", justify="right", no_wrap=True)
    use_mean_tbl.add_column("in_mC", justify="right", no_wrap=True)
    use_mean_tbl.add_column("out_mB", justify="right", no_wrap=True)
    use_mean_tbl.add_column("out_mC", justify="right", no_wrap=True)
    use_mean_tbl.add_column("req_mB", justify="right", no_wrap=True)
    use_mean_tbl.add_column("req_mC", justify="right", no_wrap=True)
    use_mean_tbl.add_column("cst_mB", justify="right", no_wrap=True)
    use_mean_tbl.add_column("cst_mC", justify="right", no_wrap=True)

    for label, basic_stem, cv_stem in README_BASIC_VS_CV:
        basic_path = (reports_dir / basic_stem).with_suffix(".jsonl")
        cv_path = (reports_dir / cv_stem).with_suffix(".jsonl")
        if not basic_path.is_file():
            missing.append(str(basic_path))
            continue
        if not cv_path.is_file():
            missing.append(str(cv_path))
            continue

        bh, bc = load_report_jsonl(basic_path)
        ch, cc = load_report_jsonl(cv_path)
        ub = _aggregate_case_usage(bc)
        uc = _aggregate_case_usage(cc)

        hb = hard_pass_rate(bh)
        hc = hard_pass_rate(ch)
        sb = soft_pass_rate(bc)
        sc = soft_pass_rate(cc)

        pass_tbl.add_row(
            label,
            f"{hb:.1%}",
            f"{hc:.1%}",
            f"{sb:.1%}",
            f"{sc:.1%}",
        )
        use_mean_tbl.add_row(
            label,
            f"{ub.n}",
            f"{uc.n}",
            f"{ub.mean_in / 1e3:.0f}k",
            f"{uc.mean_in / 1e3:.0f}k",
            f"{ub.mean_out / 1e3:.1f}k",
            f"{uc.mean_out / 1e3:.1f}k",
            f"{ub.mean_req:.2f}",
            f"{uc.mean_req:.2f}",
            f"{ub.mean_cost:.4f}",
            f"{uc.mean_cost:.4f}",
        )

    console.print(pass_tbl)
    console.print()
    console.print(use_mean_tbl)
    console.print(
        "\n[dim]Sanity: sum of per-case inputs should match JSONL header totals when leaf metrics were "
        "attributed. CV headers sum all merged traces and should match Σ per-case. "
        "The old table compared basic header to CV header, which is unfair when the CV file merges K traces "
        "(K = trace_id segments). Compare means above instead. If you need raw header totals, run: "
        "`uv run report.py table` on each file.[/dim]"
    )
    if missing:
        console.print()
        console.print("[yellow]Missing files (skipped rows):[/yellow]")
        for m in missing:
            console.print(f"  {m}")
