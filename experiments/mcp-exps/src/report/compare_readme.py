"""Compare baseline `basic-fs-*` JSONL reports to CV-aggregated tool-suggest reports (README filenames)."""

import sys
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


def _redo_jsonl(reports_dir: Path, stem_prefix: str | None) -> Path | None:
    """Newest JSONL under ``reports_dir`` with name starting with ``stem_prefix``.

    Include ``_test`` in the prefix so ``gpt54`` does not match ``gpt54mini`` / ``gpt54nano``.
    """
    if stem_prefix is None:
        return None
    candidates = list(reports_dir.glob(f"{stem_prefix}*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# (display label, basic stem prefix, OOS CV stem prefix, accum CV stem prefix) — None skips that column.
# Stem prefixes match files like ``basic-fs-redo-gpt54_test_0_<trace>.jsonl`` (see README REDO section).
README_REDO_TRIPLET: list[tuple[str, str | None, str | None, str | None]] = [
    ("Opus 4.6", "basic-fs-redo-opus46_test", None, None),
    ("Haiku 4.5", "basic-fs-redo-haiku45_test", None, "ts-fs-repro-redo-oos-accum-cv-haiku45_test"),
    (
        "GPT-5.4",
        "basic-fs-redo-gpt54_test",
        "ts-fs-repro-redo-oos-cv-gpt54_test",
        "ts-fs-repro-redo-oos-accum-cv-gpt54_test",
    ),
    (
        "GPT-5.4 mini",
        "basic-fs-redo-gpt54mini_test",
        "ts-fs-repro-redo-oos-cv-gpt54mini_test",
        "ts-fs-repro-redo-oos-accum-cv-gpt54mini_test",
    ),
    (
        "GPT-5.4 nano",
        "basic-fs-redo-gpt54nano_test",
        "ts-fs-repro-redo-oos-cv-gpt54nano_test",
        "ts-fs-repro-redo-oos-accum-cv-gpt54nano_test",
    ),
    (
        "Qwen3 Coder+",
        "basic-fs-redo-qwen3coder_test",
        "ts-fs-repro-redo-oos-cv-qwen3coderplus_test",
        "ts-fs-repro-redo-oos-accum-cv-qwen3coderplus_test",
    ),
    ("DeepSeek V3.2", "basic-fs-redo-deepseekv32_test", "ts-fs-repro-redo-oos-cv-deepseekv32_test", None),
]


def _fmt_redo_usage_cells(u: _CaseUsageAgg | None) -> tuple[str, str, str, str, str]:
    if u is None or u.n == 0:
        return "—", "—", "—", "—", "—"
    return (
        str(u.n),
        f"{u.mean_in / 1e3:.0f}k",
        f"{u.mean_out / 1e3:.1f}k",
        f"{u.mean_req:.2f}",
        f"{u.mean_cost:.4f}",
    )


@dataclass(frozen=True, slots=True)
class _RedoReadmeBundle:
    pass_rows: list[list[str]]
    use_rows: list[list[str]]
    missing: list[str]


def _collect_redo_readme_rows(reports_dir: Path) -> _RedoReadmeBundle:
    missing: list[str] = []
    pass_rows: list[list[str]] = []
    use_rows: list[list[str]] = []

    for label, b_pre, cv_pre, acc_pre in README_REDO_TRIPLET:
        b_path = _redo_jsonl(reports_dir, b_pre)
        cv_path = _redo_jsonl(reports_dir, cv_pre)
        acc_path = _redo_jsonl(reports_dir, acc_pre)
        for kind, pre, p in ("basic", b_pre, b_path), ("cv", cv_pre, cv_path), ("accum", acc_pre, acc_pre and acc_path):
            if pre and p is None:
                missing.append(f"{label} / {kind}: {reports_dir}/{pre}*.jsonl")

        hs_b, ss_b = "—", "—"
        ub: _CaseUsageAgg | None = None
        if b_path is not None:
            bh, bc = load_report_jsonl(b_path)
            hs_b = f"{hard_pass_rate(bh):.1%}"
            ss_b = f"{soft_pass_rate(bc):.1%}"
            ub = _aggregate_case_usage(bc)

        hs_cv, ss_cv = "—", "—"
        uc: _CaseUsageAgg | None = None
        if cv_path is not None:
            ch, cc = load_report_jsonl(cv_path)
            hs_cv = f"{hard_pass_rate(ch):.1%}"
            ss_cv = f"{soft_pass_rate(cc):.1%}"
            uc = _aggregate_case_usage(cc)

        hs_a, ss_a = "—", "—"
        ua: _CaseUsageAgg | None = None
        if acc_path is not None:
            ah, ac = load_report_jsonl(acc_path)
            hs_a = f"{hard_pass_rate(ah):.1%}"
            ss_a = f"{soft_pass_rate(ac):.1%}"
            ua = _aggregate_case_usage(ac)

        pass_rows.append([label, hs_b, hs_cv, hs_a, ss_b, ss_cv, ss_a])

        b_u, cv_u, a_u = _fmt_redo_usage_cells(ub), _fmt_redo_usage_cells(uc), _fmt_redo_usage_cells(ua)
        use_rows.append(
            [
                label,
                b_u[0],
                cv_u[0],
                a_u[0],
                b_u[1],
                cv_u[1],
                a_u[1],
                b_u[2],
                cv_u[2],
                a_u[2],
                b_u[3],
                cv_u[3],
                a_u[3],
                b_u[4],
                cv_u[4],
                a_u[4],
            ]
        )

    return _RedoReadmeBundle(pass_rows=pass_rows, use_rows=use_rows, missing=missing)


def _write_redo_markdown(bundle: _RedoReadmeBundle) -> None:
    out = sys.stdout
    lines = [
        "### Pass rates (REDO)\n",
        "| Model | Hard basic | Hard CV (OOS) | Hard CV+accum | Soft basic | Soft CV (OOS) | Soft CV+accum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {' | '.join(row)} |" for row in bundle.pass_rows],
        "",
        (
            "- **Hard** = `passed_tasks / total_tasks` from each JSONL header. "
            "**Soft** = fraction of evaluator scores equal to 1.0 across all case rows."
        ),
        "",
        "### Usage — per-case mean over case rows (REDO)\n",
        (
            "| Model | N_b | N_cv | N_acc | in tok B | in tok CV | in tok acc | "
            "out tok B | out tok CV | out tok acc | req B | req CV | req acc | "
            "cost B | cost CV | cost acc |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |"
        ),
        *[f"| {' | '.join(row)} |" for row in bundle.use_rows],
        "",
        (
            "Per-case means (not merged CV header totals). Some **basic** runs still show **cost 0** "
            "in rollups when Logfire did not attribute spend to leaf spans."
        ),
    ]
    out.write("\n".join(lines))
    if bundle.missing:
        out.write("\n\nMissing (expected for models not run in a condition):\n\n")
        out.write("\n".join(f"- {m}" for m in bundle.missing))
    out.write("\n")


def print_redo_readme_table(reports_dir: Path, *, markdown: bool = False) -> None:
    """Baseline ``basic-fs-redo-*`` vs OOS CV vs accum CV (REDO experiment names; trace suffix ignored)."""
    bundle = _collect_redo_readme_rows(reports_dir)
    if markdown:
        _write_redo_markdown(bundle)
        return

    console = Console(width=140, soft_wrap=True)
    pass_tbl = Table(
        title="Pass rates: REDO basic-fs vs ts-repro OOS CV vs ts-repro accum CV",
        box=box.SIMPLE,
        width=140,
    )
    pass_tbl.add_column("Model", max_width=16)
    for col in ("hard_B", "hard_CV", "hard_acc", "soft_B", "soft_CV", "soft_acc"):
        pass_tbl.add_column(col, justify="right", no_wrap=True)

    use_tbl = Table(
        title="Usage per case (mean) — N and means for basic / OOS CV / accum CV",
        box=box.SIMPLE,
        width=140,
    )
    use_tbl.add_column("Model", overflow="fold", max_width=12)
    for col in (
        "N_b",
        "N_cv",
        "N_a",
        "in_B",
        "in_CV",
        "in_a",
        "out_B",
        "out_CV",
        "out_a",
        "req_B",
        "req_CV",
        "req_a",
        "cst_B",
        "cst_CV",
        "cst_a",
    ):
        use_tbl.add_column(col, justify="right", no_wrap=True)

    for row in bundle.pass_rows:
        pass_tbl.add_row(*row)
    for row in bundle.use_rows:
        use_tbl.add_row(*row)

    console.print(pass_tbl)
    console.print()
    console.print(use_tbl)
    if bundle.missing:
        console.print()
        console.print("[dim]Missing paths (no JSONL matched):[/dim]")
        for m in bundle.missing:
            console.print(f"  {m}")
