"""Compare baseline `basic-fs-*` JSONL reports to CV-aggregated tool-suggest reports (README filenames)."""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from .constants import PASSED_EPS
from .models import CaseRow, ExperimentHeader

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
    """Print Rich tables: pass rates, then usage (basic vs CV per model)."""
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

    use_tbl = Table(title="Usage (header totals): basic vs CV", box=box.SIMPLE, width=120)
    use_tbl.add_column("Model", overflow="fold", max_width=14)
    use_tbl.add_column("in_B", justify="right", no_wrap=True)
    use_tbl.add_column("in_CV", justify="right", no_wrap=True)
    use_tbl.add_column("out_B", justify="right", no_wrap=True)
    use_tbl.add_column("out_CV", justify="right", no_wrap=True)
    use_tbl.add_column("req_B", justify="right", no_wrap=True)
    use_tbl.add_column("req_CV", justify="right", no_wrap=True)
    use_tbl.add_column("cost_B", justify="right", no_wrap=True)
    use_tbl.add_column("cost_CV", justify="right", no_wrap=True)

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
        use_tbl.add_row(
            label,
            f"{bh.input_tokens / 1e6:.2f}M",
            f"{ch.input_tokens / 1e6:.2f}M",
            f"{bh.output_tokens / 1e3:.1f}k",
            f"{ch.output_tokens / 1e3:.1f}k",
            f"{bh.requests:.2f}",
            f"{ch.requests:.2f}",
            f"{bh.cost:.3f}",
            f"{ch.cost:.3f}",
        )

    console.print(pass_tbl)
    console.print()
    console.print(use_tbl)
    if missing:
        console.print()
        console.print("[yellow]Missing files (skipped rows):[/yellow]")
        for m in missing:
            console.print(f"  {m}")
