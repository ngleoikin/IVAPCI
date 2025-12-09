"""Analyze diagnostics outputs from simulation runs.

Implements the reporting described in ``docs/pacd_benchmark_design.md``:
* Load ``simulation_diagnostics_results.csv``.
* Compute Spearman correlations between diagnostics and ATE errors per method.
* Plot scatter relations for proxy strength, residual risk, and proximal condition number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


_DIAGNOSTIC_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("proxy_score", "abs_err"),
    ("resid_score", "abs_err"),
    ("prox_cond_score", "abs_err"),
)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_dir: Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = sorted(df["method"].unique())
    colors = plt.cm.tab10(range(len(methods)))
    for color, method in zip(colors, methods):
        subset = df[df["method"] == method]
        ax.scatter(subset[x], subset[y], label=method, alpha=0.7, s=30, color=color)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()

    out_path = output_dir / f"{x}_vs_{y}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _print_spearman(df: pd.DataFrame, x: str, y: str) -> None:
    print(f"{x} vs {y} (Spearman)")
    for method, group in df.groupby("method"):
        if len(group) < 2:
            corr = float("nan")
        else:
            corr = group[x].corr(group[y], method="spearman")
        print(f"  [method={method}] rho = {corr:.4f}")


def analyze_simulation_diagnostics(
    diagnostics_path: Path,
    output_dir: Path,
    pairs: Iterable[Tuple[str, str]] = _DIAGNOSTIC_PAIRS,
) -> None:
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {diagnostics_path}")

    df = pd.read_csv(diagnostics_path)
    required_cols = {"method"}
    for x, y in pairs:
        required_cols.update([x, y])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Diagnostics CSV missing columns: {missing}")

    _ensure_output_dir(output_dir)

    for x, y in pairs:
        _print_spearman(df, x, y)
        title = f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}"
        out_path = _scatter_plot(df, x, y, output_dir, title)
        print(f"Saved plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulation diagnostics outputs.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("simulation_diagnostics_results.csv"),
        help="Path to diagnostics CSV produced by run_diagnostics_on_simulation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_simulation_diagnostics(args.diagnostics, args.output_dir)


if __name__ == "__main__":
    main()
