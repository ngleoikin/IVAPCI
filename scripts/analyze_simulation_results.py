"""Analyze simulation benchmark summaries and create plots.

This script follows the design in ``docs/pacd_benchmark_design.md``:
* Load ``simulation_benchmark_summary.csv``.
* Visualize per-scenario mean absolute error and RMSE across methods.
* Optionally export LaTeX tables for papers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    ylabel: str,
    title: str,
) -> Path:
    """Create a grouped bar plot of a metric across scenarios/methods."""
    pivot = df.pivot(index="scenario", columns="method", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scenario")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = output_dir / f"{metric}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _save_latex_table(df: pd.DataFrame, output_dir: Path) -> Path:
    """Export the summary table to LaTeX for papers."""
    latex_path = output_dir / "simulation_benchmark_summary.tex"
    latex = df.to_latex(index=False, float_format="{:.4f}".format)
    latex_path.write_text(latex)
    return latex_path


def analyze_simulation_results(
    summary_path: Path,
    output_dir: Path,
    export_latex: bool = False,
    metrics: Iterable[str] = ("mean_abs_err", "rmse"),
) -> None:
    """Load summary CSV, plot metrics, and optionally export LaTeX."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    required_cols = {"scenario", "method"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {missing}")

    _ensure_output_dir(output_dir)

    for metric in metrics:
        if metric not in df.columns:
            continue
        ylabel = metric.replace("_", " ").title()
        title = f"{ylabel} by Scenario and Method"
        out_path = _plot_metric(df, metric, output_dir, ylabel, title)
        print(f"Saved plot: {out_path}")

    if export_latex:
        latex_path = _save_latex_table(df, output_dir)
        print(f"Saved LaTeX table: {latex_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulation benchmark summaries.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("simulation_benchmark_summary.csv"),
        help="Path to simulation benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated plots and tables.",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Export the summary table to LaTeX in addition to plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_simulation_results(args.summary, args.output_dir, args.export_latex)


if __name__ == "__main__":
    main()
