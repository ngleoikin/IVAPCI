"""Run simulation benchmarks across scenarios and methods.

This script follows the structure described in
``docs/pacd_benchmark_design.md``. It simulates datasets for a list of
scenarios, evaluates multiple causal estimators, records ATE accuracy
metrics, and produces both per-run and aggregated CSV summaries.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure repository root is on the import path when invoked as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baselines import DRGLMEstimator, DRRFEstimator, NaiveEstimator, OracleUEstimator
from models.ivapci_gold import IVAPCIGoldEstimator
from models.ivapci_v21 import IVAPCIv21Estimator, IVAPCIv21GLMEstimator, IVAPCIv21RADREstimator
from models.ivapci_v2_1_pacd_glm import IVAPCIPACDTGLMEstimator
from models.ivapci_v31_pacd_encoder import IVAPCIv31PACDEncoderEstimator, IVAPCIv31RADREstimator
from models.ivapci_v31_theory import IVAPCIv31TheoryRADREstimator
from models.pacdt_v30 import PACDTv30Estimator
from simulators.simulators import list_scenarios, simulate_scenario


def _build_estimator(name: str):
    if name == "naive":
        return NaiveEstimator()
    if name == "dr_glm":
        return DRGLMEstimator()
    if name == "dr_rf":
        return DRRFEstimator()
    if name == "oracle_U":
        return OracleUEstimator()
    if name == "ivapci_v2_1":
        return IVAPCIv21Estimator()
    if name == "ivapci_v2_1_glm":
        return IVAPCIv21GLMEstimator()
    if name == "ivapci_v2_1_radr":
        return IVAPCIv21RADREstimator()
    if name == "ivapci_v2_1_pacd_glm":
        return IVAPCIPACDTGLMEstimator()
    if name == "ivapci_v3_1_pacd":
        return IVAPCIv31PACDEncoderEstimator()
    if name == "ivapci_v3_1_radr":
        return IVAPCIv31RADREstimator()
    if name == "ivapci_v3_1_radr_theory":
        return IVAPCIv31TheoryRADREstimator()
    if name == "ivapci_gold":
        return IVAPCIGoldEstimator()
    if name == "pacdt_v3_0":
        return PACDTv30Estimator()
    raise ValueError(f"Unsupported method '{name}'.")


def _concat_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    parts = [p for key in ["X", "W", "Z"] if (p := data.get(key)) is not None]
    if not parts:
        raise ValueError("No observable proxies available to concatenate.")
    return np.concatenate(parts, axis=1)


def _latent_r2(U_true: np.ndarray, U_hat: np.ndarray) -> float:
    if U_hat is None:
        return np.nan
    U_true = np.asarray(U_true)
    U_hat = np.asarray(U_hat)
    if U_hat.ndim == 1:
        U_hat = U_hat.reshape(-1, 1)
    XtX_inv = np.linalg.pinv(U_hat.T @ U_hat)
    coefs = XtX_inv @ U_hat.T @ U_true
    preds = U_hat @ coefs
    ss_res = np.sum((U_true - preds) ** 2)
    ss_tot = np.sum((U_true - U_true.mean(axis=0)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def _normalize_scenarios(raw: List[str]) -> List[str]:
    """Normalize scenario names, supporting comma-delimited and accidental concatenation.

    Users sometimes concatenate scenario tokens when shell line breaks miss a trailing space
    (e.g., "HARD-nonlinear-strongHARD-nonlinear-extreme"). This helper attempts to recover
    such cases by greedily matching known scenario names from the registry. It also splits
    comma-delimited inputs for convenience.
    """

    available = list(list_scenarios().keys())
    normalized: List[str] = []

    for token in raw:
        # First split on commas and strip whitespace.
        parts = [p.strip() for p in token.split(",") if p.strip()]
        for part in parts:
            if part in available:
                normalized.append(part)
                continue

            # Attempt to decompose concatenated scenario names greedily by prefix.
            pos = 0
            matched_any = False
            while pos < len(part):
                match = None
                for name in sorted(available, key=len, reverse=True):
                    if part.startswith(name, pos):
                        match = name
                        break
                if match is None:
                    break
                matched_any = True
                normalized.append(match)
                pos += len(match)

            if pos != len(part):
                raise ValueError(
                    f"Unknown scenario '{part}'. Available: {available}. "
                    "Ensure scenario names are space- or comma-separated."
                )

            if matched_any:
                continue

            raise ValueError(
                f"Unknown scenario '{part}'. Available: {available}. "
                "Ensure scenario names are space- or comma-separated."
            )

    return normalized


def run_benchmark(
    scenarios: List[str],
    n_samples: int,
    repetitions: int,
    start_seed: int,
    seeds: List[int],
    methods: List[str],
    variant: str,
    results_path: str,
    summary_path: str,
) -> None:
    records = []
    for scenario in scenarios:
        seed_list = seeds or [start_seed + rep for rep in range(repetitions)]
        for seed in seed_list:
            data = simulate_scenario(scenario, n=n_samples, seed=seed, variant=variant)
            X_all = _concat_features(data)
            A = data["A"]
            Y = data["Y"]
            tau_true = data["tau"]
            for method in methods:
                est = _build_estimator(method)
                X_input = data["U"] if method == "oracle_U" else X_all
                t0 = time.time()
                est.fit(X_input, A, Y)
                ate_hat = est.estimate_ate(X_input, A, Y)
                runtime = time.time() - t0
                r2_u = np.nan
                if hasattr(est, "get_latent") and method in {
                    "ivapci_v2_1",
                    "ivapci_v2_1_glm",
                    "ivapci_v2_1_radr",
                    "ivapci_v2_1_pacd_glm",
                    "ivapci_v3_1_pacd",
                    "ivapci_v3_1_radr",
                    "ivapci_v3_1_radr_theory",
                    "ivapci_gold",
                    "pacdt_v3_0",
                }:
                    latent = est.get_latent(X_input)
                    r2_u = _latent_r2(data["U"], latent)
                abs_err = abs(ate_hat - tau_true)
                sq_err = (ate_hat - tau_true) ** 2
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "tau_true": tau_true,
                        "ate_hat": ate_hat,
                        "abs_err": abs_err,
                        "sq_err": sq_err,
                        "runtime_sec": runtime,
                        "r2_U": r2_u,
                    }
                )

    df = pd.DataFrame.from_records(records)
    df.to_csv(results_path, index=False)

    summary = (
        df.groupby(["scenario", "method"])
        .agg(
            mean_tau_true=("tau_true", "mean"),
            mean_ate_hat=("ate_hat", "mean"),
            mean_abs_err=("abs_err", "mean"),
            rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
            std_abs_err=("abs_err", "std"),
            mean_runtime=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print("Benchmark summary:")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PACD/IVAPCI simulation benchmarks.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(list_scenarios().keys()),
        help="Simulation scenarios to run.",
    )
    parser.add_argument("--n-samples", "--n", type=int, default=1000, help="Sample size per replicate.")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repeats per scenario (ignored if --seeds is provided).")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting random seed (ignored if --seeds is provided).")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of seeds to run. Overrides --start-seed/--repetitions when provided.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "naive",
            "dr_glm",
            "dr_rf",
            "oracle_U",
            "ivapci_v2_1",
            "ivapci_v2_1_glm",
            "ivapci_v2_1_radr",
            "ivapci_v2_1_pacd_glm",
            "ivapci_v3_1_pacd",
            "ivapci_v3_1_radr",
            "ivapci_v3_1_radr_theory",
            "ivapci_gold",
            "pacdt_v3_0",
        ],
        help="Causal estimators to evaluate.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full_proxies",
        help="Proxy/observability variant (full_proxies, weak_proxies, missing_Z, missing_W, partial_X).",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory to place results and summary CSVs.")
    parser.add_argument("--results-path", type=str, default=None, help="Path for the detailed results CSV (overrides --outdir).")
    parser.add_argument("--summary-path", type=str, default=None, help="Path for the aggregated summary CSV (overrides --outdir).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scenarios = _normalize_scenarios(args.scenarios)
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(args.results_path)
        if args.results_path is not None
        else (outdir / "simulation_benchmark_results.csv" if outdir else Path("simulation_benchmark_results.csv"))
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path is not None
        else (outdir / "simulation_benchmark_summary.csv" if outdir else Path("simulation_benchmark_summary.csv"))
    )
    run_benchmark(
        scenarios=args.scenarios,
        n_samples=args.n_samples,
        repetitions=args.repetitions,
        start_seed=args.start_seed,
        seeds=args.seeds or [],
        methods=args.methods,
        variant=args.variant,
        results_path=str(results_path),
        summary_path=str(summary_path),
    )


if __name__ == "__main__":
    main()
