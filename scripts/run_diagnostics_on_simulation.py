"""Run diagnostics on simulated datasets as specified in the design doc."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repository root is on the import path when invoked as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pacd_diagnostics import (
    estimate_residual_risk,
    extract_confounding_subspace,
    proximal_condition_number,
    proxy_strength_score,
)
from models.baselines import DRGLMEstimator, DRRFEstimator, NaiveEstimator, OracleUEstimator
from models.ivapci_v21 import IVAPCIv21Estimator, IVAPCIv21GLMEstimator
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
    if name == "pacdt_v3_0":
        return PACDTv30Estimator()
    raise ValueError(f"Unsupported method '{name}'.")


def _concat_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    parts = [p for key in ["X", "W", "Z"] if (p := data.get(key)) is not None]
    if not parts:
        raise ValueError("No observable proxies available to concatenate.")
    return np.concatenate(parts, axis=1)


def _latent_r2(U_true: np.ndarray, U_hat: Optional[np.ndarray]) -> float:
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


def run_diagnostics(
    scenarios: List[str],
    n_samples: int,
    repetitions: int,
    start_seed: int,
    seeds: List[int],
    methods: List[str],
    variant: str,
    results_path: str,
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

            resid_metrics = estimate_residual_risk(data.get("X"), data.get("W"), data.get("Z"), A, Y)
            proxy_metrics = proxy_strength_score(data["U"], data.get("X"), data.get("W"), data.get("Z"), A, Y)
            cond_metrics = proximal_condition_number(data.get("X"), data.get("W"), data.get("Z"))

            latent_store: Dict[str, Optional[np.ndarray]] = {}
            for method in methods:
                est = _build_estimator(method)
                X_input = data["U"] if method == "oracle_U" else X_all
                t0 = time.time()
                est.fit(X_input, A, Y)
                ate_hat = est.estimate_ate(X_input, A, Y)
                runtime = time.time() - t0
                latent = est.get_latent(X_input) if hasattr(est, "get_latent") else None
                latent_store[method] = latent

                abs_err = abs(ate_hat - tau_true)
                sq_err = (ate_hat - tau_true) ** 2

                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "tau_true": tau_true,
                    "ate_hat": ate_hat,
                    "abs_err": abs_err,
                    "sq_err": sq_err,
                    "runtime_sec": runtime,
                    "proxy_score": proxy_metrics.get("proxy_score", np.nan),
                    "proxy_r2_U": proxy_metrics.get("r2_U", np.nan),
                    "proxy_r2_Y": proxy_metrics.get("r2_Y", np.nan),
                    "proxy_auc_A": proxy_metrics.get("auc_A", np.nan),
                    "resid_score": resid_metrics.get("resid_score", np.nan),
                    "resid_corr": resid_metrics.get("resid_corr", np.nan),
                    "resid_auc_A": resid_metrics.get("auc_A", np.nan),
                    "resid_r2_Y": resid_metrics.get("r2_Y", np.nan),
                    "prox_cond_score": cond_metrics.get("prox_cond_score", np.nan),
                    "prox_cond": cond_metrics.get("cond", np.nan),
                    "prox_s_min": cond_metrics.get("s_min", np.nan),
                    "prox_s_max": cond_metrics.get("s_max", np.nan),
                    "subspace_r2_ivapci": np.nan,
                    "subspace_r2_pacdt": np.nan,
                }

                if method in {"ivapci_v2_1", "ivapci_v2_1_glm"}:
                    row["subspace_r2_ivapci"] = _latent_r2(data["U"], latent)
                elif method == "pacdt_v3_0":
                    row["subspace_r2_pacdt"] = _latent_r2(data["U"], latent)

                records.append(row)

            if {"ivapci_v2_1", "pacdt_v3_0"}.issubset(latent_store.keys()):
                extract_confounding_subspace(
                    data["U"],
                    latent_store.get("ivapci_v2_1") or latent_store.get("ivapci_v2_1_glm"),
                    latent_store.get("pacdt_v3_0"),
                    scenario=scenario,
                    rep=seed,
                )

    df = pd.DataFrame.from_records(records)
    df.to_csv(results_path, index=False)
    print("Diagnostics results saved to", results_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PACD/IVAPCI simulation diagnostics.")
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
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory to place results CSVs and plots.")
    parser.add_argument("--results-path", type=str, default=None, help="Path for the diagnostics results CSV (overrides --outdir).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    results_path = (
        Path(args.results_path)
        if args.results_path is not None
        else (outdir / "simulation_diagnostics_results.csv" if outdir else Path("simulation_diagnostics_results.csv"))
    )
    run_diagnostics(
        scenarios=args.scenarios,
        n_samples=args.n_samples,
        repetitions=args.repetitions,
        start_seed=args.start_seed,
        seeds=args.seeds or [],
        methods=args.methods,
        variant=args.variant,
        results_path=str(results_path),
    )


if __name__ == "__main__":
    main()
