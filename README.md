# IVAPCI

This repository provides the simulation, modeling, and diagnostics tooling described in `docs/pacd_benchmark_design.md`, including IVAPCI v2.1 (RF-DR + GLM-DR variants), the proxy-only IVAPCI-Gold representation, the PACD-partitioned IVAPCI-PACD-GLM variant, the PACD-regularized IVAPCI v3.1 encoder, the hierarchical IVAPCI v3.2 encoder (with optional RADR calibration), PACD-T v3.0, and supporting baselines.

## Setup

Install the Python dependencies before running any benchmarks, diagnostics, or smoke tests (NumPy is pinned to <2 for PyTorch compatibility). A fully isolated setup using conda is recommended:

```bash
# Start in the repo root
cd /root/IVAPCI

# Load conda (if not already available in the shell)
source /root/miniconda3/etc/profile.d/conda.sh

# Create and activate a dedicated environment
conda create -n ivapci311 python=3.11 -y
conda activate ivapci311

# Install core requirements
pip install -r requirements.txt

# Ensure scientific stack is up to date
pip install -U numpy scipy scikit-learn pandas torch seaborn matplotlib
```

## Quick smoke test

After installing dependencies you can run a lightweight end-to-end sanity check on the simulators, baseline estimator, IVAPCI estimator, and diagnostics:

```bash
python smoke_test.py
```

The smoke test uses a tiny synthetic dataset so it completes quickly on CPU-only environments.

## Using IVAPCI v3.2 hierarchical encoders

Two v3.2 variants are available in the CLI and scripts:

- `ivapci_v3_2_hier`: hierarchical encoder with X/W/Z-specific heads, orthogonal fusion, layered adversaries, and p-adic regularization, estimated with cross-fitted DR.
- `ivapci_v3_2_hier_radr`: the same encoder with RADR-style head calibration on top of the learned causal latent.

To benchmark them, include the method names when invoking the simulation runner (the script infers `x_dim / w_dim / z_dim` from concatenated `[X|W|Z]` features):

```bash
python scripts/run_simulation_benchmark.py \
  --scenarios EASY-linear-weak HARD-nonlinear-extreme \
  --seeds 0 1 \
  --n 500 \
  --methods ivapci_v3_2_hier ivapci_v3_2_hier_radr dr_glm dr_rf \
  --outdir outputs/bench_v32
```

Diagnostics and plotting work the same way by passing the methods to `scripts/run_diagnostics_on_simulation.py` and the corresponding analysis scripts.

## Hyperparameter search (grid + Bayesian)

Two standalone scripts support in-process sweeps over IVAPCI v3.3 configurations:

- `scripts/run_hyperparam_search.py` performs grid sweeps over preset search spaces and exports per-run/per-scenario summaries plus best/top configs.
- `scripts/bayesian_hyperparam_search_ivapci.py` runs Optuna-based Bayesian search (TPE or NSGA-II) across scenarios/seeds, writing trials to an SQLite study plus CSV/JSON artifacts. Install `optuna` (and optionally `optuna-dashboard`) before use.

Example (TPE, single-objective):

```bash
python scripts/bayesian_hyperparam_search_ivapci.py \
  --mode quick \
  --scenarios EASY-linear-weak,MODERATE-nonlinear \
  --n-trials 40 \
  --epochs 100 \
  --n 500
```
