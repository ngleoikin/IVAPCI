# IVAPCI

This repository provides the simulation, modeling, and diagnostics tooling described in `docs/pacd_benchmark_design.md`, including IVAPCI v2.1 (RF-DR + GLM-DR variants), the proxy-only IVAPCI-Gold representation, the PACD-partitioned IVAPCI-PACD-GLM variant, the PACD-regularized IVAPCI v3.1 encoder, the hierarchical IVAPCI v3.2 encoder (with optional RADR calibration), PACD-T v3.0, and supporting baselines.

## Setup

Install the Python dependencies before running any benchmarks, diagnostics, or smoke tests (NumPy is pinned to <2 for PyTorch compatibility):

```bash
pip install -r requirements.txt
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
