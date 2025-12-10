# IVAPCI

This repository provides the simulation, modeling, and diagnostics tooling described in `docs/pacd_benchmark_design.md`, including IVAPCI v2.1 (RF-DR + GLM-DR variants), the proxy-only IVAPCI-Gold representation, the PACD-partitioned IVAPCI-PACD-GLM variant, PACD-T v3.0, and supporting baselines.

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
