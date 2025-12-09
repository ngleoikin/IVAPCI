# IVAPCI

This repository provides the simulation, modeling, and diagnostics tooling described in `docs/pacd_benchmark_design.md`, including IVAPCI v2.1, PACD-T v3.0, and supporting baselines.

## Setup

Install the Python dependencies before running any benchmarks, diagnostics, or smoke tests:

```bash
pip install -r requirements.txt
```

## Quick smoke test

After installing dependencies you can run a lightweight end-to-end sanity check on the simulators, baseline estimator, IVAPCI estimator, and diagnostics:

```bash
python smoke_test.py
```

The smoke test uses a tiny synthetic dataset so it completes quickly on CPU-only environments.
