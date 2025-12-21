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

Two standalone scripts let you sweep IVAPCI v3.3 configurations without touching the benchmark CLI. Both scripts auto-resolve the estimator module (preferring the latest v3.3 theory build) and scenario generator; you can override either via CLI flags.

### 1) Grid search – `scripts/run_hyperparam_search.py`

**什么时候用？** 快速枚举一小批固定超参网格，生成“最佳/前5”配置供后续精修。

**核心特性**
- 预置 `quick` / `balanced` / `focused_w` 三套搜索空间（可按需扩展）
- 按场景×种子×重复运行，导出分场景和分配置指标
- 自动断点续跑：`checkpoint.json`
- 输出：`results.csv`（按配置汇总）、`results_by_scenario.csv`、`results_runs.csv`、`best_config.json`、`top5_configs.json`

**最小示例**（3 种参数网格、2 个场景、每个种子重复 3 次）：
```bash
python scripts/run_hyperparam_search.py \
  --mode quick \
  --scenarios EASY-linear-weak,MODERATE-nonlinear \
  --seeds 0,1 \
  --n-repeats 3 \
  --epochs 100 \
  --n 500 \
  --output-dir outputs/grid_quick
```

**常用开关**
- `--mode {quick,balanced,focused_w}`：选择预置搜索空间
- `--estimator {hier,hier_radr}`：选择 IVAPCI 或 RADR 变体
- `--scenario-module / --scenario-fn`：自定义数据生成器
- `--checkpoint-every 5`：每跑完 N 个配置写入断点

运行结束后，可直接查看 `outputs/grid_quick/best_config.json` 或用生成的 `results.csv` 做二次分析。

### 2) 贝叶斯搜索 – `scripts/bayesian_hyperparam_search_ivapci.py`

**什么时候用？** 训练成本高、参数多且连续，网格不经济时；希望自动探索到更优超参组合。

**依赖**：`pip install optuna`（可选：`optuna-dashboard plotly kaleido` 用于可视化）

**核心特性**
- Optuna TPE（默认）或 NSGA-II（多目标）采样
- 单目标：综合 `RMSE + |W_AUC-0.5| + Z泄露 + ESS` 罚项（可按需改权重）
- 多目标：`--multiobjective` 优化 `(RMSE, |W_AUC-0.5|, Z泄露)` 帕累托前沿
- 按场景×种子×重复评估，每个 trial 支持剪枝（pruning）
- 输出：`trials.csv`、`optuna_study.db`、`best_params.json`（单目标）或帕累托 JSON（多目标）

**最小示例（单目标 TPE）**
```bash
python scripts/bayesian_hyperparam_search_ivapci.py \
  --mode quick \
  --scenarios EASY-linear-weak,MODERATE-nonlinear \
  --n-trials 40 \
  --epochs 100 \
  --n 500 \
  --output-dir outputs/optuna_quick
```

**常用开关**
- `--mode {quick,balanced,focused_w}`：选择采样空间范围
- `--multiobjective`：启用帕累托优化（NSGA-II 更合适）
- `--sampler {tpe,nsga2}` / `--pruner {median,hyperband,none}`：采样与剪枝策略
- `--n-trials / --n-jobs / --timeout-hours`：控制搜索规模与并行
- 同样支持 `--estimator`、`--scenario-module`、`--scenario-fn` 覆盖

**结果查看**
- 单目标：`best_params.json` 给出最佳 trial；`trials.csv` 可用 pandas/Excel 分析；`optuna-dashboard sqlite:///outputs/optuna_quick/optuna_study.db` 可实时查看进度
- 多目标：查看 `pareto_best_trials.json` 或 `pareto_top10_by_rmse.json`

> 提示：两种搜索方式都在进程内调用 estimator（不依赖 benchmark CLI），便于对 config 做细粒度覆盖；若需在完整 benchmark 流程中复现，可将找到的最佳超参写回配置后再运行 `scripts/run_simulation_benchmark.py`。
