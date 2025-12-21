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

两种脚本都“进程内”创建并训练 v3.3 estimator，不依赖 benchmark CLI，适合快速迭代超参。二者共同特性：
- 自动解析 estimator 模块（默认 `models.ivapci_v33_theory`，可用 `--estimator-module` 指定）
- 自动查找场景生成器（默认在 `simulation_configs/simulation_scenarios/simulators` 下的 `generate_scenario`/`simulate_scenario` 等函数，必要时用 `--scenario-module`/`--scenario-fn` 指定）
- 场景列表用逗号分隔，或用引号包住的空格分隔；`--scenarios all` 依赖模块内的 `list_scenarios()`/`AVAILABLE_SCENARIOS`
- 默认 n=500、种子列表可用 `--seeds "0,1"` 或 `--seeds "0 1"` 传入
- 输出均写到 `--output-dir`（缺省按时间戳自动创建）

### 1) Grid search – `scripts/run_hyperparam_search.py`

**什么时候用？** 快速枚举一小批固定超参网格，生成“最佳/前5”配置供后续精修。

**核心特性**
- 预置 `quick` / `balanced` / `focused_w` 三套搜索空间（可按需扩展）
- 按场景×种子×重复运行，导出分场景和分配置指标
- 自动断点续跑：`checkpoint.json`
- 输出：`results.csv`（按配置汇总）、`results_by_scenario.csv`、`results_runs.csv`、`best_config.json`、`top5_configs.json`

**聚合规则**
- 同一场景同一 base seed 的 repeats 先取 median（抗偶发训练抖动）
- 同一场景的多个 base seed 取 mean/median 后再跨场景等权平均
- 误差指标包含绝对误差与相对误差（相对误差使用 `|tau_hat-tau| / (|tau|+ate_scale)`，ate_scale 由脚本自动推断）

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
- `--scenarios`：用 **逗号** 分隔（或用引号包裹的空格分隔），否则 shell 会把场景名当独立参数

运行结束后，可直接查看 `outputs/grid_quick/best_config.json` 或用生成的 `results.csv` 做二次分析。

### 2) 贝叶斯搜索 – `scripts/bayesian_hyperparam_search_ivapci.py`

**什么时候用？** 训练成本高、参数多且连续，网格不经济时；希望自动探索到更优超参组合。

**依赖**：`pip install optuna`（可选：`optuna-dashboard plotly kaleido` 用于可视化）

**核心特性**
- Optuna TPE（单目标）或 NSGA-II（帕累托）采样，仓库根目录自动加入 `sys.path`
- 单目标：乘积式目标 `(1+ATE_rel_err)*(1+2·W_violation)*(1+2·Z_violation)*(1+ESS_violation)`，只有越界才罚（hinge）
- 多目标：`--objective-mode pareto` 或 `--multiobjective` 直接最小化 `(ATE_rel_err, W_violation, Z_violation, ESS_violation)`
- 阈值与可行域可分离：`--w-auc-hinge/--w-auc-feasible`，`--z-r2-hinge/--z-r2-feasible`，`--ess-ratio-target`；ATE 归一化可选 `--ate-scale {y_std,y_mad,tau_abs}`
- 稳健聚合：同一场景同一 base seed 重复取 median，再跨 seed/场景取平均；每个 run 记录 `tau_hat/tau_true/rel_abs_err/w_auc_eff/z_r2/ess_ratio/train_time`
- 输出：单目标 `best_params.json`，多目标 `pareto_best_trials.json` + `pareto_feasible.json` + `recommended_params.json`（按可行域优先选择），同时写 `trials.csv`/`optuna_study.db`

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

- **模式/目标**：`--objective-mode {pareto,single}`（默认 pareto）；`--mode {quick,balanced,focused_w}` 选择采样空间
- **阈值/可行域**：`--w-auc-hinge`/`--w-auc-feasible`，`--z-r2-hinge`/`--z-r2-feasible`，`--ess-ratio-target` 控制惩罚与可行筛选；`--ate-scale {y_std,y_mad,tau_abs}` 控制 ATE 归一化
- **搜索控制**：`--sampler {tpe,nsga2}`，`--pruner {median,hyperband,none}`（帕累托自动禁用剪枝），`--n-trials / --n-jobs / --timeout-hours`
- **收敛/验证**：`--plateau-window/--plateau-rel-impr` 可自动检测平台期提前停止；`--validate-best` 在独立 seeds 上复核推荐超参（`--val-repeats/--val-seed-offset/--val-scenarios` 可调）
- **数据/模型**：`--estimator`、`--scenario-module`、`--scenario-fn`、`--scenarios`（用逗号或带引号的空格分隔）
- **其他**：`--agg` 设定重复层聚合（默认 median），`--output-dir`/`--study-name` 控制输出

**结果查看**
- 单目标：`best_params.json` 给出最佳 trial；`trials.csv` 可用 pandas/Excel 分析；`optuna-dashboard sqlite:///outputs/optuna_quick/optuna_study.db` 可实时查看进度
- 多目标：查看帕累托前沿 `pareto_best_trials.json`，可行集 `pareto_feasible.json`，最终推荐解 `recommended_params.json`；Top10 文件命名为 `pareto_top10_by_ate_err.json`

> 提示：两种搜索方式都在进程内调用 estimator（不依赖 benchmark CLI），便于对 config 做细粒度覆盖；若需在完整 benchmark 流程中复现，可将找到的最佳超参写回配置后再运行 `scripts/run_simulation_benchmark.py`。
