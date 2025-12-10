# 安装与测试指南

本文档提供在本仓库复现实验的最小安装与测试步骤，按顺序执行即可完成依赖安装并跑通核心流程。

## 1. 环境准备

- 建议使用 Python 3.10+，并在隔离环境（如 `venv` 或 conda）中操作。
- 若可用 GPU，PyTorch 会自动选择 CUDA 版本；若无 GPU，CPU 版本即可。

### 创建虚拟环境（可选但推荐）
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
python -m pip install --upgrade pip
```

## 2. 安装依赖

所有依赖已列在 `requirements.txt` 中，已固定一组互相兼容的版本（`numpy==1.26.4`、`scipy==1.11.4`、`scikit-learn==1.3.2`、`onnx==1.15.0`、`torch==2.2.2` 等），直接安装即可：
```bash
pip install -r requirements.txt
```
> 若机器上已有 `numpy>=2`、`scipy`、`scikit-learn` 等由其他版本编译的包，建议先“重置”三件套，以避免二进制兼容性错误：
> ```bash
> pip install --force-reinstall --no-cache-dir "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.3.2"
> pip install -r requirements.txt
> ```
> 若出现 `ImportError: cannot import name "DiagnosticOptions" from torch.onnx._internal.exporter`，通常是本地存在另一份不兼容的 PyTorch/ONNX 安装，可通过重新安装固定版本解决：
> ```bash
> pip install --force-reinstall --no-cache-dir "torch==2.2.2" "onnx==1.15.0"
> pip install -r requirements.txt
> ```
> 若处于内网/代理环境，请先配置镜像源；本项目不需要额外的系统级库。

## 3. 快速自检（smoke test）

运行轻量级自检脚本，验证模拟器、基线方法、IVAPCI 表示与诊断管线是否串通：
```bash
python smoke_test.py
```
预期输出：无异常报错，并打印若干 ATE 估计与诊断分数。

## 4. 基准与诊断完整流程

### 4.0 场景、变体与推荐测试矩阵

- **模拟场景（`--scenarios`）**：
  - `EASY-linear-weak`：线性、弱混杂。
  - `EASY-linear-strong`：线性、强混杂。
  - `MODERATE-nonlinear`：中等非线性。
  - `HARD-nonlinear-strong`：强非线性 + 强混杂。
- **数据量与重复**：通过 `--n`（或 `--n-samples`）控制样本量，通过 `--seeds` 或 `--start-seed + --repetitions` 控制重复次数。
- **proxy 充足度变体（`--variant`）**：
  - `full_proxies`（默认）
  - `weak_proxies` / `missing_Z` / `missing_W` / `partial_X`

> 推荐测试组合
> 1) **冒烟**：`--scenarios EASY-linear-weak --n 200 --seeds 0`，方法用 `naive ivapci_v2_1 ivapci_v2_1_glm ivapci_v2_1_np ivapci_gold` 对比 RF/GLM/GBDT DR。
> 2) **小规模回归测试**（示例命令见下文 4.1/4.2）。
> 3) **完整回归测试**：全部四个场景、`--n 1000`、`--seeds 0 1 2 3 4`，方法全开，运行时间较长但覆盖全面。

### 4.1 模拟基准（小规模示例）
```bash
python scripts/run_simulation_benchmark.py \
  --scenarios EASY-linear-weak EASY-linear-strong \
  --seeds 0 1 \
  --n 200 \
  --methods naive dr_glm dr_rf oracle_U ivapci_v2_1 ivapci_v2_1_glm ivapci_v2_1_np ivapci_gold pacdt_v3_0 \
  --outdir outputs/bench_small
```
> 如果不传 `--seeds`，脚本将使用 `--start-seed` 和 `--repetitions` 自动生成种子序列；默认输出文件为 `simulation_benchmark_results.csv` 与 `simulation_benchmark_summary.csv`，若指定 `--outdir` 会自动写入该目录。
生成的关键文件：
- `simulation_benchmark_results.csv`：每个场景/seed/方法的详细指标。
- `simulation_benchmark_summary.csv`：聚合后的误差与耗时汇总。

### 4.2 诊断分析（基于上一步输出）
```bash
 python scripts/run_diagnostics_on_simulation.py \
  --scenarios EASY-linear-weak EASY-linear-strong \
  --seeds 0 1 \
  --n 200 \
  --methods naive dr_glm dr_rf oracle_U ivapci_v2_1 ivapci_v2_1_glm ivapci_v2_1_np ivapci_gold pacdt_v3_0 \
  --outdir outputs/diag_small
```
> 同样地，如未提供 `--seeds`，会使用 `--start-seed` 和 `--repetitions`；诊断结果默认写入 `simulation_diagnostics_results.csv`，若传 `--outdir` 将放在指定目录。
主要产物：
- `simulation_diagnostics_results.csv`：残差风险、proxy 强度、近端条件数及子空间对齐等指标。
- `subspace_plots/`：混杂子空间的 PCA 可视化。

### 4.3 结果可视化
对基准汇总与诊断结果生成图表：
```bash
python scripts/analyze_simulation_results.py --results outputs/bench_small/simulation_benchmark_summary.csv --outdir outputs/bench_small/plots
python scripts/analyze_simulation_diagnostics.py --results outputs/diag_small/simulation_diagnostics_results.csv --outdir outputs/diag_small/plots
```
输出包含 MAE/RMSE 的柱状图、诊断指标与 ATE 误差的散点相关图，以及可选的 LaTeX 表格。

### 4.4 指标字段释义

**基准输出**（`simulation_benchmark_results.csv`）：

- `scenario` / `seed` / `method`：运行配置。
- `tau_true`：真实 ATE。
- `ate_hat`：模型估计的 ATE。
- `abs_err`、`sq_err`、`rmse`：绝对误差、平方误差、均方根误差。
- `runtime_sec`：单次运行耗时。
- `r2_U`（可选）：方法 latent 对真混杂的线性对齐 R²。

**诊断输出**（`simulation_diagnostics_results.csv`）：

- Proxy 信号：`proxy_score`（综合得分）、`proxy_r2_U`、`proxy_r2_Y`、`proxy_auc_A`。
- 残差风险：`resid_score = |Corr(r_A, r_Y)|`、`resid_corr`、`resid_r2_Y`、`resid_auc_A`。
- 近端条件数：`prox_cond_score`（条件数）、`prox_cond`、`prox_s_min`、`prox_s_max`。
- 子空间对齐（有真 U 时）：`subspace_r2_ivapci`、`subspace_r2_pacdt` 以及对应可视化 PNG 路径。

**读图指南**：

- MAE/RMSE 柱状图：越低越好，用于比较各方法精度。
- 诊断散点：
  - `proxy_score` 低且误差高 → proxy 信息不足。
  - `resid_score` 高且 proxy_score 高 → 不可识别风险（残差仍相关）。
  - `prox_cond_score` 极高 → 近端方程病态，估计不稳定。

## 5. 常见问题

- **安装缓慢或失败**：配置 PyPI 国内镜像或离线源后重试 `pip install -r requirements.txt`。
- **显存不足/训练慢**：在基准脚本中减小 `--n`、缩短 `--epochs_proxy` 等参数；或仅运行 `smoke_test.py` 进行快速验证。
- **依赖冲突**：先升级 pip，再在全新虚拟环境中安装；如需特定 PyTorch 版本，可在 `requirements.txt` 中固定镜像源对应的包名。
- **ModuleNotFoundError: No module named "models"**：请确保从仓库根目录运行脚本（如上示例的 `python scripts/...`）。脚本已自动将项目根目录加入 `sys.path`，无需额外设置 `PYTHONPATH`。

执行以上步骤后，即可完成项目的基础安装与功能验证。
