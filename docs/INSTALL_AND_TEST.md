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

所有依赖已列在 `requirements.txt` 中，直接安装（已固定 `numpy<2` 以兼容当前 PyTorch 版本）：
```bash
pip install -r requirements.txt
```
> 若之前安装过 `numpy>=2` 导致 PyTorch 加载失败，请先运行 `pip install "numpy<2" --force-reinstall` 再执行上述命令。
> 若处于内网/代理环境，请先配置镜像源；本项目不需要额外的系统级库。

## 3. 快速自检（smoke test）

运行轻量级自检脚本，验证模拟器、基线方法、IVAPCI 表示与诊断管线是否串通：
```bash
python smoke_test.py
```
预期输出：无异常报错，并打印若干 ATE 估计与诊断分数。

## 4. 基准与诊断完整流程

### 4.1 模拟基准（小规模示例）
```bash
python scripts/run_simulation_benchmark.py \
  --scenarios EASY-linear-weak EASY-linear-strong \
  --seeds 0 1 \
  --n 200 \
  --methods naive dr_glm dr_rf oracle_U ivapci_v2_1 pacdt_v3_0 \
  --outdir outputs/bench_small
```
生成的关键文件：
- `simulation_benchmark_results.csv`：每个场景/seed/方法的详细指标。
- `simulation_benchmark_summary.csv`：聚合后的误差与耗时汇总。

### 4.2 诊断分析（基于上一步输出）
```bash
python scripts/run_diagnostics_on_simulation.py \
  --scenarios EASY-linear-weak EASY-linear-strong \
  --seeds 0 1 \
  --n 200 \
  --methods naive dr_glm dr_rf oracle_U ivapci_v2_1 pacdt_v3_0 \
  --outdir outputs/diag_small
```
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

## 5. 常见问题

- **安装缓慢或失败**：配置 PyPI 国内镜像或离线源后重试 `pip install -r requirements.txt`。
- **显存不足/训练慢**：在基准脚本中减小 `--n`、缩短 `--epochs_proxy` 等参数；或仅运行 `smoke_test.py` 进行快速验证。
- **依赖冲突**：先升级 pip，再在全新虚拟环境中安装；如需特定 PyTorch 版本，可在 `requirements.txt` 中固定镜像源对应的包名。

执行以上步骤后，即可完成项目的基础安装与功能验证。
