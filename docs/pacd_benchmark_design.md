# PACD / IVAPCI 因果推断基准与诊断框架——需求与设计说明

> 目标：构建一套统一的 **模拟数据 + 基准测试 + 诊断指标 + 可视化** 框架，系统评估
> IVAPCI v2.1、PACD-T v3.0 与传统方法（DR-GLM / DR-RF / naive / Oracle-U）在“有/无未观测混杂 + proxy 质量不同”的场景下的表现，并给出可解释的诊断指标。

---

## 0. 项目整体结构建议

推荐项目目录结构如下（仅是建议，方便 Codex 实现）：

```text
project_root/
  data/                     # 各种外部数据（IHDP, Criteo 等）
  simulators/               # 模拟场景生成器
    __init__.py
    simulators.py
  models/                   # 各种因果估计方法封装（统一接口）
    __init__.py
    baselines.py            # naive, DR-GLM, DR-RF, Oracle-U
    ivapci_v21.py           # 已有，适配统一接口
    pacdt_v30.py            # 已有，适配统一接口
  diagnostics/              # 诊断模块
    __init__.py
    pacd_diagnostics.py     # 残差相关 / proxy 强度 / 近端条件数 / 子空间分解
  scripts/                  # 可执行脚本
    run_simulation_benchmark.py
    run_diagnostics_on_simulation.py
    analyze_simulation_results.py
    analyze_simulation_diagnostics.py
  docs/
    pacd_benchmark_design.md # 本文
```

---

## 1. 测试数据设计（模拟场景 + 真实数据）

### 1.1 模拟数据场景设计

#### 1.1.1 统一接口

定义一个统一的模拟函数，在 `simulators/simulators.py` 中：

```python
def simulate_scenario(
    scenario: str,
    n: int,
    seed: int,
    variant: str = "full_proxies",
) -> dict:
    """
    返回一个 dict，包含：

    X: (n, d_x)   已观察协变量
    W: (n, d_w)   outcome-proximal proxy（偏向 Y）
    Z: (n, d_z)   treatment-proximal proxy（偏向 A）
    A: (n,)       处理（二值 {0,1}）
    Y: (n,)       结果（实数）
    U: (n, d_u)   真混杂（只在模拟里可见）
    tau: float    真 ATE E[Y(1) - Y(0)]
    meta: dict    记录场景参数（非必须字段）

    variant 用于控制 proxy/混杂暴露的不同情况（见 1.1.3）。
    """
    ...
```

约定场景名（scenario）至少包括：

* `"EASY-linear-weak"`：线性、弱混杂
* `"EASY-linear-strong"`：线性、强混杂
* `"MODERATE-nonlinear"`：适度非线性
* `"HARD-nonlinear-strong"`：强非线性 + 强混杂
* `"HARD-nonlinear-extreme"`：极强非线性 + 弱重叠 + 异质效应（可配合 `variant="weak_proxies"` 打造“地狱模式”）
* `"HARD-nonlinear-extreme-heavy-tail"`：在 extreme 基础上使用 heavy-tail latent（t 分布），检验几何正则对尾部样本的鲁棒性。
* `"HARD-nonlinear-extreme-mixture"`：在 extreme 基础上使用 mixture-of-Gaussians latent，形成明显簇结构，观察 PACD/ultrametric 的适配性。
* `"HARD-nonlinear-extreme-low-var-proxy"`：在 extreme 基础上额外拼接极低方差 proxy：
  1）一个与混杂强相关、方差约 1e-4 的确定性通道；2）一个 2% 触发的稀有通道，用于检验预处理/正则是否会错误压缩掉关键低方差信号。
* `"HARD-nonlinear-weak-overlap"`：仅打开弱重叠（极端倾向评分），其他结构与 HARD-strong 相同。
* `"HARD-nonlinear-hetero-tau"`：仅打开异质处理效应 τ(U) 的强非线性，其余保持 HARD-strong。
* `"HARD-nonlinear-misaligned-proxies"`：仅打开 proxy 错配/稀疏信号（A 和 Y 的最佳 proxy 方向不同），用于测试表示分解的失效模式。

每个场景都需在模拟代码中**显式给出**：

* 真混杂 (U) 的维度 d_u（比如 2 或 3）
* (X, Z, W) 的构造方式（如何从 (U) 和噪声生成）
* (A) 的生成方式（逻辑回归/非线性）：
  [
  A \sim \mathrm{Bernoulli}(\sigma(g(U, X, Z)))
  ]
* (Y) 的生成方式（线性或非线性）：
  [
  Y = f(U, X, W, A) + \epsilon_Y
  ]
* 真 ATE：
  [
  \tau = \mathbb{E}[Y(1) - Y(0)]
  ]
  可以通过 closed-form 或 Monte Carlo 方式给出。

#### 1.1.2 基本场景的要求

对每个场景要规定：

* (d_u)：真混杂维度（比如 2）
* (d_x, d_z, d_w)：分别的维度（比如各 5~10）
* “混杂强度”参数 (\gamma)：控制 (U \to A, U \to Y) 的系数大小
* “非线性程度”参数：是否使用 (\sin)、(\tanh)、乘积项等

例如：

* `EASY-linear-weak`：
  (A, Y) 都是 (U, X, Z, W) 的线性函数，混杂系数适中；
* `EASY-linear-strong`：
  线性，但 (U \to A, U \to Y) 系数更大，偏差更严重；
* `MODERATE-nonlinear`：
  (A) 用 logistic + 少量非线性项，(Y) 用平滑非线性；
* `HARD-nonlinear-strong`：
  (A, Y) 中含有多项式/交互/非平滑非线性，混杂很强。
* `HARD-nonlinear-extreme`：
  在强非线性的基础上增加弱重叠（极端倾向评分）、错配/稀疏 proxy 结构和异质处理效应，可与 `variant="weak_proxies"` 组合形成更高难度。
* `HARD-nonlinear-extreme-heavy-tail`：
  在 extreme 的生成式上替换 heavy-tail latent（t 分布），极端样本更多，考验几何正则的鲁棒性。
* `HARD-nonlinear-extreme-mixture`：
  在 extreme 的生成式上使用 mixture-of-Gaussians latent，突出簇结构，匹配 PACD/p-adic 的层级几何假设。
* `HARD-nonlinear-weak-overlap`：
  仅打开弱重叠开关（极端倾向评分），其余结构与 HARD-strong 相同，用于隔离“重叠性”对 DR 方差的影响。
* `HARD-nonlinear-hetero-tau`：
  仅打开异质 τ(U) 的强非线性，其余保持 HARD-strong，专注评估“非线性效应”对 nuisance misspec 的影响。
* `HARD-nonlinear-misaligned-proxies`：
  仅打开 proxy 错配 + 稀疏信号（A 与 Y 依赖的 proxy 子空间不同），用于测试 U_c/U_n 分解在 proxy 错配下的稳健性。

#### 1.1.3 proxy & 混杂暴露的多种 variant

为了测试“proxy 丰富/不足”以及“已观察混杂是否充分”，
建议在 `variant` 维度上设计几种情况：

* `"full_proxies"`：
  X/Z/W 都比较丰富，原则上足以重构 U（接近你当前已有的模拟）。
* `"weak_proxies"`：
  降低 Z/W 与 U 的相关性（增加噪声、减少维度），模拟 proxy 不足。
* `"missing_Z"`：
  故意不返回 Z（set Z=None），模拟只观测 W 的情况。
* `"missing_W"`：
  同理，只观测 Z。
* `"partial_X"`：
  对 X 做子集选择 / 加噪，模拟已观察协变量不全。

后续诊断指标（residual risk / S_prox / condition number）在这些 variant 之间会有明显差异，可以系统地展示“不可识别风险 vs 信息不足风险”。

---

### 1.2 真实数据：IHDP / Criteo 等

#### 1.2.1 IHDP 仿真数据

IHDP 数据常见格式：有多个 replicate（你现在有 10 个 `ihdp_data_x.csv`），每个包含：

* 真实 covariates：(X)（比如 25 维）
* 处理：(A\in{0,1})
* 真实潜在结果：(Y(0), Y(1))（仿真真值）
* 观测结果：(Y = A Y(1) + (1-A) Y(0))

设计一个 loader 函数 `load_ihdp_replicate`:

```python
def load_ihdp_replicate(path: str) -> dict:
    """
    读入单个 IHDP 仿真 replicate CSV，返回：
    {
      "X": (n, d_x),
      "A": (n,),
      "Y": (n,),
      "Y0": (n,),   # 真 Y(0)
      "Y1": (n,),   # 真 Y(1)
      "tau_true": float  # E[Y1 - Y0]
    }
    """
```

**注意**：IHDP 没有真正的未观测 U，只有“人为设计的 selection bias”。
在框架里要清楚标注：IHDP 用于测试有完整协变量时的性能；
不适合用来验证“proximal proxy 理论”。

#### 1.2.2 Criteo uplift 数据

Criteo uplift（广告/优惠券 uplift）是真实数据集，没有真潜在结果，只能做**模型间相对比较**，不能测“差距真值程度”（因为没有真 ATE）。

可设计 loader：

```python
def load_criteo_uplift(path: str) -> dict:
    """
    返回：
    {
      "X": (n, d_x),
      "A": (n,),    # treatment / control
      "Y": (n,),    # binary outcome (click/purchase)
    }

    没有真 Y(0)/Y(1) 和 tau_true。
    """
```

在 benchmark 脚本中对该类数据只做：模型间 uplift 评分指标（Qini、AUUC 等），不做 tau 真值对照。

---

## 2. 方法封装与统一接口

你已有 IVAPCI v2.1 和 PACD-T v3.0 的实现，这里只需要定义统一接口
让 benchmark 与 diagnostics 代码**不关心内部结构**。

### 2.1 统一接口签名

在 `models/__init__.py` 或单独文件中规定：

```python
class BaseCausalEstimator:
    def fit(self, X_all, A, Y):
        """
        X_all: np.ndarray, shape (n, d_all)
               由数据方自行决定拼接哪些特征（X, W, Z）
        A:     (n,)
        Y:     (n,)
        """

    def estimate_ate(self, X_all, A, Y) -> float:
        """返回整个样本上的 ATE 估计值"""

    def get_latent(self, X_all) -> np.ndarray:
        """
        对于有潜表示的模型（IVAPCI / PACD-T），返回 U_hat (n, d_u_hat)。
        对于基线模型，可返回 None 或用 X_all 代替。
        """
```

然后：

* `IVAPCIv21Estimator(BaseCausalEstimator)`：对接你现有 IVAPCI v2.1 代码；
* `PACDTv30Estimator(BaseCausalEstimator)`：对接 PACD-T v3.0；
* `DRGLMEstimator(BaseCausalEstimator)`：
* `DRRFEstimator(BaseCausalEstimator)`：
* `NaiveEstimator(BaseCausalEstimator)`：
* `OracleUEstimator(BaseCausalEstimator)`：模拟中可直接使用真实 U 来做 DR。

benchmark 与 diagnostics 只调用统一接口，不涉及内部实现。

---

## 3. Benchmark 脚本与报告设计

### 3.1 `run_simulation_benchmark.py`

功能：
在多个场景 & 多个随机种子下，跑所有方法，输出性能汇总 CSV + 基本表格。

#### 3.1.1 输入

* 场景列表：`["EASY-linear-weak", "EASY-linear-strong", "MODERATE-nonlinear", "HARD-nonlinear-strong"]`
  扩展选项：`"HARD-nonlinear-extreme"` 及其 latent 变体（heavy-tail / mixture），
  以及单因素拆分场景：`"HARD-nonlinear-weak-overlap"`, `"HARD-nonlinear-hetero-tau"`,
  `"HARD-nonlinear-misaligned-proxies"`（可组合 `variant="weak_proxies"` 形成更极端设置）。
* 每个场景的样本量、重复次数、起始 seed
* 使用的 methods 列表：`["naive", "dr_glm", "dr_rf", "oracle_U", "ivapci_v2_1", "pacdt_v3_0"]`
* 是否保存中间 latent 可视化（可选）

#### 3.1.2 输出

生成一个 CSV，例如 `simulation_benchmark_results.csv`，列包括：

* `scenario`
* `seed`
* `method`
* `tau_true`
* `ate_hat`
* `abs_err = |ate_hat - tau_true|`
* `sq_err = (ate_hat - tau_true)^2`
* `runtime_sec`
* （可选）`r2_U`：对有真 U 的方法，用线性对齐的 R²

再生成一个 summary CSV：`simulation_benchmark_summary.csv`：

* 聚合字段：`scenario`, `method`
* 统计：

  * `mean_tau_true`
  * `mean_ate_hat`
  * `mean_abs_err`
  * `rmse`
  * `std_abs_err`
  * `mean_runtime`

并在终端打印漂亮的表格（你目前已有类似输出，可以保持风格）。

### 3.2 `run_diagnostics_on_simulation.py`

在 benchmark 的基础上再加：

* 诊断指标：

  * 残差相关 (R_{\text{res}})
  * proxy 信号强度 (S_{\text{prox}})
  * 近端条件数 (\kappa)
* 混杂子空间可视化（真 U vs U_hat）

#### 3.2.1 需要调用的诊断函数（下一节定义）

* `estimate_residual_risk(X, W, Z, A, Y, ...)`
* `proxy_strength_score(U_true, X, W, Z, A, Y, ...)`
* `proximal_condition_number(X, W, Z, ...)`
* `extract_confounding_subspace(U_true, U_hat_ivapci, U_hat_pacdt, scenario, rep, outdir)`

#### 3.2.2 结果 CSV

例如 `simulation_diagnostics_results.csv`，每行对应一个 `(scenario, seed, method)`：

* 基本字段：同 benchmark（scenario, seed, method, tau_true, ate_hat, abs_err, sq_err, runtime_sec）
* 诊断字段（不依赖方法的）：

  * `proxy_score`
  * `proxy_r2_U`
  * `proxy_r2_Y`
  * `proxy_auc_A`
  * `resid_score`（|Corr(resid_A, resid_Y)|）
  * `resid_corr`
  * `resid_auc_A`
  * `resid_r2_Y`
  * `prox_cond_score` (=cond)
  * `prox_cond`
  * `prox_s_min`
  * `prox_s_max`
* 子空间对齐（依赖方法和有真 U 的场景）：

  * `subspace_r2_ivapci`（平均对齐 R²）
  * `subspace_r2_pacdt`

混杂子空间的图（PCA 投影）单独以 PNG 保存在 `subspace_plots/` 下。

### 3.3 指标释义与解读指南（基准 + 诊断）

为便于对齐代码实现与论文叙述，下面给出基准输出和诊断输出中各字段的“公式级”定义及建议解读方式。

#### 3.3.1 基准指标（`simulation_benchmark_results.csv`）

- `ate_hat`：方法输出的 ATE 估计。若方法内置交叉拟合，则为折外影响函数的均值；否则为全样本估计。
- `tau_true`：仿真中的真 ATE（闭式或 Monte Carlo 计算）。
- `abs_err = |ate_hat - tau_true|`：绝对偏差。越小越好。
- `sq_err = (ate_hat - tau_true)^2`，`rmse = sqrt(mean(sq_err))`：平方误差与均方根误差；RMSE 聚合跨种子误差波动。
- `runtime_sec`：方法一次运行的耗时（不含前置依赖安装）。
- `r2_U`（可选）：若有真 U，则将方法 latent 与真 U 线性对齐后得到的平均 R²，反映表示是否捕捉到混杂子空间。

#### 3.3.2 诊断指标（`simulation_diagnostics_results.csv`）

**Proxy 信号（信息充足性）**

- `proxy_r2_U`：cross-fitting 下，随机森林回归预测真 U 的 R²，衡量 proxy 对混杂的可重构度。
- `proxy_r2_Y`：同上，对 Y 的 R²；高说明 proxy 可直接解释结局。
- `proxy_auc_A`：随机森林分类预测 A 的 AUC；高说明 proxy 对处理决策有预测力。
- `proxy_score = (proxy_r2_U + proxy_r2_Y + proxy_auc_A) / 3`：归一化综合分数，越高表示信息越充分。

**残差相关（不可识别风险）**

- 通过 cross-fitting 得到
  - 处理残差：`r_A = A - e_hat(V)`，其中 `e_hat` 为代理变量 V→A 的预测；
  - 结局残差：`r_Y = Y - m_hat(V)`，其中 `m_hat` 为 V→Y 的预测（可含 A）。
- `resid_corr = Corr(r_A, r_Y)`；`resid_score = |resid_corr|`：残差仍相关说明存在未被 V 捕获的结构（可能是未观测混杂或模型欠拟合）。
- `resid_r2_Y`、`resid_auc_A`：残差到 Y/A 的可预测度（越低越好），辅助判断是否纯噪声。

**近端条件数（稳定性）**

- 构造 M=[Z,W]（或退化为 X），做列标准化；奇异值为 `s_max ≥ … ≥ s_min`。
- `prox_cond = s_max / s_min`，`prox_cond_score` 同名字段；`prox_s_min`、`prox_s_max` 便于排查数值稳定性。
- 条件数过大（>1e3 量级）通常意味着病态识别，对近端类方法尤为不利。

**子空间对齐（可解释性）**

- `subspace_r2_ivapci` / `subspace_r2_pacdt`：将方法 latent 线性映射到真 U 的 R²，跨维度平均；仅在真 U 可见的仿真中计算。
- 配套 `subspace_plots/` 的 PCA 图可视化真/估计混杂子空间的几何关系。

#### 3.3.3 指标组合的诊断思路

- `proxy_score` 低 + 误差高 → 典型“信息不足”情形；补充 proxy 或收集更多协变量可能是唯一解。
- `proxy_score` 高但 `resid_score` 高 → “不可识别风险”，说明剩余结构被 proxy 未解释；需检查模型错配或潜在未观测混杂。
- `prox_cond_score` 极大 → 近端方程病态，ATE 对噪声/估计误差高度敏感；可以尝试正则化或改用更稳健的表示/方法。

---

## 4. 诊断模块 pacd_diagnostics.py 的需求

这一块就是你列出的三类指标 + 混杂子空间分解。
我们不纠结当前 bug 的实现细节，而是给 Codex 一个清晰、稳定的接口和逻辑。

### 4.1 残差相关 (R_{\text{res}})：不可识别风险

函数原型：

```python
def estimate_residual_risk(
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    A: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    返回：
    {
      "resid_score": float,   # |Corr(r_A, r_Y)|
      "resid_corr":  float,   # Corr(r_A, r_Y)
      "auc_A":       float,   # proxies -> A 的 cross-fitted AUC
      "r2_Y":        float,   # proxies -> Y 的 cross-fitted R^2
    }
    """
```

**逻辑：**

1. 组装 (V = [X, W, Z])（按列拼接，允许某些为 None）。
2. 用 RandomForestClassifier 拟合 (e(V) = P(A=1|V))，cross-fitting：

   * 得到 ( \hat{e}_i = \hat{P}(A_i=1 | V_i) )。
3. 用 RandomForestRegressor 拟合 ( m(V,A) \approx \mathbb{E}[Y|V,A] ) 或简化为 (m(V))，cross-fitting：

   * 得到 ( \hat{m}_i )。
4. 残差：

   * (r_A = A - \hat{e})
   * (r_Y = Y - \hat{m})
5. 指标：

   * (R_{\text{res}} = | \text{Corr}(r_A, r_Y) |)
   * 同时记录 Corr 本身（带符号）

**解释：**

* (R_{\text{res}}) 越大，说明在 “把 V 中的可预测部分吃掉之后”，
  A 与 Y 的残差仍然强相关 → 要么模型拟合不足，要么存在 V 无法解释的结构（未观测混杂 / 结构不可识别）。
* 实际报告中可以给出建议：

  * 若 proxy 强度高但 (R_{\text{res}}) 仍很大 → 高不可识别风险。

---

### 4.2 proxy 信号强度 (S_{\text{prox}}) + 近端条件数：信息不足风险

#### 4.2.1 proxy_strength_score：proxy 信号强度

函数原型：

```python
def proxy_strength_score(
    U_true: np.ndarray,
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    A: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    返回：
    {
      "proxy_score": float,   # 综合得分 [0,1]
      "r2_U":        float,   # proxies -> U_true 的 R^2
      "r2_Y":        float,   # proxies -> Y 的 R^2
      "auc_A":       float,   # proxies -> A 的 AUC
    }
    """
```

**逻辑：**

1. 拼接 proxy 特征 (X_{\text{proxy}} = [X, W, Z])。
2. 用 RF 回归做 cross-fitting：

   * (X_{\text{proxy}} \to U_{\text{true}})，得到 (R^2_U)。
   * (X_{\text{proxy}} \to Y)，得到 (R^2_Y)。
3. 用 RF 分类做 cross-fitting：

   * (X_{\text{proxy}} \to A)，得到 AUC。
4. 归一化（夹在 [0,1]）后取平均：
   [
   S_{\text{prox}} = \frac{R^2_U + R^2_Y + \text{AUC}_A}{3}
   ]
5. 返回所有指标。

**解释：**

* (S_{\text{prox}}) 大：说明现有 X/Z/W 对 U、A、Y 的预测力都不错，
  至少在“相关性信号”层面这些 proxy 很有信息；
* (S_{\text{prox}}) 小：说明 proxy 几乎提供不了关于 U/A/Y 的信息，
  即使有未观测混杂，现有变量也很难重构它 → 强烈的信息不足风险。

#### 4.2.2 proximal_condition_number：近端条件数

函数原型：

```python
def proximal_condition_number(
    X: Optional[np.ndarray],
    W: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    eps: float = 1e-8,
) -> dict:
    """
    返回：
    {
      "prox_cond_score": float,  # 条件数本身
      "cond":            float,
      "s_min":           float,
      "s_max":           float,
    }
    """
```

**逻辑（线性近似）：**

1. 优先用 Z/W 构造矩阵：

   * 若两者都有：(M = [Z,W])
   * 若只有一个：用那个；
   * 若都没有，则退化为用 X。
2. 对 M 做列向量标准化（z-score）。
3. 计算协方差矩阵 (\Sigma_M)，求其奇异值 (s_1 \ge \dots \ge s_r)。
4. 过滤掉过小的噪声奇异值（`s > eps`），得到 (s_{\min}, s_{\max})。
5. 条件数 (\kappa = s_{\max} / s_{\min})，作为 `prox_cond_score` 返回。

**解释：**

* 条件数大 → 特征空间高度共线/病态，意味着“近端方程”在数值上极不稳定，
  轻微噪声会引起大幅 ATE 估计波动 → 信息不足 + 识别稳定性差。
* 条件数适中 → proxy 空间相对稳定，更有希望实现可靠的近端识别。

---

### 4.3 混杂子空间提取：线性 probe + SVD / CCA

在 `pacd_diagnostics.py` 中提供：

```python
def extract_confounding_subspace(
    U_true: np.ndarray,
    U_hat_ivapci: np.ndarray,
    U_hat_pacdt: np.ndarray,
    scenario: str,
    rep: int,
    outdir: str = "subspace_plots",
) -> dict:
    """
    输入：
      U_true        : (n, d_u)
      U_hat_ivapci  : (n, d_k1)
      U_hat_pacdt   : (n, d_k2)

    逻辑：
      1. 对每个 true 维度 j：
         - 用线性回归 U_hat -> U_true[:, j]，得到 R^2_j（对齐程度）
      2. 计算 IVAPCI / PACD-T 的 mean R^2（对齐质量）
      3. 用 PCA 把 U_true 投影到 2D，并用同一个 PCA 变换 U_hat_*，
         画出三张散点图（True / IVAPCI / PACD-T），保存 PNG。
      4. 可选：进一步做线性 probe + SVD 提取 U1,U2,… 的方向。

    返回：
    {
      "ivapci": {
        "mean_best_r2": float,      # 对真 U 的线性对齐 R^2 平均
        "all_r2":       np.ndarray, # shape (d_u,)
        "plot_path":    str,        # PCA 可视化路径
      },
      "pacdt": {
        "mean_best_r2": float,
        "all_r2":       np.ndarray,
        "plot_path":    str,
      },
    }
    """
```

你如果还想继续深入做“U1, U2, … 的解释”，可以在这个函数里追加：

1. 对 A & Y 分别做线性 probe：

   * (A \sim U_{\hat{*}}) 得到 (\beta^A)
   * (Y \sim U_{\hat{*}}, A) 得到 (\beta^Y)
2. 形成矩阵 (B = [\beta^A, \beta^Y])，做 SVD，得到 top-2 canonical directions。
3. 对每一个方向，计算与原始 feature 的相关性，输出 top-k 原始变量列表。

这个部分可以写成子函数 `probe_and_decompose(U_hat, X, A, Y, ...)`，可选实现。

---

## 5. 分析与可视化脚本

### 5.1 `analyze_simulation_results.py`

* 读入 `simulation_benchmark_summary.csv`；
* 按场景展示各方法的：

  * mean_abs_err / RMSE 条形图；
  * 对比曲线图；
* 可选：表格 LaTeX 导出，用于论文。

### 5.2 `analyze_simulation_diagnostics.py`

* 读入 `simulation_diagnostics_results.csv`；

* 计算：

  * 诊断指标 vs ATE 误差的 Spearman 相关：

    * (S_{\text{prox}}) vs `abs_err`
    * (R_{\text{res}}) vs `abs_err`
    * `prox_cond_score` vs `abs_err`

* 画散点图：

  1. x = proxy_score, y = abs_err（按 method 分颜色）
  2. x = resid_score, y = abs_err
  3. x = prox_cond_score, y = abs_err

* 打印结果：

  ```text
  Proxy strength vs ATE abs error (Spearman)
    [method=naive] ρ = ...
    [method=ivapci_v2_1] ρ = ...
    ...
  ```

这些基本上就是你现在已经跑出来的图（`proxy_score_vs_abs_err.png`, `resid_risk_vs_abs_err.png`, `prox_cond_vs_abs_err.png`），
但用更加稳定、统一的实现方式重新组织一遍。

---

## 6. 层次化使用方式（给未来论文 & 实战用）

1. **Step 1：基础基准**
   只跑 `run_simulation_benchmark.py`，得到 ATE 偏差 / RMSE 表格，
   展示 IVAPCI / PACD-T 在各种场景优于/劣于 DR-GLM / DR-RF / naive。

2. **Step 2：风险诊断**
   再跑 `run_diagnostics_on_simulation.py` + `analyze_simulation_diagnostics.py`，
   把每个 replicate 上的 (S_prox, R_res, prox_cond) 与 ATE 误差关联起来，
   形成“红灯指标”图：

   * 当 S_prox 很低时，所有方法 ATE 误差都大 → 信息不足风险；
   * 当 S_prox 高但 R_res 很大时 → 不可识别风险；
   * 当 prox_cond 数很大时 → 识别高度不稳定（对 noise 极度敏感）。

3. **Step 3：混杂子空间解释**
   在模拟场景中，通过 `extract_confounding_subspace`
   对比真 U 与 U_hat 的对齐情况（R² + PCA 可视化），
   并进一步提取 U1/U2… 的“语义标签”，
   把“去噪得到更好 ATE”的故事用“可视化 + 文本解释”的方式固定下来。

---

如果你愿意，可以把这整份设计直接丢给 Codex，让它：

* **先实现 simulators + diagnostics 四个函数**；
* 再实现两个脚本：`run_simulation_benchmark.py` 和 `run_diagnostics_on_simulation.py`；
* 最后再补上 `analyze_simulation_*` 的 plotting 部分。

这样整个系统就会比我们迭代改 bug 的方式稳健很多，也更容易复用到论文里。
