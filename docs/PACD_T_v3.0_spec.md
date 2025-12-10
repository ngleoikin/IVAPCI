# PACD-T v3.0 技术规范（草案）

## 0. 目标与直觉

PACD-T（p-adic Causal Decomposition – Targeted）想做的事可以一句话概括：

> **把代理变量里的“对因果效应有用的混杂子空间”和“只是噪声/扰动的部分”分开，用前者做 DML，后者丢给对抗器去吸收，外加一个 p-adic 几何约束，让“有用子空间”的结构更稳定、更接近树状/层级因果几何。**

对应到 v3.0 最小原型：

- 从 IVAPCI v2.1 的编码器输出 **两个潜变量分支**
  - (U_c)：causal / confounding 子空间（**用于估计 ATE 的那一块**）
  - (U_n)：nuisance / noise 子空间（**专门装噪声 & 代理里与因果无关的变化**）
- 加入 **adversarial nuisance loss**：  
  强迫 (U_n) 里尽量 **不含** 和 (A,Y) 相关的信息；
- 加入 **p-adic 距离正则**：  
  在 (U_c) 上施加“类 p-进”的 ultrametric 几何约束，鼓励同一因果 regime 的样本在树状结构里“成簇”，有利于小样本下的稳健泛化。
- 第二阶段用 (U_c) 做 **DR+DML（交叉拟合）** 估计 ATE。

---

## 1. 因果模型与表示分解

### 1.1 基本因果图

沿用 IVAPCI v2.1 的数据生成图：

[  
U \to (X,W,Z),\quad U \to A,\quad (U,A) \to Y.  
]

这里只是进一步假设/解释：

- 未观测混杂因子 (U) 可以拆成两部分：  
  [  
  U = (U_c, U_n),  
  ]  
  其中：
  - (U_c)：决定 **处理 A 和结局 Y 的相关结构** 的那部分（混杂子空间）；
  - (U_n)：只影响代理 (X,W,Z) 或与测量噪声相关，对 ((A,Y)) 的条件分布几乎没有贡献。

形式上可以写成（理想化）：

[  
\begin{aligned}  
(X,W,Z) &\leftarrow f_X(U_c, U_n, \varepsilon_X),\  
A &\leftarrow f_A(U_c, \varepsilon_A),\  
Y &\leftarrow f_Y(U_c, A, \varepsilon_Y),  
\end{aligned}  
]  
其中 (\varepsilon_\bullet) 为独立噪声。

**重要理念**：  
从 ATE 的角度看，我们只需要 (U_c)，**不需要完整恢复 (U)**。PACD-T 正是显式把“只要 (U_c)”编码进模型结构。

---

## 2. 模型结构：双分支 VAE + 监督头 + 对抗头

### 2.1 编码器（Encoder）

输入：  
[  
X_{\text{all}} = [X,W,Z] \in \mathbb{R}^d  
]

经过若干层 MLP，输出两个高斯后验参数：

- causal 分支：  
  [  
  q_\phi(U_c\mid X_{\text{all}}) = \mathcal N(\mu_c(X_{\text{all}}), \operatorname{diag}(\sigma_c^2(X_{\text{all}}))),  
  ]
- nuisance 分支：  
  [  
  q_\phi(U_n\mid X_{\text{all}}) = \mathcal N(\mu_n(X_{\text{all}}), \operatorname{diag}(\sigma_n^2(X_{\text{all}}))).  
  ]

采样采用 reparameterization：

[  
U_c = \mu_c + \sigma_c \odot \epsilon_c,\quad U_n = \mu_n + \sigma_n \odot \epsilon_n,\quad \epsilon_c,\epsilon_n\sim \mathcal N(0,I).  
]

> **规范建议**
> - 潜维度：建议 ( \dim(U_c) = 2 \sim 4)，( \dim(U_n) = 2 \sim 4)，总维度和 v2.1 保持同量级；
> - 激活：ReLU 或 GELU；
> - 初始化：Xavier/He；
> - **不要**在编码器里使用 BatchNorm（容易干扰 KL），可以用 LayerNorm。

### 2.2 解码器（Decoder）与监督头

#### 2.2.1 代理重构头

解码器输入 ((U_c,U_n))，输出对 (X_{\text{all}}) 的分布参数，例如高斯：

[  
p_\theta(X_{\text{all}}\mid U_c, U_n) = \mathcal N\big(\mu_X(U_c,U_n), \operatorname{diag}(\sigma_X^2)\big).  
]

对应重构 loss：

[  
\mathcal{L}_{\text{rec}} = -\mathbb E_{q_\phi}\big[\log p_\theta(X_{\text{all}}\mid U_c,U_n)\big].  
]

#### 2.2.2 处理头 (p(A\mid U_c))

只用 (U_c) 预测处理：

[  
p_\theta(A=1\mid U_c) = \sigma\big(f_A(U_c)\big),  
]

loss 取交叉熵：

[  
\mathcal{L}_A = -\mathbb E_{q_\phi}\big[ A\log p_\theta(A\mid U_c) + (1-A)\log(1-p_\theta(A\mid U_c))\big].  
]

> **注意**：
> - 这里故意不让 (U_n) 进 A 头，强化“**A 只看 causal 分支**”的结构假设。

#### 2.2.3 结果头 (p(Y\mid A,U_c))

用 ([A,U_c]) 预测 (Y)：

- 连续 (Y)：高斯回归  
  [  
  p_\theta(Y\mid A,U_c) = \mathcal N(\mu_Y(A,U_c), \sigma_Y^2),  
  \quad  
  \mathcal{L}_Y = \mathbb E_{q_\phi}\big[(Y-\mu_Y)^2\big].  
  ]
- 二元 (Y)：逻辑回归  
  类似 A 头。

---

## 3. 对抗去扰动（ADR 风格）与 p-adic 正则

### 3.1 对抗 nuisance loss（核心）

设计一个 **对抗预测器** (g_{\psi})，试图用 (U_n) 去预测 (A)（或者 (A,Y)）：

[  
p_\psi(A=1\mid U_n) = \sigma\big(g_\psi(U_n)\big).  
]

对抗目标：

- **更新对抗器参数 (\psi)**：最小化交叉熵  
  [  
  \mathcal{L}_{\text{adv_cls}} = -\mathbb E\big[ A\log p_\psi(A\mid U_n) + (1-A)\log (1-p_\psi(A\mid U_n))\big],  
  ]  
  让它尽量从 (U_n) 中把 A 学好；
- **更新编码器参数 (\phi)**：通过 Gradient Reversal Layer (GRL) 或直接乘 (-\lambda) 的方式，**最大化**该 loss，  
  等价于让 (U_n) 变得 **对 A 不可预测**。

最终在整体 loss 里表现为：

[  
\mathcal{L}_{\text{adv}} = -\mathcal{L}_{\text{adv_cls}} \quad (\text{对 encoder 来说}).  
]

> 对应 ADR 思想：
> - (U_c) = “带 confounder 的表示”，保留 (A,Y) 有用信息；
> - (U_n) = “专门吸收多余/噪声”，且对 (A) 尽量中性（不可预测）。
> - 这一块是 PACD-T 与普通半监督 VAE 的**关键区别**，**不要删**。

可以选做：再加一个 (Y) 的 adversary (p_\psi(Y\mid U_n,A))，原则同上。

### 3.2 p-adic 距离正则（几何层）

我们在 (U_c) 上希望有一种“树状/层级”的 ultrametric 结构，直觉是：

> 状态“远” ≈ 频率/阶数上的差异比较大；  
> 这与 p-进距离的“按位比较”直觉相似。

#### 3.2.1 一个可实现的 p-adic 型距离定义（规范版本）

选定一个质数 (p)（实践中可以固定为 2 或 3），以及每个维度的量化步长 (s_j>0)。对每个样本 (i) 的 (U_c^{(i)}\in\mathbb R^d)，做量化：

[  
 c_j^{(i)} = \left\lfloor \frac{U_{c,j}^{(i)}}{s_j} \right\rfloor \in \mathbb Z.  
]

对两个样本 (i,k) 的某一维差：

[  
 n_{j}^{(ik)} = c_j^{(i)} - c_j^{(k)}.  
]

定义 p-进阶数（valuation）：

[  
 v_p(n) =  
 \begin{cases}  
 +\infty, & n = 0,\  
 \text{最大整数 }r\text{ 使得 }p^r\mid n, & n\neq 0.  
 \end{cases}  
]

再定义该维上的 p-进距离：

[  
 d_{p,j}(i,k) =  
 \begin{cases}  
 0, & n_j^{(ik)} = 0,\[2mm]  
 p^{-v_p(n_j^{(ik)})}, & n_j^{(ik)}\neq 0.  
 \end{cases}  
]

整体距离取最大（∞-范）：

[  
 d_p(i,k) = \max_{j=1,\dots,d} d_{p,j}(i,k).  
]

这是一个标准的 ultrametric：  
[  
 d_p(i,k) \le \max\{d_p(i,j), d_p(j,k)\}\quad \forall i,j,k.  
]

#### 3.2.2 正则项

在 mini-batch 中随机采样若干三元组 ((i,j,k))，定义违背 ultrametric 的“惩罚”：

[  
 \Delta_{ijk} = \max\Big(0,\, d_p(i,k) - \max\{d_p(i,j), d_p(j,k)\}\Big).  
]

p-adic loss 为：

[  
 \mathcal{L}_{p\text{-adic}} = \mathbb E_{(i,j,k)}[\Delta_{ijk}].  
]

> 直觉：
> - 若 latent 空间自然呈现“层级聚簇”，则大部分三元组已经满足 ultrametric，不怎么惩罚；
> - 若空间结构乱七八糟，很多三元组违反强三角不等式，就会被驱向“树状几何”。

> **规范注意**：
> - 量化步长 (s_j) 建议根据 batch 内 (U_c) 的标准差设定，如 (s_j = 0.5 \cdot \text{std}(U_{c,j}))；
> - 不要求绝对“几何上正宗的 p-进结构”，只要保证：**同一 A-regime / 相近 Y 的样本更容易聚在相似的枝上**，就达到了 PACD-T v3.0 期望的几何效果。

---

## 4. 训练目标与优化流程

### 4.1 总损失函数

综合以上，PACD-T v3.0 的最终 loss：

[  
\begin{aligned}  
\mathcal{L}  
=&\ \mathcal{L}_{\text{rec}}\\
&- \lambda_A \mathcal{L}_A\\
&- \lambda_Y \mathcal{L}_Y\\
&- \beta_c \,\text{KL}\big(q_\phi(U_c\mid X)\,\|\,p(U_c)\big)\\
&+ \beta_n \,\text{KL}\big(q_\phi(U_n\mid X)\,\|\,p(U_n)\big)\\
&- \gamma_{\text{adv}} \mathcal{L}_{\text{adv}}\\
&- \gamma_{p} \mathcal{L}_{p\text{-adic}}.  
\end{aligned}  
]

其中：

- (p(U_c)=p(U_n)=\mathcal N(0,I))；
- 超参建议（可在仿真中固定，作为“规范版本”）：
  - (\lambda_A = 1.0), (\lambda_Y = 1.0)
  - (\beta_c = 1.0), (\beta_n = 1.0)
  - (\gamma_{\text{adv}} = 1.0)
  - (\gamma_{p} = 0.1) （p-adic 正则作为弱约束）

### 4.2 优化过程（单轮训练）

1. **采 mini-batch**：((X_{\text{all}},A,Y))。
2. **前向传播**：
  - encoder 得到 (q_\phi(U_c\mid X), q_\phi(U_n\mid X))，采样 (U_c,U_n)；
  - decoder 重构 (X_{\text{all}})，计算 (\mathcal{L}_{\text{rec}})；
  - A/Y 头计算 (\mathcal{L}_A,\mathcal{L}_Y)；
  - 对抗器用 (U_n) 预测 A，得到 (\mathcal{L}_{\text{adv_cls}})，进而得到 encoder 一侧的 (\mathcal{L}_{\text{adv}})；
  - 在 (U_c) 上构造三元组，计算 (\mathcal{L}_{p\text{-adic}})；
  - 计算 KL 项。
3. **更新对抗器 (\psi)**：
  - 只用 (\mathcal{L}_{\text{adv_cls}}) 对 (\psi) 反向传播，**梯度下降**。
4. **更新 encoder+decoder+监督头 ((\phi,\theta))**：
  - 用总 loss (\mathcal{L}) 对 ((\phi,\theta)) 反向传播，**梯度下降**。
  - 如果使用 GRL，对抗项自动实现“encoder 反向更新时乘 -1”。

> **规范建议**
> - Optimizer 统一用 AdamW (lr=1e-3, weight_decay=1e-4)；
> - 训练轮数：在仿真中固定为 500 epochs（或直到验证 loss 收敛）；
> - **始终在同一超参配置下比较 DR-GLM / DR-RF / IVAPCI / PACD-T**，避免“超参欺负”。

---

## 5. 第二阶段：基于 (U_c) 的 DML/DR 估计

训练完 PACD-T 表示之后：

1. 对每个样本 (i)，取 **posterior mean** 作为表示：  
  [  
  \hat U_{c,i} = \mu_c(X_{\text{all},i}).  
  ]
2. 在 ((A_i, Y_i, \hat U_{c,i})) 上做 **K 折交叉拟合 DML**（与 IVAPCI v2.1 完全一致）：

  对每个折 (k)：
  - 用折外数据拟合倾向得分模型（推荐：Logistic + Elastic Net）：  
    [  
    \hat e_k(u) \approx P(A=1\mid U_c = u).  
    ]
  - 用折外数据拟合结果模型（建议：线性+二次项或 RF）：  
    [  
    \hat m_k(a,u) \approx \mathbb E[Y\mid A=a, U_c=u].  
    ]
  - 对折内样本，用折外模型计算 DR 得分：  
    [  
    \varphi_i =  
    \frac{A_i - \hat e_k(\hat U_{c,i})}{\hat e_k(\hat U_{c,i})(1-\hat e_k(\hat U_{c,i}))}\big(Y_i - \hat m_k(A_i,\hat U_{c,i})\big)  
    - \hat m_k(1,\hat U_{c,i}) - \hat m_k(0,\hat U_{c,i}).  
    ]

3. ATE 估计：  
  [  
  \hat\tau_{\text{PACD-T}} = \frac{1}{n} \sum_{i=1}^n \varphi_i.  
  ]

> **必须保持**：
> - **交叉拟合**（不同 folds，折外预测）；
> - DR/AIPW 形式不变；
> - 使用同一类基学习器与 IVAPCI v2.1 对比（否则性能差异难以归因于表示本身）。

---

## 6. 与 IVAPCI v2.1 / ADR 的关系总结

- **相对 IVAPCI v2.1**：
  - v2.1：只有一个 latent (U)，通过 A/Y 头引导它朝“混杂子空间”靠近；
  - PACD-T：显式拆成 (U_c) + (U_n)，并用 adversarial loss 驱动“信息分工”，再用 p-adic loss 赋予 (U_c) 稳定的层级几何。
  - 在理论上：
    - 两者都依赖“只要学到 (g(U)) 即可”的因果充分表示理念；
    - PACD-T 进一步利用 **对抗分解 + 几何约束** 来减少无关噪声，提高有限样本下的方差表现。

- **相对 ADR（Adversarial Deconfounding Representation）**：
  - ADR 的核心：通过对抗训练，学习一个对 A “平衡”的表示，去掉观察 confounders 的可分辨性。
  - PACD-T 借用的是 **“表示分解 + 对抗让某一块不含 confounding 信息”** 这套思想，只是方向反过来：
    - ADR：让最终表示不要携带 confounder；
    - PACD-T：让 nuisance 分支吸收 confounder 无关的信息，并逼迫 **causal 分支 (U_c)** 单独承载与 ((A,Y)) 相关的因果信息。

---

## 7. 复现建议与“防止性能漂移”的约束

为了避免后面别人“改一改”就跑出完全不一样的结果，至少建议在文档里写死下面这些点：

1. **结构约束**
  - 必须有 (U_c) / (U_n) 双分支；
  - A 头 **只看 (U_c)**，不允许直接看 (U_n)；
  - 对抗器只看 (U_n) 预测 A（或 A,Y）。

2. **损失项**
  - (\mathcal{L}_{\text{rec}}, \mathcal{L}_A, \mathcal{L}_Y, \text{KL}_c, \text{KL}_n, \mathcal{L}_{\text{adv}}) 都是**必需项**；
  - (\mathcal{L}_{p\text{-adic}}) 推荐保留为弱正则（(\gamma_p \approx 0.1)），如果完全去掉，就不再叫“PACD-T”，而更接近 ADR-VAE。

3. **DML 阶段**
  - 必须使用 **交叉拟合的 DR/AIPW**；
  - 倾向模型、结果模型类型与 baseline（DR-GLM, DR-RF）尽量统一。

4. **超参**
  - 给出一套“规范配置”（你在仿真中用的那套）作为 default；
  - 在论文 / 文档里明确写明：若超参变动需要在附录中 report ATE 性能敏感性分析。

---

如果你愿意，下一步我可以把这份“PACD-T v3.0 技术规范”再压缩成一个 **真正要写在论文 appendx 的版本**：包括伪代码（Algorithm 1: Representation Learning；Algorithm 2: DML Estimation）、以及和 v2.1 的对比表，这样你以后不管换谁来写代码或复现，都有一份“官方标准”。
