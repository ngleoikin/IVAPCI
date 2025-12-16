# IVAPCI v2.1 技术规范文档

> _Invariant Proxy Causal Inference (IVAPCI) v2.1 – β-VAE-XWZ-AY 规范版_

本文档用于**规范化** IVAPCI v2.1 的理论假设、模型结构、训练流程与下游因果估计步骤，以保证后续复现与扩展工作在同一基准上进行，避免由于实现细节漂移导致性能下降或结果难以对比。

---

## 0. 版本信息

- **算法名称：** IVAPCI v2.1（β-VAE-XWZ-AY）
- **核心思想：**  
  使用带有处理 (A) 和结果 (Y) 监督头的 β-VAE，从代理变量 ((X,W,Z)) 中抽取与因果效应相关的**混杂子空间表示** (\hat U)，作为下游 DR / DML 估计的调整变量。
- **架构特征：**
  - 标准 VAE（高斯先验 + 高斯后验 + MSE 重构）
  - Encoder 仅使用代理变量 ((X,W,Z))
  - Decoder 重构 ((X,W,Z))，并有两个 supervision head：
    - (p(A \mid U))
    - (p(Y \mid U, A))

---

## 1. 问题设定与因果图假设

### 1.1 变量定义

- **代理变量（proxies）：**
  - (X \in \mathbb R^{d_X})
  - (W \in \mathbb R^{d_W})
  - (Z \in \mathbb R^{d_Z})
  - 记 (X_{\text{all}} = [X,W,Z] \in \mathbb R^{d_{\text{all}}})，其中 (d_{\text{all}} = d_X + d_W + d_Z)。
- **处理：** (A \in \{0,1\})（可扩展到多值）
- **结果：** (Y \in \mathbb R)（可扩展到二元 / 计数）
- **未观测混杂因子：** (U \in \mathbb R^{d_U})
- **目标：** 从观测数据 ((X,W,Z,A,Y)) 构建潜在表示 (\hat U)，用于估计平均处理效应（ATE）：
  [  
  \tau = \mathbb E[Y(1) - Y(0)].  
  ]

### 1.2 因果图假设（固定）

IVAPCI v2.1 的标准因果图为：

[  
U \rightarrow (X,W,Z),\quad U \rightarrow A,\quad (U, A) \rightarrow Y.  
]

假设：

- (G1) 真实数据生成过程服从上述图结构；
- (G2) ((X,W,Z)) 作为 (U) 的代理，包含可用于推断 (U) 的信息；
- (G3) 条件于真正的 (U)，处理分配满足  
  [  
  Y(a) \perp A \mid U \quad (\text{强可忽略性}).  
  ]

### 1.3 目标表示的因果含义

IVAPCI v2.1 并不尝试还原整个 (U)，而是希望学到一个函数

[  
\hat U = g(U)  
]

使得对因果推断而言，(\hat U) 与 (U) **在因果充分性的意义上等价**：

[  
Y(a) \perp A \mid \hat U,\quad  
\forall a\in\{0,1\}.  
]

这可理解为：(\hat U) 近似捕捉了 (U) 中**同时影响 (A,Y) 的混杂子空间**，而丢弃对因果效应估计无关的噪声成分。

---

## 2. 生成模型（Generative Model）

### 2.1 潜变量先验

- 潜变量 (U) 的先验为标准多元高斯：
  
  [  
  p(U) = \mathcal N(0, I_{d_U}).  
  ]

### 2.2 代理变量生成

- 将代理变量拼接为 (X_{\text{all}} = [X,W,Z])。
- 条件生成模型：
  
  [  
  p_\theta(X_{\text{all}} \mid U)  
  = \mathcal N\big( \mu_X(U),\ \mathrm{diag}(\sigma_X^2) \big),  
  ]

其中：

- (\mu_X(U) = f_X(U; \theta_X))：MLP 解码器输出均值；
- (\sigma_X^2)：
  - 规范默认：**常数方差 = 1**（对应 MSE loss）；
  - 如果实现中使用 learnable log-variance，需要在文档中明确标注。

### 2.3 处理生成

- 二元处理 (A \in \{0,1\})：
  
  [  
  p_\theta(A \mid U) = \mathrm{Bernoulli}(\pi_A(U)),\quad  
  \pi_A(U) = \sigma(f_A(U; \theta_A)),  
  ]

其中 (\sigma) 为 sigmoid，(f_A) 为 MLP。

> 规范要求：
> - (A) **仅作为监督目标**，不输入 encoder；
> - 不允许通过把 A 拼到 (X_{\text{all}}) 来“伪造”监督。

### 2.4 结果生成

- 连续结果 (Y \in \mathbb R)：
  
  [  
  p_\theta(Y \mid U, A)  
  = \mathcal N\big( \mu_Y(U,A),\ \sigma_Y^2 \big),  
  ]

其中：

- (\mu_Y(U,A) = f_Y([U,A]; \theta_Y))：MLP，输入为 ([U,A])；
- (\sigma_Y^2)：规范默认**常数方差 = 1**（对应 MSE loss）。

> 若 Y 为二元，可改为 Bernoulli 形式；  
> 若为计数，可改为 Poisson / Negative Binomial。  
> 这些扩展不属于 v2.1 标准版，需另行编号。

---

## 3. 推断模型（Inference / Encoder）

### 3.1 变分后验

采用标准 VAE 变分后验：

[  
q_\phi(U \mid X_{\text{all}})  
= \mathcal N\big( \mu_\phi(X_{\text{all}}),\ \mathrm{diag}(\sigma_\phi^2(X_{\text{all}})) \big).  
]

- **重要规范：** Encoder 的输入为 (X_{\text{all}} = [X,W,Z])，**禁止**将 (A)、(Y) 输入 encoder。

### 3.2 Encoder 网络结构（推荐）

- 输入维度：(d_{\text{all}})
- 隐藏层：例如两层 MLP + ReLU：
  - `[128, 64]` 或 `[64, 64]`（根据当前 best practice 固定）
- 输出：
  - `mu = W_μ h + b_μ`（维度 (d_U)）
  - `logvar = W_σ h + b_σ`（维度 (d_U)）

### 3.3 重参数化采样

[  
\epsilon \sim \mathcal N(0, I),\quad  
U = \mu_\phi + \sigma_\phi \odot \epsilon,  
\quad \sigma_\phi = \exp(0.5\,\log\sigma_\phi^2).  
]

> 规范要求：
> - 使用 diagonal Gaussian + reparameterization；
> - 不使用 flow / mixture / 其他复杂后验结构（否则需升级版本号）。

---

## 4. 损失函数与训练目标

### 4.1 扩展 ELBO

对单个样本，IVAPCI v2.1 的扩展 ELBO 为：

[  
\begin{aligned}  
\mathcal J(\theta,\phi)  
&= \mathbb E_{q_\phi(U\mid X_{\text{all}})}  
\big[ \log p_\theta(X_{\text{all}}\mid U) \big] \  
&\quad + \lambda_A \,\mathbb E_{q_\phi(U\mid X_{\text{all}})}  
\big[ \log p_\theta(A\mid U) \big] \  
&\quad + \lambda_Y \,\mathbb E_{q_\phi(U\mid X_{\text{all}})}  
\big[ \log p_\theta(Y\mid U,A) \big] \  
&\quad - \beta\,\mathrm{KL}\big(q_\phi(U\mid X_{\text{all}}) \,|\, p(U)\big).  
\end{aligned}  
]

整体验证集上的目标是最大化 (\mathcal J)，实现中用**负号**作为损失：

[  
\mathcal L = -\mathcal J.  
]

### 4.2 损失分解（实现形式）

在实现中，将损失写为：

[  
\mathcal L  
= \mathcal L_{\text{proxy}}
- \lambda_A \mathcal L_A
- \lambda_Y \mathcal L_Y
- \beta \mathcal L_{\text{KL}},  
]

其中：

- **代理重构损失**（MSE，对应高斯对数似然）：  
  [  
  \mathcal L_{\text{proxy}}  
  = \frac{1}{d_{\text{all}}}\|X_{\text{all}} - \hat X_{\text{all}}(U)\|^2.  
  ]
- **处理预测损失（BCE）**：  
  [  
  \mathcal L_A  
  = -\big( A\log \pi_A(U) + (1-A)\log(1-\pi_A(U)) \big).  
  ]
- **结果预测损失（MSE）**：  
  [  
  \mathcal L_Y  
  = (Y - \hat Y(U,A))^2.  
  ]
- **KL 损失（高斯 KL）**：  
  [  
  \mathcal L_{\text{KL}}  
  = \frac{1}{2}\sum_{j=1}^{d_U}  
  \big( \mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2 \big).  
  ]

### 4.3 标准超参数（建议固定）

为了保证复现一致性，建议将以下作为 v2.1 的标准默认值（除非注明）：

- β = 1.0
- λ_A = 1.0
- λ_Y = 1.0

如需调参，应记录在实验日志中，并在文档或 README 中明确标记为“非标准配置”。

---

## 5. 训练与实现细节规范

### 5.1 数据预处理

1. 拼接代理变量：

   ```python
   X_all = np.concatenate([X, W, Z], axis=1)  # shape (n, d_all)
   ```

2. 对 (X_{\text{all}}) 做列级标准化：

   - 每一列减去均值，除以标准差；
   - 保存 `mean_`、`std_` 以便复现；
   - 规范建议：按训练集统计，它们用于训练 + 测试集的变换。

3. 处理 A：

   - 使用 0/1 编码；
   - 不做标准化。

4. 处理 Y：

   - 建议对 Y 做标准化（减均值 / 除标准差）以利于训练；
   - 在报告 ATE 时再还原到原尺度。

5. 训练 / 验证划分（若使用 early stopping）：

   - 固定随机种子（例如 1234）；
   - 常用比例：80/20 或 90/10；
   - 将划分策略写入文档。

### 5.2 网络结构参考

以 PyTorch 为例，推荐“标准”结构（可以根据当前 best 配置写死）：

- Encoder MLP：
  - 输入：`d_all`
  - 隐层：`[128, 64]` + ReLU
  - 输出：`mu` (d_U), `logvar` (d_U)
- Latent 维度：
  - `d_U = 4` 或 `8`（需在文档固定）
- Proxy Decoder：
  - 输入：`U`
  - 隐层：`[64, 64]` + ReLU
  - 输出：`X_recon` (d_all)（线性层 + MSE）
- A Head：
  - 输入：`U`
  - 隐层：`[32]` + ReLU
  - 输出：`logits_A` (1)（线性层，训练时用 `BCEWithLogitsLoss`）
- Y Head：
  - 输入：`[U, A]`
  - 隐层：`[64, 32]` + ReLU
  - 输出：`Y_pred` (1)

> 若实际代码采用不同层数 / 宽度，应在 `IVAPCI_v2.1_spec.md` 或代码注释中明确写出，并视为该环境下的“标准实现”。

### 5.3 训练超参数

推荐设置（可视当前 best 实践微调）：

- 优化器：Adam
- 学习率：`1e-3`
- 批大小：`batch_size = 128`
- 训练轮数：`epochs = 500`（配合 early stopping）
- Gradient clipping：可选，例如 `max_norm = 5.0`
- Weight decay：`0` 或 `1e-4`
- 随机种子：
  - Python / NumPy / PyTorch (CPU & CUDA) 全部设定（例如 `seed = 42`）

### 5.4 标准训练 Loop（伪代码）

```python
for epoch in range(num_epochs):
    model.train()
    for X_all_batch, A_batch, Y_batch in train_loader:
        # 1. Encode
        mu, logvar = encoder(X_all_batch)  # (B, d_U)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        U = mu + std * eps  # reparameterization

        # 2. Decode proxies
        X_recon = proxy_decoder(U)

        # 3. Heads for A and Y
        logits_A = A_head(U)  # (B, 1)
        Y_pred = Y_head(torch.cat([U, A_batch], dim=1))  # (B, 1)

        # 4. Loss terms
        recon_loss = mse_loss(X_recon, X_all_batch)
        A_loss = bce_with_logits_loss(logits_A, A_batch)
        Y_loss = mse_loss(Y_pred, Y_batch)

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / X_all_batch.size(0)

        # 5. Total loss
        loss = recon_loss + lambda_A * A_loss \
             + lambda_Y * Y_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # (Optional) validation + early stopping
```

规范要求：

- KL 采用标准 closed-form；
- 所有 loss 在 batch 内做平均或按样本数归一；
- 使用 reparameterization trick 而非直接对 U 采样。

---

## 6. 下游 DR / DML 使用规范

### 6.1 使用哪一个 Û：均值 vs 采样

标准做法：

- **使用 encoder 输出的均值 (\mu_\phi(X_{\text{all}})) 作为表示 (\hat U)**：  
  [  
  \hat U_i = \mu_\phi(X_{\text{all},i}).  
  ]
- 不使用随机采样的 U（除非特定实验需求），理由：
  - 均值表示更平滑、可重复；
  - 下游 ATE 估计的方差更小；
  - 当前仿真结果也基于该策略。

### 6.2 DR / DML 的标准流程（单数据集）

以 DR-GLM 为例，给出“规范版”的步骤：

1. **训练 VAE：**
  - 使用训练集数据 ((X,W,Z,A,Y)) 训练 IVAPCI v2.1 模型；
  - 固定参数 ((\hat\theta,\hat\phi))。

2. **抽取潜在表示：**
  - 对全体样本（或按 cross-fitting 划分），计算：  
    [  
    \hat U_i = \mu_{\hat\phi}(X_{\text{all},i}).  
    ]

3. **K 折 cross-fitting：**
  - 例如 K = 2 或 5；
  - 将样本划分为 K 份 ({\mathcal I_k})。

  对每一折 (k)：
  - 在折外样本 (\mathcal I_{-k}) 上拟合：
    - 倾向得分模型：  
      [  
      \hat e_{-k}(u) \approx \mathbb P(A=1\mid \hat U=u),  
      ]  
      标准配置为：logistic 回归（GLM）。
    - 结果回归模型（对每个 a = 0,1）：  
      [  
      \hat m_{-k}(a,u) \approx \mathbb E[Y\mid A=a,\hat U=u],  
      ]  
      标准配置为：线性回归（带或不带正则化）。
  - 对折内样本 (i \in \mathcal I_k)，计算 DR/AIPW 打分：  
    [  
    \psi_i =  
    \Big( \frac{A_i - \hat e_{-k}(\hat U_i)}{\hat e_{-k}(\hat U_i)(1-\hat e_{-k}(\hat U_i))} \Big)  
    \Big( Y_i - \hat m_{-k}(A_i, \hat U_i) \Big)  
    - \hat m_{-k}(1,\hat U_i) - \hat m_{-k}(0,\hat U_i).  
    ]

4. **ATE 估计：**  
  [  
  \hat\tau_{\text{IVAPCI-DR}}  
  = \frac{1}{n}\sum_{i=1}^n \psi_i.  
  ]

> 规范要求：
> - VAE 训练和 DR 模型拟合要做到**数据切分清晰**，建议：
>   - VAE 用全 train 数据训练（不 cross-fit）；
>   - 倾向得分与结果回归用 (\hat U) + cross-fitting；
> - 不在 DR 步骤中再次调参或更新 encoder。

---

## 7. 理论直觉简述（可选）

从信息论与因果的视角，IVAPCI v2.1 的目标可以理解为求解：

[  
\max_q\ I(\hat U; X,W,Z) + \alpha\cdot I(\hat U; (A,Y))\quad  
\text{s.t. } I(\hat U; \text{data}) \le C,  
]

其中 (\alpha) 由 (\lambda_A,\lambda_Y,\beta) 等超参决定。  
在有限复杂度约束下，(\hat U) 被鼓励：

- 保留与代理、处理、结果相关的共同信息；
- 丢弃只与代理有关但与 (A,Y) 无关的噪声成分。

因此，(\hat U) 近似于 (U) 的**混杂子空间投影**，比“完整 U”（Oracle U）更不容易导致下游模型过拟合，从而在有限样本下获得更小的方差与 RMSE。

---

## 8. 复现 Checklist

为了避免未来“改一改结果就跑偏”，建议每次实验 / 代码版本都检查以下项目：

1. **因果图结构是否保持不变？**
  - 是否仍为  
    (U \to (X,W,Z), U\to A, (U,A)\to Y)？
  - Encoder 输入中是否**没有** A、Y？

2. **VAE 结构是否为标准高斯 VAE？**
  - 高斯先验 / 高斯后验；
  - MSE 重构代理；
  - reparameterization 采样。

3. **AY 监督是否通过 log-likelihood 形式加入？**
  - 损失中有明确的 (\mathcal L_A, \mathcal L_Y)；
  - 没有通过“把 A,Y 当成 extra feature 重构”来偷懒。

4. **超参数是否记录完整？**
  - β, λ_A, λ_Y；
  - Latent 维度 d_U；
  - 网络层数、宽度、激活函数；
  - 学习率、batch size、epochs、early stopping 策略；
  - 随机种子、数据预处理方式。

5. **下游 DR / DML 是否遵循标准流程？**
  - 使用 (\hat U = \mu_\phi(X_{\text{all}}))；
  - 使用 cross-fitting；
  - 清楚区分 VAE 训练和 DR 模型训练。

只要这些核心点被严格记录和遵守，IVAPCI v2.1 的性能就可以在不同环境和代码库之间保持高度可复现，后续改进（如加入 nuisance 分支、p-adic loss 等）也可以在此基准上进行清晰对比。

---

> 如需扩展版本（如 IVAPCI v2.2 / PACD-T v3.0），建议新建独立规范文档，并在开头明确与 v2.1 相比增加了哪些结构（例如 U_n 分支、对抗损失、p-adic 距离项等），避免概念混淆。
