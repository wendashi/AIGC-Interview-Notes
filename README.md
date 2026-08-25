# AIGC-八股记录📝

## 📋 目录
- [Vision Language](#vision-language)
- [Image Generation](#image-generation)
  - [CFG](#cfg)
  - [DDPM，DDIM](#ddpmddim)
  - [LDM (Stable Diffusion)](#ldm-stable-diffusion-cvpr22)
  - [Controllable Image Generation Methods](#controllable-image-generation-methods)
  - [DiT](#dit-iccv23)
  - [Flow matching](#flow-matching-iclr23)
  - [Rectified flow](#rectified-flow-iclr23)
  - [SD3/Flux](#sd3flux-icml24)
- [3D Generation](#3d-generation)
- [General AI/ML Notes 手撕](#general-aiml-notes-手撕)
  - [Backpropagation](#backpropagation)
  - [Loss](#loss)
    - [MSE Loss](#mse-loss)
  - [Transformer](#transformer)
  - [ROPE](#rope)
  - [Normalization](#normalization)


- 每个知识点由原始论文的 "动机→贡献→局限→可追问" 组成。

## Vision Language

#### CLIP

#### DINO(v3)

#### Qwen3.8-VL

## Image Generation

#### CFG

#### DDPM，DDIM，

- `Ho/Diffusion lineage` 指的是从 **Ho et al. 的 DDPM (Denoising Diffusion Probabilistic Models, 2020)** 开始的这条技术主线。
- DDPM的加噪方式
- DDPM和Flow Matching 区别为什么大家转FlowMatching

核心思想是：

- 用固定步数往干净图像上加噪声（正向过程），再学习一个网络逐步去噪（反向过程）；
- 反向过程通常用网络预测噪声 \(\epsilon\) 或 \(x_0\)；
- 这条路线的代表模型包括 DDPM/ADM、DDIM、Latent Diffusion 里的 U‑Net 去噪器、Stable Diffusion 的部分骨架方法等。DiT 的“差异化贡献”其实是：它在这条主线上，把传统的 U‑Net 主干替换成 Transformer。

#### LDM (Stable Diffusion)-CVPR22

#### Controllable Image Generation Methods
- DreamBooth-CVPR23，Textual Inversion-ICLR23，Custom Diffusion-CVPR23，ControlNet-ICCV23，IP-Adapter-Arxiv
- 可控生成，定制生成

#### DiT-ICCV23

<details>
<summary>📖 详细内容（点击展开）</summary>

- 原始论文: [Scalable_Diffusion_Models_with_Transformers_ICCV_2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)
- Background：xie saining，beeples，成为了后来 SD3、Flux、Sora 的 backbone
- Motivation：
    - 提出了一个反事实问题：扩散模型里大家长期默认 U-Net 是不是必须的？DiT 这个工作把这个默认假设打掉了，在 LDM latent 空间里用 ViT 风格的 transformer 主干。
- Contribution：
    - 架构替换：把 LDM 的 U-Net 换成纯 transformer 主干，输入是 latent patch token。
      - latent patch token：DiT 不是“把图像先 patch 再进模型”，而是先经过 VAE 到 latent，再 patchify latent。完整链路：image → VAE encoder（下采样到 32×32、4 通道 latent）→ latent patch token（如 p=2/4）→ Transformer 去噪主干 → unpatchify 回 4×32×32/64×64 latent → VAE decoder 回图像。
    - 条件注入机制：DiT 要求模型在每个扩散时间步 t 和类别 y 下行为都不同，所以作者用了 adaLN / adaLN-Zero：
      - 让归一化和残差有条件参数，具体是由 (t,y) 经过 MLP 生成 scale/shift/gate，动态调制每一层。Zero 初始化让训练初期更稳定（残差门控从 0 开始，后续再逐步学）。
    - 他们发现：用前向计算量（Gflops）而不是参数量来看扩展更关键，`更高 GFLOPs 下 FID 明显下降`，验证了 scaling law，支持了"扩展优先于单点调参"，与"transformer 在 NLP/Vision 中有良好缩放性"思路一致。
        - **单点调参**：把模型结构和规模基本不变，只是在同一模型上改学习率、batch、步长、数据增强、采样步数、权重衰减等"单个/局部超参"来榨性能。
        - **扩展**：在给定算力预算内，按规律增大模型规模（更深更宽、更大输入 token、更多层等），即"系统性增加容量和计算量"（DiT里用 GFLOPs 描述）。
        - `GFLOPs = 10^9 FLOPs`；`FLOPs`（floating point operations）是浮点运算次数，常按一次乘加（MAC）计 2 次浮点运算（不同文献有时用约定不同，但方向一致）。
        - "一次前向传播大约要做多少`十亿次浮点运算`"，`数值越高通常意味着模型越耗算力、也通常能承载更复杂表达能力`。
    - 关键就是 '随模型所需算力的增加，模型效果是否稳定提升'，DiT 的贡献在于把这种规律成功搬到 denoising 主干上。
      - 计算规模规律：提出并验证"模型规模（深度宽度）与 token 数（patch 大小时）一起决定复杂度和质量"，并比较了 12 个模型（S/B/L/XL × patch 8/4/2）。
      - 实验上：DiT-XL/2 在 ImageNet 256 和 512 达到 SOTA 水平（256 上 FID 2.27、512 上 FID 3.04）。
- 面试常见追问：为什么这些设计是对的
    - 为什么必须在 latent 空间？
        - 因为像素空间 token 数暴涨，attention 的二次复杂度会崩；`latent 32×32 (256x256/8) 空间下 token 数可控`。
        - 用的是 LDM 的 VAE，只是改 backbone 部分。
    - 为什么不用参数量判断规模？
        - 因为论文强调参数量不能比较不同 patch 解析度下的复杂度，Gflops 更公平。
    - 为什么 adaLN-Zero 有用？(一种暂时性的妥协)
        - 完整路径: 'time/label -> MLP -> adaLN-Zero 的 shift/scale/gate 来调节每层特征'。
        - 是为了先把 transformer “能在 diffusion 框架下跑得稳”这件事解决掉”，所以尽量减少额外复杂性，做一个清晰、稳定的 baseline。
        - 基于 ImageNet 类别生成这个任务里，time/label 本身是“低维全局条件”，用 adaLN-Zero 做全局 FiLM 调制（shift/scale/gate）就是很自然、很高效的注入方式。
          - DiT 原版的场景不需要先把复杂度加到那么高，因此用了更“轻”的注入。
          - adaLN‑Zero = 低成本全局门控，cross‑attention = 高成本高表达力的局部对齐。
    - 为什么与 U-Net 无关？
        - 论文把模型尽量做成"最标准 ViT"，避免工程 trick 过多。
- Limitation
    - 论文里明确把问题留到后续：更大规模更大 token 的上限、能否直接做文本条件的通用主干、VAE 能否被更 transformer 化替代。
    - 当前不是端到端消融一体化：仍依赖已有 VAE 编解码器（虽然这也算贡献清晰性的好处）。
    - 主要验证在 `class-conditional` ImageNet（非文本条件多模态）上，跨任务/多模态泛化是后续。
    - attention 复杂度对 token 增长仍敏感，latent 空间是实用解但也限制了"纯像素"的直接处理。

</details>

#### Flow matching-ICLR23

<details>
<summary>📖 详细内容（点击展开）</summary>

- 原始论文: *[Flow Matching for Generative Modeling(Meta FAIR)](https://github.com/facebookresearch/flow_matching)*，ICLR2023。
- Background:
    ```
    传统 CNF(Continuous Normalizing Flows)
      ├── CNF 学习/定义一个高斯分布和图像分布之间的连续、可逆的映射
      └── ODE(Ordinary Differential Equation 常微分方程）生成，但最大似然训练昂贵,需要反复求解 ODE、计算散度，因此难以扩展到高维图像。
    Diffusion/Score Matching (训练更稳定，但其概率路径主要受扩散过程约束，路径可能比较弯曲，采样通常需要较多函数评估。
      ├── DDPM：随机反向过程
      ├── DDIM：可确定性采样
      └── Probability-flow ODE：也能写成 ODE
    Flow Matching
      └── 训练 CNF 的新方法
          ├── 可以使用 diffusion path
          └── 也可以使用 OT 等非 diffusion path
    ```
- Motivation
  - 能否直接指定一条从噪声到数据的概率路径，并监督产生这条路径的速度场？
  - 能否训练 CNF 时不求解 ODE，同时允许使用比 diffusion path 更简单的路径？
  - 如何绕过通常无法显式计算的边缘概率路径 $p_t(x)$ 和边缘速度场 $u_t(x)$？

- Contribution
  - 提出 Flow Matching 目标：
     $$\mathcal L_{\mathrm{FM}}(\theta) = \mathbb E_{t,\,X_t\sim p_t}\left[\left\|v_\theta(t,X_t)-u_t(X_t)\right\|_2^2\right].$$  
  - 提出可计算的 Conditional Flow Matching（CFM），证明它与 FM 对参数 $\theta$ 的梯度完全相同。  
  - 给出一般高斯条件路径的解析速度场。  
  - 证明 diffusion probability path 是该框架的特例。因此 FM 并不是只能使用 OT path，也不是与 diffusion 完全无关的模型。  
  - 引入条件 OT 路径。条件粒子沿直线、匀速运动，使速度回归和 ODE 数值求解更加简单。  
  - 在论文的同架构实验中，FM-OT 在 likelihood、FID 和 NFE 上总体优于所比较的 diffusion baselines。

- Limitation

  - **训练 simulation-free，不代表采样 simulation-free**：生成阶段仍然需要数值求解 ODE。
  - 条件 OT 路径对每个条件高斯分布是 OT displacement interpolation，但**不保证边缘化后的整体向量场是全局 OT 解**。
  - 论文将条件终点设置为  $$p_1(x\mid x_1)=\mathcal N\!\left(x\mid x_1,\sigma_{\min}^2I\right),$$ 因此当 $\sigma_{\min}>0$ 时，边缘终点只是近似数据分布。
  - 最终效果仍然受到神经网络拟合误差和 ODE 离散误差影响。
  - 计算 likelihood 仍需要积分向量场散度，实际中通常使用 Hutchinson trace estimator。
  - 原论文实验集中在 CIFAR-10、ImageNet 和 U-Net，不能只根据这些实验断言它对所有模态和架构都最优。

</details>


#### Rectified flow-ICLR23

- Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023.

#### SD3/Flux-ICML24

- **Scaling `rectified flow` transformers for high-resolution image synthesis**
## 3D Generation

#### SiT-ECCV24

- https://github.com/willisma/SiT

## General AI/ML Notes 手撕

#### Backpropagation

<details>
<summary>📖 详细内容（点击展开）</summary>

- 它利用链式法则，从 Loss 开始逐层向前计算每个模型参数的梯度。
- 常见要求是：1. 手写 Loss 的前向计算。2. 推导 Loss 对预测值的梯度。3. 解释梯度如何通过链式法则传到参数。4. 区分 `zero_grad()`、`backward()`、`step()`。

例如线性模型：

$$\hat Y=XW+b, L=\frac{1}{N}\sum(\hat Y-Y)^2$$


```python
# 手写反向传播：
# forward
pred = X @ W + b
error = pred - target
loss = (error ** 2).sum() / error.numel()

# backward
grad_pred = 2 * error / error.numel()
grad_W = X.T @ grad_pred
grad_b = grad_pred.sum(dim=0)

# update
with torch.no_grad():
    W -= lr * grad_W
    b -= lr * grad_b
```

对应链式法则: $$\frac{\partial L}{\partial W}=\frac{\partial L}{\partial \hat{Y}}\frac{\partial \hat{Y}}{\partial W}$$

```python
# PyTorch 版本：
optimizer.zero_grad()  # 清空上一次梯度
loss.backward()        # 自动计算 grad_W、grad_b
optimizer.step()       # 更新 W、b
```

面试最关键的一句： `backward()` 只计算并保存梯度，真正更新参数的是 `optimizer.step()`。
</details>

#### Loss

##### [MSE Loss](https://docs.pytorch.org/docs/2.13/generated/torch.nn.MSELoss.html)

<details>
<summary>📖 详细内容（点击展开）</summary>
  
- MSE（Mean Squared Error，均方误差）用于回归任务：计算预测值与真实值之差的平方，再对所有样本取平均。 
  - 为什么 MSE 适合回归？因为回归预测的是连续值。
  - Diffusion 本质是回归吗？整体目标是生成建模，但训练被转化成了回归问题。
- 平方的作用：1) 消除正负误差抵消。2) 大误差会受到更强惩罚。3) 连续可导，方便梯度下降。4) 缺点是对异常值比较敏感。
- 公式: $$\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2$$, 其中 $\hat y_i$ :模型预测值, $y_i$ :真实值, $N$ :元素或样本数量。
- 梯度: $$\frac{\partial \mathrm{MSE}}{\partial \hat{y}_i}=\frac{2}{N}(\hat{y}_i-y_i)$$
     
```
# x1：真实数据，例如一批图像
# x0：与 x1 形状相同的高斯噪声
x0 = torch.randn_like(x1)

# t：随机时间，并扩展维度以便与图像广播
t = torch.rand(x1.shape[0], device=x1.device)
t = t.view(-1, 1, 1, 1)

# t 时刻的中间状态
xt = (1 - t) * x0 + t * x1

# 正确答案：真实速度
target = x1 - x0

# pred：模型根据当前状态 xt 和时间 t 预测的速度
pred = model(xt, t)
# pred.shape   == target.shape == [batch_size, channels, height, width]
squared_error = (pred - target) ** 2
loss = squared_error.sum() / squared_error.numel()  # .numel() 是 PyTorch 张量 Tensor 的函数，返回张量中元素的总数量。
```
</details>

BCE Loss

Softmax

KL penalty 
KL散度计算

AUC(Area Under the Curve)

#### Transformer

多头注意力MHA，(MQA)

Transformer Encoder Block /Decoder Block

#### Self-Attention/Casual-Attention

#### Cross-Attention

#### Multi-head Attention



#### ROPE

#### KV cache

#### LoRA


#### Normalization

手撕RMSnorm/layerNorm/BatchNorm

#### DeepSpeed

#### 训练精度

LayerNorm，RMSNorm



