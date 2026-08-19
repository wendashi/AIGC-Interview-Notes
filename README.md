# AIGC-八股记录📝

## 📋 目录

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
  - [Loss](#loss)
  - [Transformer](#transformer)
  - [ROPE](#rope)
  - [Normalization](#normalization)

## Image Generation

#### CFG

#### DDPM，DDIM，

(1) `Ho/Diffusion lineage` 指的是从 **Ho et al. 的 DDPM (Denoising Diffusion Probabilistic Models, 2020)** 开始的这条技术主线。

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
- "动机→贡献→局限→可追问"准备
- Motivation：
    - 提出了一个反事实问题：扩散模型里大家长期默认 U-Net 是不是必须的？DiT 这个工作把这个默认假设打掉了，在 LDM latent 空间里用 ViT 风格的 transformer 主干。
    - 他们发现：用前向计算量（Gflops）而不是参数量来看扩展更关键，`更高 GFLOPs 下 FID 明显下降`，验证了 scaling law，支持了"扩展优先于单点调参"，与"transformer 在 NLP/Vision 中有良好缩放性"思路一致。
        - **单点调参**：把模型结构和规模基本不变，只是在同一模型上改学习率、batch、步长、数据增强、采样步数、权重衰减等"单个/局部超参"来榨性能。
        - **扩展**：在给定算力预算内，按规律增大模型规模（更深更宽、更大输入 token、更多层等），即"系统性增加容量和计算量"（DiT里用 GFLOPs 描述）。
        - `GFLOPs = 10^9 FLOPs`；`FLOPs`（floating point operations）是浮点运算次数，常按一次乘加（MAC）计 2 次浮点运算（不同文献有时用约定不同，但方向一致）。
        - "一次前向传播大约要做多少**`十亿次浮点运算`**"，`数值越高通常意味着模型越耗算力、也通常能承载更复杂表达能力`。
    - 关键就是 '随 compute 规模化后的泛化规律是否稳定'，DiT 的贡献在于把这种规律成功搬到 denoising 主干上。
- Contribution：
    - 架构替换：把 LDM 的 U-Net 换成纯 transformer 主干，输入是 latent patch token。
    - 条件注入机制：作者强调"adaLN / adaLN-Zero"是关键设计，显著影响 FID，核心差异几乎只在此，说明贡献是"标准 ViT 的有效移植"。
    - 计算规模规律：提出并验证"模型规模（深度宽度）与 token 数（patch 大小时）一起决定复杂度和质量"，并比较了 12 个模型（S/B/L/XL × patch 8/4/2）。
    - 实验上：DiT-XL/2 在 ImageNet 256 和 512 达到 SOTA 水平（256 上 FID 2.27、512 上 FID 3.04）。
- 面试常见追问：为什么这些设计是对的
    - 为什么必须在 latent 空间？
        - 因为像素空间 token 数暴涨，attention 的二次复杂度会崩；`latent 32×32 空间下 token 数可控`。
    - 为什么不用参数量判断规模？
        - 因为论文强调参数量不能比较不同 patch 解析度下的复杂度，Gflops 更公平。
    - 为什么 adaLN-Zero 有用？
        - 它把时刻/类别条件稳定地注入残差路径，且参数初始上更利于优化。
    - 为什么与 U-Net 无关？
        - 论文把模型尽量做成"最标准 ViT"，避免工程 trick 过多。
- Limitation
    - 论文里明确把问题留到后续：更大规模更大 token 的上限、能否直接做文本条件的通用主干、VAE 能否被更 transformer 化替代。
    - 当前不是端到端消融一体化：仍依赖已有 VAE 编解码器（虽然这也算贡献清晰性的好处）。
    - 主要验证在 `class-conditional` ImageNet（非文本条件多模态）上，跨任务/多模态泛化是后续。
    - attention 复杂度对 token 增长仍敏感，latent 空间是实用解但也限制了"纯像素"的直接处理。

</details>

#### Flow matching-ICLR23

- Flow matching是怎么做的
    - Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. ICLR, 2023.
- DDPM的加噪方式
- DDPM和Flow Matching 区别为什么大家转FlowMatching
- 在上一段实习过程中遇到的问题以及如何解决
- 如果方法和模型都不变，如何去提高效果，涉及到一些优化trick，比较工程上的经验
- 现在ar做visual generation是怎么做的coding:给定一个数A，给定一组数字B，求由B中数字构成的小于A的最大数字。

#### Rectified flow-ICLR23

- Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023.

#### SD3/Flux-ICML24

- **Scaling `rectified flow` transformers for high-resolution image synthesis**
## 3D Generation

## General AI/ML Notes 手撕
#### Loss

MSELoss，BCE Loss，KL penalty 

#### Transformer

多头注意力MHA，(MQA)

Transformer Encoder Block /Decoder Block

#### LoRA

### ROPE

#### Normalization

LayerNorm，RMSNorm



