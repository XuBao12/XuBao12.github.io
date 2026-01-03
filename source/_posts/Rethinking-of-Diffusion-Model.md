---
title: Rethinking of Diffusion Model
tags:
  - DiffusionModel
date: 2024-06-05 11:00:11
---


# 摘要

从 VAE 原理出发，从数学角度推导扩散模型的 ELBO 和优化目标的来龙去脉。用三种方式（原始数据、噪声和分数）改写优化目标 ELBO 的去噪匹配项，解释 DDPM 为什么要使用学习噪音的方式建模扩散模型。

<!-- more -->


# 生成模型

对于我们感兴趣的某一**分布** $p(x)$，可以从中**采样**得到许多样本 $x$。生成模型（generative model）的目标是从这些样本中学习到真实的分布，从而采样得到新的样本值。

生成模型的几个大分类：
- GANs：基于对抗训练
- Likelihood-based：基于似然函数，包括 AE、flow、VAE 等
- Energy-based：基于能量函数
	- Score-based：把优化能量函数的问题转换成优化分数（score）

# 从 VAE 开始

对于样本 $x$，认为是隐变量（latent variable）$z$ 由某种方式决定的。变分自编码器的变分体现在使用概率分布来描述 $x$ 和 $z$，而不是确定的值。随机变量 $X$ 服从由 $z$ 决定的分布 $p_\theta(x|z)$，而隐变量 $z$ 在隐空间上服从先验分布 $p_\theta(z)$。我们最终的优化目标是最大化对数似然 $\log p_\theta(x)$，虽然无法直接求得，但是可以转换为在参数 $\phi$ 下最大化 ELBO（Evidence Lower Bound，证据下界）：

<div>
$$
\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x,z)}{q_\phi(z|x)}\right] 
$$
</div>

![](/Assets/4262cb07caf520717d7b5c36aacc69a.jpg)

最大化 ELBO 的原因有以下两点：
1. 式子 16 说明了 ELBO 是似然的下界。
2. 式子 15 说明：$\log p(x)=ELBO+KL$。$p(x)$ 关于 $\phi$ 是常数，最大化 ELBO 等价于最小化 KL 散度，也就意味着我们可以用 **$q_\phi(z|x)$ 拟合真实的后验分布 $p_\theta(z|x)$** 。
> 因为 $p(x)=\int p(z)p(x|z)dz$，当固定 $p(x|z)$ 时，$p(x)$ 的值是不变的。调节 $q_\phi(z|x)$，也就是 ELBO 变大，会导致 KL 散度越来越小，最终为 0。宏观上就是用 $q_\phi(z|x)$ 拟合了真实后验分布 $p_\theta(z|x)$。

由此设计出如下的变分自编码器 VAE 结构，和 AE 隐变量是一个固定值不同，VAE 的 z 确切地说是隐空间。Encoder 中 x 通过预估的后验分布 $q_\phi(z|x)$ 生成隐变量 z，Decoder 中通过 $p_\theta(x|z)$ 解码得到 x。其中，$\phi$ 表示拟合的后验分布 q 的参数，$\theta$ 是真实分布的参数，两者都是要学习的，即 $\mathop{\arg\max}\limits_{\phi,\theta}(ELBO)$。

![](/Assets/005ff289bc56f378d5e662a8b02ff00%204.jpg)

ELBO 可以写成两项之和，最大化 ELBO 就等于最大化它的第一项，最小化它的第二项。
- 第一项衡量 decoder 在变分分布中的似然，这确保了根据学习到的隐变量 z 的分布可以重建出原始 x 的分布。
- 第二项衡量的是 encoder 学到的变分分布保持了多少隐变量的先验信息 $p(z)$。减少这项鼓励编码器实际学习分布，而不是分解成狄拉克函数。

![](/Assets/b54ace7a1741e06ae099dd89f1619c8.jpg)

假设隐空间先验分布 $p(z)$ 为标准正态，后验分布 q 为均值 $\mu$，协方差 $\sigma^2 I$ 的正态，即：
![](/Assets/Pasted%20image%2020240605160941.png)

ELBO 的第二项 KL 散度可以直接计算。第一项用蒙特卡洛估计，先从分布 $q_\phi(z|x)$ 中对数据集所有的 x 采样（采样时用重参数化技巧使得能够反向传播）得到 $z^{l}|_{l=1}^L$，再对每个 $z^l$ 代入 $\log p_\theta(x|z)$ 求平均。从而目标函数简化为：
![](/Assets/Pasted%20image%2020240605160933.png)

上式的第一项也可以理解为在给定 z 的时候，decoder 重构得到 x 的 $p(x|z)$ 尽量高，类似于一个重构损失，即 $\hat{x}$ 和 $x$ 的 MSE，所以整个模型可以用下图概括：

![](/Assets/3b4ca9b16d7840fa1e8f2f0cdc1e208.jpg)

## 分层 VAE（HVAE）

![](/Assets/Pasted%20image%2020240605174207.png)

考虑分层 VAE 的一种特殊情况，称之为马尔可夫 HVAE (MHVAE)。在 MHVAE 中，生成过程为马尔可夫链，也就是说，解码每个隐变量 $z_t$ 只需要前一个隐变量 $z_{t-1}$ 作为条件。由于是马尔可夫，因此概率分布可以递归表示为每一层的累乘：
![](/Assets/Pasted%20image%2020240605174601.png)

此时可以对优化目标 ELBO 进行形式上的推广，在下一章变分扩散模型 VDM 中，将会把其分解为几个特定项之和。

![](/Assets/Pasted%20image%2020240605174618.png)
![](/Assets/Pasted%20image%2020240605174703.png)

# Variantional Diffusion Model

在分层 VAE 基础上添加<span id="restriction">三个条件</span>：
1. 潜变量 z 的维度和数据 x 保持一致。
2. 每步的 encoder 固定为线性高斯模型，只依赖于上一个时间步的结果（马尔科夫条件）。意味着后验分布不再是由参数 $\phi$ 决定，而是由上一个时间步决定均值和方差的高斯分布：
<div>
$$
q\left( x_t|x_{t-1}\right ):=N\left(\sqrt{1-\beta_t}x_{t-1},\beta_t\right )
$$
</div>
3. 最终时刻的隐变量分布为标准高斯分布。

![](/Assets/Pasted%20image%2020240605182459.png)

不同于 VAE 模型需要学习 encoder 和 decoder，对于 VDM 模型，只需要学习 decoder $p_\theta(x_{t-1}|x_t)$。模型优化完成后，从标准高斯分布中采样一个 $x_T$，递归地从分布 $p_\theta(x_{t-1}|x_t)$ 中依次采样 $x_{T-1},x_{T-2},...,x_0$，得到采样结果。

## ELBO 

对于 VDM 模型，在式 29 基础上对 ELBO 进行形式上的扩展：

![](/Assets/c83b6bfc438220b4db1905f489beeb4.jpg)

ELBO 最终分解为三项，它们的意义分别如下：
- Reconstruction term：通过第一层 $x_1$ 重建得到的 $x_0$ 对数似然尽量大。
- Prior matching term：使得最后一层的 q 分布趋近于 $x_T$ 的先验分布，即标准高斯分布。注意到由于 q 分布都是预先设定的，因此这一项事实上不用优化，且在 T 足够大时保持为 0.
- Consistency term：对于中间每一层的 $x_t$ 的分布，要保持从正向过程 $q(x_t|x_{t-1})$ 和反向过程 $p_\theta(x_t|x_{t+1})$ 得到的结果一致，也就是最小化这两个分布的 KL 散度。

对于 ELBO 的三项表达式，由于都是期望的形式，可以用蒙特卡洛法结合重参数化进行计算（参考 VAE 中式 22 的处理方式）。但是第三项 consistency term 与两个随机变量 $x_{t-1}$ 和 $x_{t+1}$ 有关，用蒙特卡洛估计会导致方差比单个变量的情形要大（？），随着 T 增加，方差的变大会线性增长。因此希望用另一种变换方式把 ELBO 拆解为只与单个随机变量有关的形式。

![](/Assets/8a93256e776e09aad050b0e97443744.jpg)

同样将 ELBO 最终分解为三项，它们的意义分别如下：
- Reconstruction term：通过第一层 $x_1$ 重建得到的 $x_0$ 对数似然尽量大。
- Prior matching term：使得最终的 q 分布趋近于 $x_T$ 的先验分布，即标准高斯分布。注意到由于 q 分布都是预先设定的，因此这一项事实上不用优化，且在 T 足够大时保持为 0.
- **Denoising matching term**：后验分布 $q(x_{t-1}|x_t,x_0)$ 理解为真实的后验分布（ground-truth denoising transition step），$p_\theta(x_{t-1}|x_t)$ 理解为要拟合的去噪分布，去噪匹配项的目标就是最小化两者的 KL 散度，使两个分布逼近。

注意到以上两种 ELBO （式 45 和式 58）的推导仅使用到了马尔可夫性质，因此对于任意 MHVAE 都适用。而 VDM 比通常的 MHVAE 多了 [三条限制](#restriction)，导致 q 分布是已知的，这是两者主要的区别。另外，当 T=1 时，VDM 就退化为了普通的 VAE，ELBO 的形式也退化为 VAE 的形式（式 19）。

总结对 ELBO 的形式上的拆分，我们发现最大化 ELBO 这一优化目标最终成为了最小化 Denoising matching term，即去噪匹配项，这也是整个扩散模型的**核心**。后面我们将对去噪匹配项进行进一步变换，转换为对其他形式参数的优化，从而让神经网络进行学习。


## 去噪匹配项

首先对于真实的后验分布 $q(x_{t-1}|x_t,x_0)$ ，通过递归以及重参数化技巧可以计算得到其精确值，可以证明它满足均值为 $\mu_q(x_t,x_0)$ ，方差为 $\Sigma_q(t)$ 的正态分布。类似 {% post_link "论文精读 DDPM" %} 中的证明步骤：

![](/Assets/d4da6e21e6e0cc3731a8aafbb1f172b.jpg)

回顾我们的最终目标：最小化去噪匹配项。真实后验分布 q 已经求得，现在处理要拟合的去噪分布 $p_\theta(x_{t-1}|x_t)$ 。为了和 q 形式一致，假设其为正态分布，方差和 q 保持相同 $\Sigma_q(t)$，均值为 $\mu_\theta(x_t,t)$，**也就是均值中不能出现 $x_0$**。对两个高斯分布计算 KL 散度，最终将优化目标转变为最小化 $\mu_\theta$ 和 $\mu_q$ 的距离。


![](/Assets/c5a6740fda09acfe72a38839460235a.jpg)


使用神经网络预测得到 $\mu_\theta$，代入公式 92 即为最终的优化目标。 $\mu_\theta(x_t,t)$ 的形式仍是未知的，我们希望它和 $\mu_q(x_t,x_0)$ 尽可能接近。然而 $\mu_q(x_t,x_0)$ 与 $x_0$ 有关，这就要求我们对其进行重构，表达成仅包含 $x_t$ 和 $t$ 的形式，然后根据 $\mu_q$ 的形式来构造 $\mu_\theta$。

下面给出了模型在任意时刻 t 上建模的三种方式：直接预测原始数据 $x_0$，预测噪声 $\epsilon$，预测分数 $\nabla \log p(x_t)$。例如 DDPM 就是采用了第二种预测噪声的方式，而 scored-based 模型则是第三种。三种方式事实上是等价的。
![](/Assets/938f0d571bc0441a7034f31a2f19915%201.jpg)

### 预测原始数据

第一种直接的想法是，把式 93 的 $x_0$ 换成神经网络预测的 $\hat{x}_\theta(x_t,t)$, 这样就得到了 $\mu_\theta$ 的一种形式：
![](/Assets/b78d2fc3eb1ad05df66a189fcfdf05b.jpg)

由 93 94 代入式子 92 就能得到优化目标：
![](/Assets/3f83e3fadcb0aaa6578ed38d62f032a.jpg)

### 预测噪声

第二种方式，使用重参数化技巧把 $x_0$ 改写为 $x_t$ 和 $\epsilon$ 的形式，代入 $\mu_q$ 的表达式。
![](/Assets/307b73d92f36aa1901100feb574b493.jpg)

注意，上面的 $\epsilon_0$ 应当理解为真实的加噪噪声，它是由 $x_0$ 产生的。模仿 $\mu_q$ 的形式构造 $\mu_\theta$ ，这里 $\hat{\epsilon}$ 就是模型需要预测的值。

![](/Assets/4420e754c25962a18dc3771de9802fc.jpg)

类似地，优化目标就成了最小化 $\hat{\epsilon}$ 和 $\epsilon_0$ 的距离：
![](/Assets/b5faf087099eea23f7f9fc654620b8f.jpg)

### 预测分数

第三种方式，首先通过 [Tweedie's Formula](https://www.bilibili.com/video/BV1BK411X7mu/?p=5&vd_source=811347702e7ea15b02fafca7671039ad)，把 $x_0$ 转化为 $\nabla \log p(x_t)$ 的表达式，进而代入 $\mu_q$ 的式子。其中 $\nabla \log p(x_t)$ 就被称为“分数”，表征了如何移向数据空间最大似然的方向（由梯度决定）。

![](/Assets/c1d188fd555f63b2f7533b8cb507e69.jpg)

模仿 $\mu_q$ 的形式构造 $\mu_\theta$ ，这里的 $s_\theta(x_t,t)$ 就是模型预测的“分数”。

![](/Assets/49cc44df73838073a40278747e2a112%201.jpg)

类似地，优化目标就成了最小化模型预测的分数和真实分数的距离。
![](/Assets/98c8503b10b51a280fc0c598933e70c%201.jpg)





# Reference
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)
- [What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)
- [变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/5253)
