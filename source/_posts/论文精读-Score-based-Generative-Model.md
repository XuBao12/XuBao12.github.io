---
title: 论文精读 Score-based Generative Model
tags:
  - 论文精读
  - DiffusionModel
date: 2024-06-17 21:17:55
---



# 摘要

基于分数的生成模型是与 DDPM 不同的一个分支。主要介绍宋飏老师提出的 NCSN (Noise Conditional Score Network)，模型主要基于**分数匹配**和**朗之万动力学采样**，以及训练时的加噪策略。

> 论文：[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

<!-- more -->


# 分数函数及郎之万动力学采样

首先定义 Stein Score：$s_\theta(x)= \nabla_x \log p_\theta (x)$。如图 19 (a) 所示，分数本质上是对数似然的梯度，表征了如何移向数据空间最大似然的方向，在似然较大的位置梯度较小，而在最大似然处的梯度为 0。分数函数 $s_\theta (x)$ 可以用来表征一个概率分布，使用参数 $\theta$ 建模。

> 注意 Stein 分数和最大似然的区别。Stein 分数是对 x 求导，而最大似然 $s_x(\theta)= \nabla_\theta \log p_\theta (x)$ 是对参数 $\theta$ 求导。换句话说，Stein 分数是已知参数 $\theta$ 下，计算最大概率取到的样本 x ；而最大似然则是在给定样本 x 下，求分布的参数 $\theta$ 最大概率的取值。

为了从分数表征的分布 $s_\theta (x)$ 中采样数据点，可以采用**郎之万动力学方法**。在数据空间中随机取一个初始点，将其朝着分数函数指示的方向移动，即**随机梯度下降**过程。迭代此过程最终稳定到似然取到最大位置的附近。此时得到的点 x 就是分布 $p(x)$ 概率密度局部最大的点，因此可以视为该分布下的一个采样值。随机项的引入是为了采样的多样性。

<div>
$$
X_{t+1}=x_t+\tau \nabla_x \log p (x_t) + \sqrt{2\tau}z,\qquad z\sim \mathcal{N}(0,1)
$$
</div>

![](/Assets/f86c766e7247d8ac5f5943981a65c40.jpg)

# （降噪）分数匹配

为了对分数函数进行优化，显然目标函数可以直接定义为网络估计出来的分数 $s_\theta(x)$ 和真实分数 $\nabla_x \log p(x)$ 的均方误差，希望两者越接近越好。

<div>
$$
\mathcal{L} = \mathbb{E}_{p (x)}[\left \| \nabla_x \log p (x) - s_\theta (x) \right \|^2 ]
$$
</div>

如果知道真实分布 $p(x)$，就可以完成损失函数的优化。但并不知道真实分布是什么，于是提出了分数匹配（score matching），该方法可以在未知 $p(x)$ 的情况下最小化损失函数。分数匹配有很多种处理方式，例如分层分数匹配（Sliced score matching）、降噪分数匹配（Denoising score matching）等。下面详述**降噪分数匹配**。

降噪分数匹配方法中，首先给原始数据添加随机噪声 $\epsilon$，即 $\tilde{x}=x+\epsilon$，如此构建了新的分布 $q(\tilde{x}|x)$。如果加噪很小，可以不破坏原有分布，这个新的分布可以近似认为和原来分布相同，因此可以转化为对这个新的分布估计其分数函数。代入损失函数推导如下：

![](/Assets/860a6d95c508f8bc26ac6fdb0b1e94a.jpg)

对上面 85 式使用蒙特卡洛方法取样计算其均值即可计算损失函数。下图所示的模型结构展示了这一过程。
![](/Assets/549d8c3bdc091b68754fb83c2d6687e%201.jpg)


# 面临的困难

1. **数据样本集中在嵌入高维空间的低维流形上**。说人话就是样本的各个维度并不完全互相独立，即列不满秩。对于 $64*64=4096$ 维的图像来说，实际互相独立的可能只有 2000 个维度，反映到分数匹配上就是偏导数理论上有任意解，训练不稳定。
2. **低概率密度区域的估计不准确**。本质上是观测样本数量不足的问题，概率密度低的地方其样本的产生概率也低，此时分数匹配方法对这些区域的拟合和学习就不足，导致预测不准确。
	![](/Assets/Clip_2024-06-17_19-51-51.png)
3. **朗之万采样的缺陷**。以高斯混合分布 $p(x)=c_1p_1(x)+c_2p_2(x)$ 为例，朗之万采样只能采样到概率密度极大位置，但是不同的概率密度的极大点亦有大小区别，也就是丢失了系数 $c_1$、$c_2$ 的信息。另外存在迭代速度慢的问题。

# 噪声条件分数网络

为了解决上述三个问题，NCSN 通过给数据增加噪声扰动的方式，一方面破坏了原来数据各个维度的相关性，另一方面扩大了数据分布高密度区域的面积。在数据分布上加上高斯噪声后，均值保持不变，方差变大，这会把高密度区域的面积增大，使得更多区域的分数函数被准确估计出来。

具体来说，定义了一个噪声序列，递进式地添加 L 个不同强度的噪声。即 $\{\sigma_i\}_{i=1}^L$，其中 $\sigma_1<\sigma_2<...<\sigma_L$，保证添加的是均值 0 方差 $\sigma_i$ 的高斯噪声，即条件概率 $p(\tilde{x}|x)=\mathcal{N}(\tilde{x};x,\sigma_i^2I)$。新得到的添加噪声 $\sigma_i$ 的分布如下：

![](/Assets/62322ff4605d847cd5d0d6e0d4f6d83%201.jpg)

> 注意：采样时只需取 $x\sim p(x)$ ，然后 $\tilde{x}=x+\sigma_iz,z\sim \mathcal{N}(0,I)$，如此得到的 $\tilde{x}$ 即满足加噪后的分布。

对于每一个 $\sigma_i$，计算其损失函数，用系数 $\lambda_i=\sigma_i^2$ 加权得到整体的损失函数。注意这里神经网络拟合的 score 需要 x 和 $\sigma_i$ 两个输入，相当于 $\sigma_i$ 作为条件控制。注意到这里的 x 是从原始样本加噪生成的，自然想到使用降噪分数匹配方法。

<div>
$$
\mathcal{L}=\sum_{i=1}^{L}\lambda_i\mathbb{E}_{p_{\sigma_i}(x)} \left [ \left \| s_\theta (x,\sigma_i)-\nabla_x\log p_{\sigma_i}(x) \right \|^2  \right ] 
$$
</div>

<div>
$$
\mathcal{L}=\sum_{i=1}^{L}\lambda_i\mathbb{E}_{p(x)} \left [ \left \| s_\theta(x+\sigma_iz,\sigma_i)+\frac{z}{\sigma} \right \|^2  \right ] 
$$
</div>

训练得到分数估计值 $s_\theta (x,\sigma_i)$ 之后，在采样时，对朗之万方法进行改进，称为退火朗之万算法。从噪声大的一端 $\sigma_L$ 开始逐步往噪声小的 $\sigma_1$ 迭代采样。后一层采样的初始点是上一层采样的结果。

![](/Assets/ald.gif)

# Reference

- [Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song‘s Blog](https://yang-song.net/blog/2021/score/)
- [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)