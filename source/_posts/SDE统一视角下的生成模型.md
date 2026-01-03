---
title: SDE统一视角下的生成模型
tags:
  - DiffusionModel
date: 2024-07-01 23:06:08
---



# 摘要

从随机微分方程角度，尝试对生成模型的两类分支：DDPM 和 Score-based model 建立统一的模型框架。

> 论文：[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

<!-- more -->


# SDE 基本数学知识

随机微分方程 SDE 最早是物理上描述布朗运动（Wiener process）的工具， $d\mathbf{w}$ 表示了布朗运动的增量。定义 $w(t) \sim \mathcal{N}(0,t)$，那么在微小的时间步 dt 上，有以下关系成立：

![](/Assets/06290f1ea877a60f459f9a039164fea.jpg)

用下式描述一个一般的**前向 SDE**，$\mathbf{f} (\mathbf{x} ,t)$ 和 $g(t)$ 分别称为 drift coeff 和 diffusion coeff：
<div>
$$
d\mathbf{x} =\mathbf{f} (\mathbf{x} ,t)dt+g(t)d\mathbf{w} 
$$
</div>

对于一个给定的前向 SDE（已知 drift 和 diffusion 系数），能够通过数学方式[推导]( https://ludwigwinkler.github.io/blog/ReverseTimeAnderson/ )出逆向过程，即 **Reverse SDE**：
<div>
$$
d\mathbf{x} =[\mathbf{f} (\mathbf{x} ,t)-g(t)^2\nabla _x\log p_t(x)]dt+g(t)d\bar{\mathbf{w} } 
$$
</div>

# SDE 视角下的 DDPM

下面证明了 DDPM 的前向过程等价于一个 SDE，采样过程等价于对应的 Reverse SDE。
> 注意：虽然 DDPM 没有显式地计算 Score，但是下面的 ** \* ** 式给出了一种等价变换得到 Score 的方式。

![](/Assets/3d13fc2dcc9dcbd9efc83844ed4cadd.jpg)


# SDE 视角下的 SMLD

由于 SMLD 本身就是基于 Score 给出的模型，因此 SDE 视角下的 SMLD 推导就显得简洁许多。

![](/Assets/ccfb4144d2adb62fe6ec252da83527a.jpg)


# Reference

- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)
- [Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song‘s Blog](https://yang-song.net/blog/2021/score/)
