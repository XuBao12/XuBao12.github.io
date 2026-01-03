---
tags: []
parent: ""
collections:
    - 'Generative Model'
title: '论文精读 DDPM'
date: 2024-05-20T21:45:49.000Z
$version: 248
$libraryID: 1
$itemKey: 7WVAIN6R

---
## 摘要

2020 年 6 月，Jonathan Ho 等学者对之前的扩散概率模型进行了简化，并通过变分推断，将后验问题转为优化问题进行建模，提出了经典的**去噪扩散概率模型（DDPM）**，将扩散概率模型的思想用于图像生成，目前所说的扩散模型，大多是基于该模型进行改进。

> 论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

<!-- more -->

简单来说，DDPM包括两个过程：正向过程和逆向过程。

正向过程用 $q(x_t|x_{t-1})$ 表示，理解为以固定方式添加噪声，破坏原有的数据分布。

逆向过程用 $p_\theta(x_{t-1}|x_t)$ 表示，理解为从噪声中重建出数据分布的过程。$\theta$ 表示涉及模型的参数。 ![](/Assets/f604799df1c26e57ae4d557771ccf31.jpg)

## 正向过程

从真实数据分布 $X_0 \sim q(x)$ 中采样得到数据点 $x_0$（其实是一张图），在每一个时间步 t 上添加噪声，得到加噪后的样本 $x_t$。t 不断增大，直到最终 T 时刻称为各项独立的高斯分布。加噪方式为：

\$\$ q\left( x\_t|x\_{t-1}\right ):=N\left(\sqrt{1-\beta\_t}x\_{t-1},\beta\_t\right ) \$\$

由于每个时间步相对独立，因此是一个**马尔科夫链**过程，这是下面推导的重要前提。

\$\$ q(x\_{1:T}|x\_0):=\prod\_{t=1}^{T}q(x\_t|x\_{t-1}) \$\$

下面基于 $x_0$ 和 $\beta_t$ 推导**任意时刻的带噪样本** $x_t$：

![](/Assets/a5116fdf4e5b6647275b4b0dfff2292.jpg) ![](/Assets/dad4f3a5305f67ce72de0d9518c13c5.jpg)

## 反向过程

整体上是一个去噪的过程，根据真实分布 $q(x_{t-1}|x_t)$ 从 $x_T$ 采样得到 $x_{T-1}$，一步步逐步还原得到 $x_0$。

假设真实分布 $q(x_{t-1}|x_t)$ 满足高斯分布，但是无法直接拟合，所以用神经网络估计这一分布：

\$\$p\_\theta(x\_{t-1}|x\_t)=N\left( \mu\_\theta (x\_t,t),\Sigma \_\theta (x\_t,t)\right)\$\$

下面推导**后验条件概率**： ![](/Assets/fe52f2125e868399abe8766bbc29f49.jpg)

优化目标是使得神经网络估计的分布和后验条件分布越接近越好，KL 散度计算推导，详见 {% post\_link Rethinking-of-Diffusion-Model %}。最终的优化目标是使两者的均值最为接近，进一步重参数化变成**估计噪声**。

## 流程

![](/Assets/Pasted%20image%2020240520123433.png) ![](/Assets/558c0b1db8643c7cf5c72bad3001cef.jpg)

## 注意点

1.  重参数化 使不可微的从分布中采样这一过程转化成了可微过程，从而反向传播。 ![](/Assets/Pasted%20image%2020240520115030.png)
2.  $\beta_t$ 的选择 使用Linear schedule加噪方式，$\beta_0 \to 0$，$\beta_T \to 1$，且$\beta_t$随着t递增。

## 代码演示

[Diffusion Model.ipynb](/Assets/Diffusion%20Model.ipynb)

1.  正向加噪过程 ![](/Assets/Pasted%20image%2020240520124450.png)
2.  经过4000 轮训练后，反向生成结果 ![](/Assets/Pasted%20image%2020240520124557.png)
