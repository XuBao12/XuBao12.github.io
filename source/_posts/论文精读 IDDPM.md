---
title: 论文精读 IDDPM
date: 2024-06-03 12:19:56
tags:
  - 论文精读
  - DiffusionModel
---



# 摘要
IDDPM对原始DDPM做出了改进，提高了性能。引入了：
1. 参数化方差项，可学习 
2. 噪声策略$\beta_t$的改进 
3. 均匀采样改进为重要性采样

> 论文：[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

<!-- more -->

# DDPM回顾
正向过程q：
<div>
$$
q\left( x_t|x_{t-1}\right ):=N\left(\sqrt{1-\beta_t}x_{t-1},\beta_t\right )
$$
</div>
<div>
$$
q(x_{1:T}|x_0):=\prod_{t=1}^{T}q(x_t|x_{t-1})
$$
</div>
反向过程p无法直接求得，假设其为高斯分布：
<div>
$$
p_\theta(x_{t-1}|x_t):=N\left( \mu_\theta(x_t,t), \Sigma_\theta(x_t,t) \right )
$$
</div>
q的后验分布：
<div>
$$
q(x_{t-1}|x_t,x_0)=N\left(\tilde{\mu} (x_t,x_0), \tilde{\sigma}_t(x_t,x_0) \right )
$$
$$
\tilde{\sigma}_t(x_t,x_0) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t=\tilde{\beta}_t
$$
$$
\tilde{\mu}_t (x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1} )}{1-\bar{\alpha}_{t} }x_t+\frac{\sqrt[]{\bar{\alpha }_{t-1}}\beta _t }{1-\bar{\alpha}_{t}}x_0
或
\tilde{\mu}_\theta  (x_t,t)=\frac{1}{\sqrt{\alpha _t} }\left (x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha }_t }}\epsilon _\theta(x_t,t )\right )
$$
</div>

优化目标是使得q的后验分布$q(x_{t-1}|x_t,x_0)$和模型学习到的p分布$p_\theta(x_{t-1}|x_t)$尽量接近，即优化两者的KL散度。可以通过预测$x_0$，也可以预测噪声来拟合均值。实验发现预测噪声效果更好，即：
<div>
$$
L_{simple} = E[||\epsilon-\epsilon _\theta(x_t,t )||^2]
$$
</div>

# IDDPM
## 参数化方差项
DDPM中的方差$\tilde{\sigma}_t(x_t,x_0)$忽略了前面的系数直接设置为了$\beta_t$，对结果影响不大，这是因为两者差异很小（尤其在T很大，或者扩散步数增大时）。然而在步数较小时的变分下界对损失有较大的影响，因此设计更好的方差学习方案有利于优化对数似然。

![image-20240603125842046](/Assets/image-20240603125842046.png)

![image-20240603130252631](/Assets/image-20240603130252631.png)

新的方差为$\Sigma_\theta(x_t,t)=exp(v\log \beta_t +(1-v)\log \tilde{\beta}_t)$,v是可学习参数，本质上是对两个方差进行加权。相应地，设计新的损失函数：$L_{hybrid}=L_{simple}+\lambda L_{vlb}$，实验中设置系数$\lambda$为0.001

## 噪声策略的改进
DDPM的$\beta_t$是线性增长的，导致前向过程加噪速度过快，改进为cosine schedule的方式。

![image-20240603131203765](/Assets/image-20240603131203765.png)

## 重要性采样
希望直接优化$L_{vlb}$而不是$L_{hybrid}$，但是很难训练。采用下面的方法对$L_{vlb}$进行重要性采样，直观上讲就是维护了一个$L_t$，大的地方多采样一些点，小的地方少采样。

![image-20240603132424956](/Assets/image-20240603132424956.png)


![image-20240603132511968](/Assets/image-20240603132511968.png)
