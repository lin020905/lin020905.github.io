---
title: Characteristics of random variables
date: 2025-11-16
categories: [TOP_CATEGORY, SUB_CATEGORY]
tags: [Probability Theory]     # TAG names should always be lowercase
math: true
---

### 4.1*数学期望*
#### 4.1.1 *随机变量数学期望定义*
1. 二项分布 $$\begin{gather}
X \sim B(n,p) \\
P(X = k) = C^k_n p^k q^{n-k}, \quad k = 0,1,2,\dots,n, \quad q=1-p \\
EX = np
\end{gather}$$
2. 泊松分布 $$\begin{gather}
X \sim P(\lambda)\\
P(x=k) = \frac{\lambda^k}{k!}e^{-\lambda}\\
EX = \lambda
\end{gather}$$
3. $X$ 服从参数为 $p$ 的几何分布 $$\begin{gather}
P(x=k) = q^{k-1}p, \quad k=1,2,\cdots,0 < p <1,q = 1-p \\
EX = \frac{1}{p} \\
\end{gather}$$
4. 超几何分布 $$\begin{gather}
P(x=k) = \frac{C^k_M C^{n-k}_{N-M}}{C^n_N}, \quad k = 1,2,\cdots,min\{M,n\}
\end{gather}$$
5. 连续函数 $$ EX = \int_{-\infty}^{+\infty} xf(x) dx $$
  - $X$ 在区间 $(a,b)$ 均匀分布,$\quad X \sim U(a,b)$$$\begin{gather}f(x)=\begin{cases}\frac{1}{b-a}, & a<x<b \\ 0, & 其他 \end{cases} \\ EX = \frac{a+b}{2} \end{gather}$$
  - $X$ 服从参数为 $\mu、\sigma^2$ 的正态分布，$\quad X \sim N(\mu,\sigma^2)$ $$\begin{gather} f(x) = \frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\ EX = \mu \end{gather}$$
  - $X$ 服从参数为 $\lambda$ 的指数分布，$\quad X \sim E(\lambda)$ $$\begin{gather}f(x)=\begin{cases}\lambda e^{-\lambda x}, & x > 0 \\ 0, & x \leq 0 \end{cases} \\ EX = \frac{1}{\lambda} \end{gather}$$
#### 4.2.2 *随机变量函数的数学期望*
##### *定理 1*
设离散型随机变量 $X$ 的分布律为 $P(X = x_i) = p_i \ (i = 1, 2, \cdots) ， y = g(x)$ 为已知的连续函数，如果级数 $\sum_{i} g(x_i)p_i$ 绝对收敛，则
$$EY = E[g(X)] = \sum_{i} g(x_i)p_i$$
##### *定理 2*
设连续型随机变量 $X$ 的概率密度为 $f(x) ， y = g(x)$ 为已知的连续函数，如果积分 $\int_{-\infty}^{+\infty} g(x)f(x)dx$ 绝对收敛，则
$$EY = E[g(X)] = \int_{-\infty}^{+\infty} g(x)f(x)dx$$
##### *定理 3*
设二维离散型随机变量 $(X, Y)$ 的联合分布律为 $P(X = x_i, Y = y_j) = p_{ij} \ (i, j = 1, 2, \cdots) ， z = g(x, y)  为已知的连续函数，如果级数 \sum_i \sum_j g(x_i, y_j)p_{ij}$ 绝对收敛，则
$$EZ = E[g(X, Y)] = \sum_i \sum_j g(x_i, y_j)p_{ij}$$
##### *定理 4*
设二维连续型随机变量 $(X, Y)$ 的联合概率密度为 $f(x, y) ， z = g(x, y)$ 为已知的连续函数，如果积分
$$\int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} g(x, y)f(x, y)dxdy$$
绝对收敛，则
$$EZ = E[g(X, Y)] = \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} g(x, y)f(x, y)dxdy$$
#### 4.1.3 *数学期望的性质*
$设 C 为常数，X，Y为随机变量$
1. $E(C)=CE(C)=C$
2. $E(CX)=CE(X)$
3. $E(X+Y)=E(X)+E(Y)（可推广到任意有限个随机变量）$
4. $若 X 与 Y 相互独立，则 E(XY)=E(X) \cdot E(Y)$
### 4.2 *方差*
#### 4.2.1 *随机变量方差定义*

用 $E(X - EX)$ 来度量随机变量可能取值与其均值 $EX$ 的偏离程度，但由于该表达式中带有绝对值，破坏了函数的一些分析性质，不便于数学运算。因此，为了方便起见，常用 $E(X - EX)^2$ 来度量，这就产生了随机变量方差的概念。

##### *定义*
1. 设 $X$ 是一随机变量，如果 $E(X - EX)^2$ 存在，则称之为随机变量 $X$ 的方差，记为 $D(X)$ 或 $DX$ ，即
   $$DX = E(X - EX)^2 \quad 称 \sqrt{DX}  为随机变量 X 的均方差或标准差。$$
2. 方差的统计意义，随机变量的方差 $DX$ 刻画了随机变量 $X$ 可能取值的集中程度：

    - $DX 越小，X 可能取值越集中（在 EX 附近）$
    - $DX 越大，X 可能取值越分散。$

3. 方差的计算公式，从方差的定义可以看出，随机变量的方差 $DX$ 是随机变量 $X$ 的函数的数学期望：

    - $当 X 是离散型随机变量时，其分布律为 P(X = x_i) = p_i \ (i = 1, 2, \cdots) 则 DX = \sum_{i}(x_i - EX)^2 p_i$
    - $当 X 为连续型随机变量时，其概率密度为 f(x) ，则 DX = \int_{-\infty}^{+\infty}(x - EX)^2 f(x) dx$

4. 方差的重要性质，根据方差的定义，结合随机变量数学期望的性质，容易得到：
    - $DX = EX^2 - (EX)^2$

5. 方差（记忆）
    - $两点分布 (0-1分布) \quad D(X) = p(1-p)$
    - $二项分布 X \sim B(n, p) \quad D(X) = np(1-p)$
    - $泊松分布 X \sim P(\lambda) \quad D(X) = \lambda$
    - $几何分布 X \sim Ge(p) \quad D(X) = \frac{1-p}{p^2}$
    - $均匀分布 X \sim U(a, b) \quad D(X) = \frac{(b-a)^2}{12}$
    - $指数分布 X \sim E(\lambda) \quad D(X) = \frac{1}{\lambda^2}$
    - $正态分布 X \sim N(\mu, \sigma^2) \quad D(X) = \sigma^2 \quad 特例 - 标准正态分布: 当 X \sim N(0,1)  时，E(X)=0, \quad D(X)=1$

#### 4.2.2 *随机变量函数方差性质*
$设 C 为常数，X，Y为随机变量$
1. $DC = 0$
2. $D(X + C) = DX$
3. $D(CX) = C^2 DX$ 
4. $D(X + Y) = DX + DY \quad 推广：设  X_1, X_2, \cdots, X_n 是相互独立的随机变量，则 D(X_1 + X_2 + \cdots + X_n) = DX_1 + DX_2 + \cdots + DX_n$
5. $DX = 0  的充要条件是 X 以概率 1 取常数 EX ，即 P(X = EX) = 1$
6. 重要定理
    - $X \sim B(n, p) ，则 X = X_1 + X_2 + \cdots + X_n ，其中 X_1, X_2, \cdots, X_n 相互独立且同服从参数为 p 的 0-1 分布$
    - $X_i \sim N(\mu_i, \sigma_i^2) \ (i = 1, 2, \cdots, n) ，且它们相互独立， c_1, c_2, \cdots, c_n 是不全为零的常数，则 c_1 X_1 + c_2 X_2 + \cdots + c_n X_n \sim N(c_1 \mu_1 + c_2 \mu_2 + \cdots + c_n \mu_n, c_1^2 \sigma_1^2 + c_2^2 \sigma_2^2 + \cdots + c_n^2 \sigma_n^2)$

### 4.3 *协方差与相关系数*
#### 4.3.1 *协方差定义和性质*
##### *定义1*
$设 X, Y 是随机变量，若 E[(X - EX)(Y - EY)]  存在，则称之为 X 与 Y 的协方差，记为  \text{cov}(X, Y)$ ，
$即 \text{cov}(X, Y) = E[(X - EX)(Y - EY)]$

由协方差的定义，结合随机变量数学期望的性质，容易得到：
$\text{cov}(X, Y) = E(XY) - EX \cdot EY \tag{4.3.2}$

##### *性质*
1. $\text{cov}(X, X) = DX$
2. $\text{cov}(X, Y) = \text{cov}(Y, X)$
3. $\text{cov}(aX + b, cY + d) = ac \cdot \text{cov}(X, Y) ，其中  a, b, c, d  为常数$
4. $\text{cov}(X_1 + X_2, Y) = \text{cov}(X_1, Y) + \text{cov}(X_2, Y)$
5. $D(X \pm Y) = DX + DY \pm 2\text{cov}(X, Y)$

#### 4.3.2 *相关系数定义和性质*
##### *定义2*
$设 X, Y 是随机变量，且  DX > 0, DY > 0$，则称

$\rho_{XY} = \frac{\text{cov}(X, Y)}{\sqrt{DX}\sqrt{DY}}$

$为  X  与  Y  的相关系数。$
##### *定理*
1. $设二维连续型随机变量  (X,Y) \sim N(\mu_1, \mu_2; \sigma_1^2, \sigma_2^2; \rho) ，则$ $\rho_{XY} = \rho$
2. $(Cauchy-Schwarz 不等式)设  X, Y  是随机变量，且  EX^2 < +\infty ， EY^2 < +\infty ，则$ $[E(XY)]^2 \leq EX^2 EY^2$
    性质：
     - $|\rho_{XY}| \leq 1$
     - $设随机变量  X, Y  相互独立且方差均大于零，则  \rho_{XY} = 0$
     - $|\rho_{XY}| = 1  的充要条件是存在常数  a (a \neq 0)  与  b ，使得  P(Y = aX + b) = 1$

#### 4.3.3 *不相关及条件*
相关系数 $\rho_{XY}$ 是用来刻画随机变量 $X, Y$ 之间线性关系强弱的数字特征。

- 其绝对值越大，$X, Y$ 之间的线性关系就越强。
- 其绝对值越小，$X, Y$ 之间的线性关系就越弱。

当 $X, Y$ 之间不存在线性关系时，有如下定义：
##### *定义3*
$设X, Y 是随机变量，若 \rho_{XY} = 0，则称 X 与 Y 不相关$
##### *定理*
1. $若随机变量X, Y 相互独立且方差均大于零，则 X, Y 不相关，但反之不然。$
2. $若二维连续型随机变量(X, Y) 服从二维正态分布，则 X, Y 不相关的充要条件是 X, Y 相互独立。$
3. $设X, Y 是方差均大于零的随机变量，则 X, Y 不相关的充要条件是以下任一条件成立：$
     1. $\text{cov}(X, Y) = 0$
     2. $E(XY) = EX \cdot EY$
     3. $D(X + Y) = DX + DY$
     4. $D(X + Y) = D(X - Y)$
### 4.4 *n维正态随机变量*
#### 4.4.1 *矩的概念*
##### *定义1* 
$设  X, Y  是随机变量。$
- $k 阶原点矩  E(X^k), \quad k = 1, 2, \cdots   \mu_k = EX^k$
- $k 阶中心矩  E(X - EX)^k, \quad k = 1, 2, \cdots   \nu_k = E(X - EX)^k$
- $(k+l) 阶混合矩  E(X^k Y^l), \quad k, l = 1, 2, \cdots   \mu_{kl} = E(X^k Y^l)$
- $(k+l) 阶混合中心矩  E[(X - EX)^k (Y - EY)^l], \quad k, l = 1, 2, \cdots   \nu_{kl} = E[(X - EX)^k (Y - EY)^l]$

一阶矩和二阶矩的重要关系：$EX^2 = DX + (EX)^2 \quad$
#### 4.4.2 *n维正态随机变量*
##### *定义2*
$设  \mathbf{X} = (X_1, X_2, \cdots, X_n)  为 n 维随机变量。$
- $均值向量： \mathbf{\mu} = (\mu_1, \mu_2, \cdots, \mu_n) ，其中  \mu_i = EX_i 。$
- 协方差矩阵：$$\mathbf{B} = (\sigma_{ij})_{n \times n} = \begin{bmatrix}
 \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1n} \\
 \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2n} \\
 \vdots & \vdots & \ddots & \vdots \\
 \sigma_{n1} & \sigma_{n2} & \cdots & \sigma_{nn}
 \end{bmatrix}$$
该矩阵为对称矩阵。
##### *定义3*
$设  \mathbf{X} = (X_1, X_2, \cdots, X_n)  的均值向量为  \mathbf{\mu} ，协方差矩阵为  \mathbf{B} ，若其联合概率密度为：$

$f(\mathbf{x}) = \frac{1}{(2\pi)^{\frac{n}{2}} |\mathbf{B}|^{\frac{1}{2}}} e^{-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})\mathbf{B}^{-1}(\mathbf{x} - \mathbf{\mu})^T}, \quad x_i \in R$

$则称  \mathbf{X}  服从 n 维正态分布，记为  \mathbf{X} \sim N(\mathbf{\mu}, \mathbf{B}) 。$

##### *性质*
1. 分量性质：$n 维正态分布的每一个分量都是正态随机变量。反之，若各分量相互独立且均为正态，则联合分布为 n 维正态分布。$
2. 线性组合判定： $\mathbf{X}  服从 n 维正态分布的充要条件是  \mathbf{X}  的任意非零线性组合  c_1 X_1 + c_2 X_2 + \cdots + c_n X_n  服从一维正态分布。$
3. 线性变换不变性：$n 维正态随机变量的线性变换仍服从多维正态分布。$
4. 边缘分布：$n 维正态随机变量的任意 m (m < n) 个分量构成的 m 维随机变量服从 m 维正态分布。$
5. 独立与不相关等价：$对于 n 维正态分布，分量  X_1, X_2, \cdots, X_n  相互独立的充要条件是它们两两不相关。$