[MIT 6.S184](https://youtu.be/GCoP2w-Cqtg?si=uDdnmfEIZhUb1jxc)

这门课：Flow/Diffusion模型的理论与实践

- 理论：第一性原理，必要而最少量的数学知识 ODE、SDE
- 实践：如何实现

## 第一章 利用随机微分方程的Gen AI

### 第一节：从生成到采样

我们将图像/视频/蛋白质表示为向量

 $z_{图像}\in R^{H \times R \times 3} $，$z_{视频}\in R^{T \times H \times R \times 3} $，$z_{分子结构}\in R^{N \times 3} $（N个原子有3坐标）。

一张图像的“好”程度 ≈ 它在数据分布下的可能性有多高

> 学术一点的说法：图像的质量可以近似等同于它在数据分布中的似然性

生成就是从数据分布中采样

数据分布一般用概率密度函数$p_{data}$表示。

数据集包含了数据分布中有限个数的样本：$z_1,...,z_N \sim p_{data}$

条件生成指的是从条件分布中采样：$z \sim p_{data}(\cdot \mid y)$，比如 $y$ 是提示词。意味着给定这个提示，数据的分布是什么。这是我们最感兴趣的课题。

生成模型将初始分布（例如高斯分布）中的样本转换为数据分布中的样本。

$x \sim p_{init}$  ➡️  $Generative Model$  ➡️  $z \sim p_{data}$



### 第二节 流模型与扩散模型

#### 2.1 流模型

流的基本对象：轨迹（Trajectory）、矢量场（Vector Field）、常微分方程（ODE）

轨迹：   $X[0,1] \to R^d, t \to X_t $

矢量场：$u$：$R^d \times [0,1] \to R^d, 给定(x, t) \to u_t(x)$

常微分方程：描述轨迹上的条件

$X_o = x_o (初始条件)$ 沿着矢量场指定的方向前进。

$\frac{d}{dt}X_t = u_t{X_t}$ （ODE）

轨迹的导数或速度 是由 $X_t$ 当前所在位置的矢量 $u_t(X_t)$ 给出的。

> [!TIP]
>
> 也许我们中的一些人听说过ODE在工程和物理学中是力学的基础。但“<font color="brown">流</font>”这个术语不太常见。流是遵循ODE的轨迹的集合。
>
> <font color="brown">本质上是我们收集大量针对不同初始条件的解决方案，然后将它们全部收集到一个函数中，并称之为流。</font>

流：$\phi : R^d（空间） \times [0,1]（时间分量） \to R^d$，

$给定(x_0, t) \to（映射） \phi_t(x_0)$，对于每个初始条件$x_0$，我希望它是我的ODE的解。

$\phi_0(x_0) = x_0$

$\frac{d}{dt}\phi_t(x_0) = u_T(\phi_t(x_0))$

所以：

- ODE由矢量场（VF）定义。
- 轨迹是ODE的解法。
- 流则是各种初始条件的轨迹的集合。



