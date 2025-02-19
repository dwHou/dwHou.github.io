<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=1455514465&bvid=BV1zi421v7Dr&cid=1575395206&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

### 铁口直断

基于Splatting和机器学习的三维重建方法

#### 1.1 什么是splatting

定义：

- 一种体渲染的方法：从3D物体渲染到2D平面
- Ray-casting是被动的（NeRF）
  - 计算出每个像素点受到发光粒子的影响来生成图像
- Splatting是主动的
  - 计算出每个发光粒子如何影响像素点

为什么叫splatting：

- Splat：拟声词，【啪唧一声】
- 类似于像墙面扔雪球的声音和过程
- 每扔一个雪球，墙面上会有扩散痕迹，足迹（footprint）
- 所以这个算法也称为抛雪球算法，或足迹法。
- 有将splatting翻译成喷溅也很有灵性。

splatting的核心：

1. 选择【雪球】
2. 抛掷雪球：从3D投影到2D，得到足迹
3. 加以合成，形成最后的图像

为什么使用核（雪球）：

- 点是没有体积的
- 需要给点一个核，来进行膨胀
- 核可以是 高斯（3D高斯椭球）/圆/正方体

为什么选择3D高斯椭球？

- 很好的数学性质：
  - 仿射变换后高斯核仍然闭合
  - 3D降维到2D后（沿着某一个轴积分）
  - 依然为高斯

- 定义：
- 椭球高斯 $G(x) = \frac{1}{\sqrt{(2\pi)^k|\sum|}}e^{-\frac{1}{2}(x-\mu)^{T}\sum^{-1}(x-\mu)}$
- 这就是我们常见的高斯分布的公式（$f(x)=\frac{1}{\sqrt{2\pi}\delta}e^{-\frac{(x-\mu)^2}{2\delta^2}}$），只是在高维时方差变为了协方差矩阵。
- $\sum$表示协方差矩阵，半正定，$|\sum|$是其行列式

3D高斯为什么是椭球？

- 椭球面：

  - (0,0,0)为中心：$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$
  - $Ax^2+By^2+Cz^2+2Dxy+2Exz+2Fyz=1$

- 3D高斯

  - 高斯分布：
    1. 一维：均值 & 方差
    2. 高维：均值 & 协方差矩阵

  - 协方差矩阵

    $\begin{bmatrix}
    \delta{^2_x} & \delta_{xy} & \delta_{xz} \\ 
    \delta_{yx} & \delta{^2_y} & \delta_{yz} \\ 
    \delta_{zx} & \delta_{zy} & \delta{^2_z}\end{bmatrix}$

    1. 是一个对称矩形，决定高斯分布形状

    2. 对角线上元素为x轴/y轴/z轴的方差

    3. 反斜对角线上的值为协方差

       表示x和y，x和z...的线性相关程度

  - $constant = {-\frac{1}{2}(x-\mu)^{T}\sum^{-1}(x-\mu)}$ 

    展开可转化为$ Ax^2+By^2+Cz^2+2Dxy+2Exz+2Fyz=1$

  - 说明：

    - ${-\frac{1}{2}(x-\mu)^{T}\sum^{-1}(x-\mu)} = constant$，定义了一个椭球面
    - $\therefore G(x;\mu, \sum) = [0,1]$ 定义了一个大椭球壳 套 小椭球壳。
    - 具体推导，会发现是实心的椭球

- 各向同性 & 各向异性

  - 各向同性
    1. 在所有方向具有相同的扩散程度（梯度）
    2. 球
    3. $\begin{bmatrix}
       \delta{^2} & 0 & 0 \\ 
       0 & \delta{^2} & 0 \\ 
       0 & 0 & \delta{^2}\end{bmatrix}$
  - 各向异性
    1. 在不同方向具有不同的扩散程度（梯度）
    2. 椭球
    3. $\begin{bmatrix}
       \delta{^2_x} & \delta_{xy} & \delta_{xz} \\ 
       \delta_{yx} & \delta{^2_y} & \delta_{yz} \\ 
       \delta_{zx} & \delta_{zy} & \delta{^2_z}\end{bmatrix}$

  

- 协方差矩阵如何控制椭球形状的
  - 高斯分布的仿射变换：
    - $w = Ax + b$
    - $w \sim N(A\mu+b, A\sum{A^T})$
  - 任意高斯可以看作是标准高斯通过仿射变换得到的$\sum = A \cdot I \cdot A^T$
- 协方差矩阵怎么就能用旋转和缩放矩阵表达？
  - <font color="green">$A = RS$</font>，仿射变换就是旋转+缩放+平移，来完成。A来表示旋转和缩放，b是平移。
  - $\sum = A \cdot I \cdot A^T = R \cdot S \cdot I \cdot (R \cdot S)^T = R \cdot S \cdot S^T \cdot R^T$
  - $\therefore$ 协方差矩阵可以用旋转和缩放矩阵表示
- 感觉可以回头再补一补大学《线性代数》和《概率论》，这俩估计半个月就能回顾好。最多一个月，还能再做些题目。

#### 1.2 如何进行参数估计