## 资源

https://3dgstutorial.github.io/ 

ppt: https://3dgstutorial.github.io/

[3DGS (3D Gaussian Splatting) 原理介绍](https://www.bilibili.com/video/BV1o4UUYbEdW/?share_source=copy_web&vd_source=f26f3b6f5d88e251cea5e8fa6c584e4a)

https://github.com/MrNeRF/awesome-3D-gaussian-splatting

3dgs历程：https://www.youtube.com/watch?v=DjOqkVIlEGY

精读：https://learnopencv.com/3d-gaussian-splatting/ 配套代码：https://github.com/spmallick/learnopencv/tree/master/3D-Gaussian-Splatting-Code/

## 3D GS原理理解

### NeRF到3D GS的演变

#### 简要回顾NeRF原理

- 由相机内参c2w和随机采样的像素pixels确定一条射线$r(t)=o+td$，沿射线采样
- 将采样点$x_i$的位置信息$(x, y, z)$和视角方向信息$(\theta, \varphi)$编码
- 输入神经网络【隐式辐射场】中预测采样点$x_i$的颜色与体密度信息
- 通过体渲染得到沿该射线$r(t)$可观测的最终颜色
- 根据渲染图与真实图的误差，梯度下降更新，优化神经网络参数

#### Point-NeRF：NeRF与<font color="brown">点云</font>的初遇

1. 神经点云：

   为每个点赋予**神经特征**，该特征代表对周围局部3D场景几何和外观的编码，CNN提取。

2. 体渲染过程：

   射线采样点周边搜索半径r内的点的**特征融合**，得到的特征解码得到采样点的颜色和体密度。

3. 自适应点云的生长和剔除：

   在射线具有**最高不透明度的位置**且周围没有点的情况下添加新点；每10k迭代，基于点位于物体表面的概率，设定阈值剔除无用点

#### Plenoxels：无神经网络NeRF+球谐函数

基于稀疏体素网格表示场景，每个体素格点存储体密度和球谐函数系数，

射线采样点的颜色值和体密度，由其所在体素网格的八个格点 三线性插值计算。

**球谐函数**:

简称SH函数，可以看成球坐标系中的傅里叶级数，可用于拟合球面分布，例如光强，记录空间中某个点从不同方向看过去的不同颜色。阶数越高，拟合能力越强。

#### 3D Gaussian Splatting的简要介绍

3D Gaussian Splatting（3D高斯泼溅）

使用大量的3D高斯基元来表示场景，可视作<font color="brown">参数化点云</font>，每个基元有位置，大小，方向，颜色信息（球谐函数系数）和不透明度等信息。

渲染速度有很大优势：

<img src="../../../images/typora-images/image-20250618163027486.png" alt="image-20250618163027486" style="zoom:50%;" />



### 3D GS原理介绍

#### 3D GS框架图介绍

<img src="../../../images/typora-images/image-20250618171234127.png" alt="image-20250618171234127" style="zoom:50%;" />

3D GS高斯椭球形状为什么受协方差矩阵控制？

3D GS高斯椭球如何从3D空间投影到2D空间？

3D GS基于图像块（Tile）的光栅化图像渲染

#### 3D GS概览

3D 高斯点渲染（3D Gaussian Splatting）提出了一种基于光栅化（rasterization-based）的重建辐射场的方法。该方法既能得到高品质的重建效果，又具有显著的速度和性能优势，甚至适用于移动设备。我们不再需要在质量与速度之间做妥协，这一事实促使从业者构建大量工具和应用。3DGS（3D Gaussian Splatting）也已被集成到 Unity、Unreal、three.js 和 NeRFStudio 等框架中。此外，利用 3DGS 展开的大量研究已经发表，涵盖不同主题：动态与人体重建、SLAM、三维生成模型（3D Generative Models）等许多方向。在本教程中，我们将首先解释 3DGS 及其为何迅速风靡研究界，接着给出在研究场景中使用若干有用工具的实用建议，并讨论该研究领域的进展。

> [!NOTE]
>
> **3D Gaussian Splatting（3DGS）**：想象把三维空间里的物体用大量小的、带有模糊边界的“高斯云点”（每个点像小模糊球）表示。渲染时把这些云点投影到图像上并混合，得到最终画面。
>
> **基于光栅化（rasterization-based）**：光栅化是一种从几何（点、线、三角形）快速生成像素图像的传统渲染方法，常用于实时渲染（比如游戏）。这里意味着 3DGS 使用类似游戏中那种把东西快速变成像素的方法来做重建，而不是更慢的逐光线跟踪方法。
>
> **重建辐射场（radiance field）**：辐射场可以理解为“从任意位置和方向看，场景应当发出怎样的光”，也就是能合成新视角图像的隐式场表示。NeRF（神经辐射场）就是这类概念的典型代表。
>
> **整体意思**：这句说的是 3DGS 提出用“高斯云点 + 光栅化”的方式来重建那种能合成新视角图像的光照／颜色场。

<img src="assets/colmap_nerf_3dgs_compare.webp" alt="colmap_nerf_3dgs" style="zoom:80%;" />

> [!TIP]
>
> **传统光栅化**：会计算每个三角形在屏幕上的位置（即透视投影），确定其覆盖的像素范围，并为这些像素记录颜色、深度等属性，存入 Z-buffer。当新的三角形绘制到同一像素时，比较深度值，只保留距离相机更近的那个。最终，再根据 Z-buffer 中的光照、材质、法线等信息计算每个像素的最终颜色。
>
> **3DGS 中的光栅化：**同样会计算每个高斯体在屏幕上的位置（即透视投影），但 3DGS 不使用传统的“硬性遮挡”式 Z-buffer。由于高斯体是半透明、具有体积感的元素，多个高斯在重叠时，其颜色会**按深度连续混合（blending）**，而不是“前者覆盖后者”。渲染时，3DGS 会根据每个高斯的颜色、透明度和深度等属性，对它们进行深度排序与 α 混合，逐层累积，得到最终像素颜色。
>
> 简而言之，传统光栅化强调表面可见性（只渲染最前方的几何），而 3DGS 的光栅化强调体积累积效应（所有高斯对最终颜色都有贡献）。与传统渲染依赖“材质 + 灯光”模型不同，3DGS 直接学习场景的整体外观，也就是辐射场本身。基础版的 3DGS 不需要显式的光照、材质或法线信息，因为大多数数据集的光照条件是固定的；不过，一些扩展变体会在此基础上进一步显式建模光照或表面属性。

对于熟悉传统计算机图形学的人来说，NeRF 从概念上更接近光线追踪（ray tracing）技术，而 3DGS 概念上更接近光栅化（rasterization）技术。前者backward mapping，基于光线追踪的体积采样。后者forward mapping，一批高斯体进行 $\alpha$混合，基于光栅化的体积投影。 

<img src="assets/image-20251030105414395-1792859.png" alt="image-20251030105414395" style="zoom:80%;" />

#### 3D GS技术解读

一个三维高斯（3D Gaussian）由一个三维协方差矩阵 $\Sigma$ 定义，并以点（均值）$\mu$ 为中心。

可学习的参数包括：

1. 均值（Mean）
2. 各向异性协方差（Anisotropic Covariance）
3. 不透明度（Opacity）
4. 球谐系数（Spherical Harmonic, SH coefficients）

深入理解 3D 高斯溅射

3D 高斯溅射是一种**光栅化（rasterization）技术**，它使用数百万个高斯分布来表示场景，而不是使用三角形。

主要步骤包括：

#####  **步骤 1：初始化（Initialization）**

无论我们从运动结构中找到什么点云，我们用它作为初始化。我们所做的就是使高斯的均值等于点的坐标。

#####  **步骤 2：优化（Optimization）**

自适应密度控制（Adaptive Density Control）——动态调节高斯的数量与分布密度。

优化目标：

- 控制高斯数量，防止密度过高；
- 自适应地删除冗余高斯或分裂新高斯；
- 保持视觉质量不变的情况下优化渲染效率。

它本质上是一个 **密度正则化 + 动态采样平衡** 过程。

具体操作：

1. Pruning（剪枝）：移除不重要或冗余的高斯。

2.  Densification（增密）：在场景细节不足或几何结构复杂区域**增加高斯数量**。分为克隆和拆分两种方式。

   

   **触发条件**

   - 欠重建：局部渲染误差或该高斯的梯度（对 loss 的贡献）持续较高 → 标记为需要克隆。
   - 过重建：单个高斯的空间尺度（variance）远大于邻域平均、或该高斯在多个视图投影时覆盖范围异常、或边界贡献下降 → 标记为需要拆分。
   - 剪枝：opacity 很低且梯度/贡献几乎为零，或长期不被采样/不贡献像素 → 删除。

   **实现细节**

   - Clone（克隆）：复制参数（position, scale, orientation, color, alpha），给子副本加小扰动（位置 jitter、尺度小缩放、颜色微扰），在之后的优化中允许它们独立更新。
   - Split（拆分）：把一个高斯分成 2（或更多）个子高斯，通常子高斯初始继承父参数但把尺度调小、位置做双向小偏移；也可以把父的权重按比例分配到子高斯上。
   - Prune（剪枝）：安全地删除前请做平滑策略（比如标记、延迟删除若干步，以免误删短暂不贡献的高斯）。

> [!NOTE]
>
> 这儿自适应密度控制的优化方法，属于一种Heuristic（启发式方法） = 一种帮助我们发现或解决问题的经验性方法。
>
> 它并不保证最优解、完美性或严格证明的正确性，
>  但在复杂、搜索空间巨大或无解析解的情况下，
>  **能“比较快地找到一个够好的解”。**



#####  **步骤 3：光栅化（Rasterization）**

得到渲染图像，可以用于和GT计算损失。

3D立体图形（Gaussians）如何投影到2D平面（camera plane）上？

3D高斯泼溅投影 取自 《EWA volume splatting》这篇文章。

> [!NOTE]
>
> "Gaussians are closed under affine mappings and convolution, and integrating a 3D Gaussian along one coordinate axis results in a 2D Gaussian."  —— EWA （Elliptical Weighted Average） volume splatting
>
> “高斯函数在仿射变换和卷积运算下是封闭的，并且将一个三维高斯在某一坐标轴上积分后，得到的结果仍然是一个二维高斯。” —— EWA（基于椭圆加权平均的）体积溅射

这体现了高斯分布在积分/投影运算下的**封闭性（closure property）**：
 积分或投影不会破坏其高斯形态，只是维度降低。这个性质非常重要：

- 每个 3D 高斯在被投影到 2D 图像平面时（相当于沿视线方向积分）会变成一个 **2D 高斯**；
- 因此可以用解析的二维高斯来表示它在屏幕上的光照/颜色分布，而不必进行数值采样。

👉 这就是 3D Gaussian Splatting 能高效渲染的理论基础之一。

> 高斯分布的有趣之处，在于它们在数学上如此优美。通常，无论你对高斯函数做什么，最终得到的仍然是高斯函数。

<img src="assets/image-20251104134410841-2235053.png" alt="image-20251104134410841" style="zoom:50%;" />

###### 步骤3.1 世界坐标系 到 相机坐标系

首先，从物体坐标 通过视图转换，本质上是一次仿射变换（乘旋转分量，加平移分量），转换到相机坐标下。

<img src="assets/image-20251104140259914.png" alt="image-20251104140259914" style="zoom:50%;" />

高斯分布就是由均值（位置）和协方差（分布）表示。

均值（$p_w \to p_c$）和协方差（$\sum \to \sum_c$）就都转换过来了。最后的公式很简单：

① <font color="brown">$p_c = R_{cw}p_w + t_{cw}$</font>

② <font color="brown">$\sum_c = R_{cw}\sum R_{cw}^T$</font>，这儿都是3x3的矩阵。

###### 步骤3.2 相机坐标系 到 光线坐标系

在相机坐标系（Camera Space）下，每个点 $p_c = (x, y, z)$ 都是在相机前方的三维空间中。
 渲染时，我们要把这个点投影到屏幕上的某个像素（也就是光线方向）。

为了更好地表示这种沿视线方向的投影过程，我们引入了一个中间坐标系——**Ray Space（光线空间）**

它的设计目的是：

- 把一个三维高斯的空间分布分解为：

  沿光线方向的分量；

  在屏幕平面上（垂直于光线）的分量；

- 这样可在光线方向上解析积分，得到投影后的  2D 协方差。

从相机空间到光线空间，是一个非线性变换（non-affine transformation）。



 **步骤 4：损失计算与更新（Loss Calculation & Updation）**

计算渲染图像和GT之间的损失，根据损失计算梯度，然后

![3dgs-pipeline](assets/3dgs_pipeline_v3-1.webp)

