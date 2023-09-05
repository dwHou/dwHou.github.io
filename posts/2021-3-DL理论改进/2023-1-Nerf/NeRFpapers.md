### NeRF paper list

#### :page_with_curl:Local Deformation Fields

单位：CMU

从2D视频获取可控的<font color="brown">3D头部化身</font>十分有价值。而神经隐式场为具有个性化形状、表情和面部部位（例如头发和嘴巴内部）的 3D 头部化身建模提供了强大的表示，这超出了线性 3D 可变形模型 (3DMM)。

现有方法：

- 没有对具有精细面部特征的面部进行建模

  > 基于参数网格表示（parametric mesh representation） 3DMM的传统/神经重建流程是高效、可控的，并且可以很好地集成到图形流程中，但代价是缺乏
  >
  > 1. 细粒度的脸部特征
  > 2. 头发、眼睛和嘴巴内部等重要的脸部特征。
  > 3. 非脸部特征，如帽子/眼镜/饰品

- 没有对面部部位进行局部控制，从而从单眼视频中推断出不对称的表情。 

- 大多数条件仅依赖于局部性较差的 3DMM 参数，并使用全局神经场来解析局部特征。 

  > 线性 3DMM 受到全局或大规模表达分解的限制，因此在更精细的水平上控制局部变形相对困难，例如眨眼时眼睛周围形成的皱纹。
  >
  > 从线性 3DMM 衍生的隐式场有同样的问题。

本文方法：

我们建立基于部分的隐式形状模型，将全局变形场分解为*局部变形场。

> *局部 ：有着局部语义，绑定式（rig-like）的控制。
>
> 全局：现有方法。估计指的这篇Nerfies: Deformable neural radiance fields.

效果：呈现出更清晰的局部可控非线性变形，特别是嘴部内部、不对称表情和面部细节。

具体方法：

step 1.使用DECA（a 3DMM face tracker）获得每一帧的表情参数

step 2.稀疏的3D face landmarks

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230830225520580.png" alt="image-20230830225520580" style="zoom:50%;" />

step 3.局部变形场由一定半径内的非线性函数表示，并由的step1 tracked 表情参数调节。（Attention Mask Based on 3DMM Bases：表情参数会有<font color="brown">注意力</font>掩模加权，来过滤对landmark不影响的冗余参数）

$A_l$:

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230830224505199.png" alt="image-20230830224505199" style="zoom:50%;" />

$t$ （合） and $t_l$ （分）

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230830230458466.png" alt="image-20230830230458466" style="zoom:50%;" />

$\gamma$是位置编码(PE)

> Fourier features let networks learn high frequency functions in low dimen- sional domains.
>
> sinusoidal positional encodings：<font color="brown">平移不变性，单调性和对称性</font>

$T = D(\gamma(x_{obs}),e_i,p_i,\omega_{i}^{p})$,  $\omega_{i}^{p}$是姿态的不准确度

$\sigma, c = F(x_{can},d,\omega_{i}^{a})$，$\omega_{i}^{a}$是外观的不一致程度，$d$是视角方向。

体积渲染对已经求解（采样）的体素中的$\sigma, c$（密度和颜色）进行插值。通过对所有体素进行插值，可以得到整个场景的颜色和密度值。

<img src="../../../images/typora-images/image-20230725152329052.png" alt="image-20230725152329052" style="zoom:50%;" />

$d : (x,y,z,\theta, \phi)$ （每一束光线）

$x_{can}: voxel$

由于面部表情变形通常是非刚性的，我们将 T 预测为三维空间的平移。

step 4. 最后，局部变形将基于距离的权重相加 $\sum_{l}{W(x_{obs}-c_l)t_l}$，用于将全局点变形到<font color="brown">规范空间</font>，并检索用于体积渲染的辐射率（$c$）和密度（$\sigma$）。

> 图释：左门牙下方从$x_{obs}$到$x_{can}$
>
> 规范空间（canonical space）：中立表情，或说不带表情。规范空间的含义要根据上下文判断，比如在驱动人体的工作中，分为posed space和canonical space。这里posed space就是带有一定动作的人体模型，canonical space就是不带任何动作的人体模型。

——————————————————————————————————————

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230830225132467.png" alt="image-20230830225132467" style="zoom:50%;" />

方法概述：给定输入视频序列，我们运行DECA来获取每帧线性 3DMM的以下参数：表情 $e_i$、姿势 $pose_i$ 和稀疏 3D landmarks $c_l$。step3中具有局部空间支持的注意力掩膜也是从 3DMM 预先计算的。 

我们将动态变形建模为每个观察点 $x_{obs}$ 到规范空间的平移，表示为 $x_{can}$ （第 3.1 节）。 我们将全局变形场 $t$ 分解为多个局部场 $\{tl\}$，每个局部场以代表性的landmark $c_l$ 为中心（第 3.2 节）。 我们通过调节 $e_i$ 的注意力掩码 $A_l$ 来强制每个局部场$t_l$ 的稀疏性（第 3.3 节）。 我们的隐式表示是使用 RGB 信息、几何正则化和先验以及新颖的局部控制损失来学习的（第 3.4 节）。

> 原本NeRF用于静态物体，就是<font color="brown">五维照射野预测图片</font>。光沿直线传播，经过采样的a、b、c体素。渲染方程用于计算从相机位置和方向发出的光线与场景中对象的相交点，并确定每个点的颜色和密度值。
>
> 但用于Avatar时，物体有了表情、姿态，要将公式改造、带入到规范空间进行（光仍经过a、b、c体素，但这里实际a、b、c不再处于一条直线了）。

> 总结：Decomposing a global deformation field into local ones based on attention mask of 3DMM bases leads to great facial details.

#### :page_with_curl:PointAvatar

单位：ETH

现有方法：

当前的方法要么建立在显式 3D 可变形网格 (3DMM) 上，要么利用神经隐式表示。 前者受到固定分类法的限制，而后者变形困难且渲染效率低下。

> **3DMM:** 
>
> a-priori fixed topologies, and limited to surface-like geometries
>
> **neural implicit representations:** 
>
> inefficient since rendering a single pixel requires querying many points along the camera ray

此外，现有的方法纠缠了光照和颜色估计，因此它们在新环境中重新渲染化身方面受到限制。

本文方法：

相比之下，我们提出了 PointAvatar，一种可变形的基于点的表示（<font color="blue">点云</font>），它将源颜色分解（**disentangles**）为固有反照率（albedo）和法线相关的阴影。 我们证明 PointAvatar （**bridges the gap**）<font color="brown">弥合了现有网格表示和隐式表示之间的差距</font>，将高质量的几何和外观与拓扑灵活性、易于变形和渲染效率相结合。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230831102921464.png" alt="image-20230831102921464" style="zoom:50%;" />

> 总结：An oriented point cloud representation bridges the gap between existing mesh- and implicit representations.

#### :page_with_curl:ER-NeRF

单位：Beihang

现有方法：普通（vanilla） NeRF速度都比较慢，没法满足实时要求；最近几项关于高效神经表示的工作已经证明，通过用稀疏特征网格替换部分 MLP 网络，比普通 NeRF 获得了巨大的加速：

<font color="brown">Instant-NGP</font>引入了用于静态场景建模的哈希编码体素网格，允许通过紧凑的模型实现快速和高质量的渲染。 <font color="brown">RAD-NeRF</font>首先将该技术应用于语音肖像合成，并构建了SOTA的实时框架。 然而，这种方法需要一个复杂的基于 MLP 的网格编码器来隐式学习区域的 音频-运动 映射，这限制了其收敛和重建质量。

#### :page_with_curl:Instant-NGP