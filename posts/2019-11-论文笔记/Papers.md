### 李沐团队带读系列

#### Paper-reading-principle

Pass1: 读标题，摘要，结论，方法、实验的图表 —— 得出文章是否适合自己，要不要读。Pass2: 从头到尾读。Pass3: 精读，仿佛自己从头到尾在做这个研究。

### 沈向洋带读系列

#### Ten Questions

论文十问由沈向洋博士提出，鼓励大家带着这十个问题去阅读论文，用有用的信息构建认知模型。写出自己的十问回答，还有机会在当前页面展示哦。

**Q1**论文试图解决什么问题？

**Q2**这是否是一个新的问题？

**Q3**这篇文章要验证一个什么科学假设？

**Q4**有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

**Q5**论文中提到的解决方案之关键是什么？

**Q6**论文中的实验是如何设计的？

**Q7**用于定量评估的数据集是什么？代码有没有开源？

**Q8**论文中的实验及结果有没有很好地支持需要验证的科学假设？

**Q9**这篇论文到底有什么贡献？

**Q10**下一步呢？有什么工作可以继续深入？

#### :page_with_curl: Video imprint

构造并学习一个更通用的视频时空表征。

#### :page_with_curl: Swin Transformer

**分析现有方法：**

Transformer本来是处理文本的，不需要处理不同尺度的信息的问题。

ViT对Transformer的改造很简单直接，但没考虑视觉信号的一些特点。

主要有以下挑战：

- 视觉实体变化大，在不同场景下视觉Transformer性能未必很好（所以ViT只能做做分类问题，不太适合检测、分割）
- 图像分辨率高，像素点多，Transformer基于全局自注意力的计算导致计算量较大

本工作就是希望Transformer能更好的适应视觉信号（层次性hierarchy/局部性locality/平移不变性translation invarianc）

<img src="../../images/typora-images/image-20221020174023945.png" alt="image-20221020174023945" style="zoom:50%;" />

针对现有问题，提出了一种**包含滑窗操作，具有层级设计**的Swin Transformer。

**本文方法：**

用shifting window而不是sliding window。

### Low-level Vision

#### :page_with_curl: Multi-Stage Progressive Image Restoration

[MPRNet](https://arxiv.org/pdf/1803.05407.pdf)

CVPR 2021 *Inception Institute of AI, UAE*

**分析现有方法：**

1. encoder-decoder结构 ：<b>优：</b>更宽的上下文信息，<b>劣：</b>保持细节不足。
2. single-scale pipeline ：<b>优：</b>保持空间上的精确细节，<b>劣：</b>语义上不太可靠。

**该文观点：**

1. 在multi-stage结构中，结合encoder-decoder与single-scale pipeline，是必要的。
2. multi-stage不仅仅是上一阶段的输出，也作为下一阶段的输入。
3. 每一阶段都用上监督信息是很重要的，渐进式学习。由此，设计supervised attention module (SAM)模块。
4. 该文提出了将上一阶段特征(contextualized)传递给下一阶段。由此，设计cross-stage feature fusion (CSFF)方法。

**网络结构：**

<img src="MPRNet.png" style="width:80%; height: 80%;">



#### :page_with_curl: Series-Parallel Lookup Tables

[SPLUT]()

ECCV 2022 *Tsinghua University*

**分析现有方法：**

1. LUT通过快速内存访问代替耗时的计算，具有实用性。
2. 但是大多数现有的基于 LUT 的方法只有一层 LUT。 如果使用 n 维 LUT并且用于查询v个可能值，则 LUT 的尺寸有 v^n。 因此，通常将 v 和 n 设置为较小的值以避免无法承受的大 LUT，这严重限制了效果。



#### :page_with_curl: RAISR: Rapid and Accurate Image Super Resolution

[RAISR]()

TCI 2016 *Google*

**分析现有方法：**

1. 传统插值方法，是内容无关的线性方法，表达能力不足。
1. 全局滤波效果不如大量参数且非线性的神经网络；所以改全局为图像块内容自适应。

**该文观点：**

<img src="../../images/typora-images/image-20220926211611036.png" alt="image-20220926211611036" style="zoom:50%;" />

1. 倾向example-based方法，即使用外部数据集学习LR patch到HR patch的映射。

2. RAISR 背后的核心思想是通过在图像块上应用一组预先学习的过滤器来提高画质，这些过滤器由高效的散列机制选择。

   1. 过滤器是基于 LR 和 HR 训练图像块对学习的
   2. 哈希是通过估计局部梯度的统计信息来完成的。

3. 过程：

   1. 插值上采样或pixelshuffle机制，参考：edge–SR: Super–Resolution For The Masses

   2. 局部梯度的函数得到{角度, 强度, 连贯性}3个哈希表的键。最终经过散列计算得到索引值（散列意味着冲突尽可能少）

      哈希表键是局部梯度的[函数](https://www.academia.edu/download/42362012/feng_asilomar.pdf)，哈希表条目是相应的预学习过滤器。比“昂贵”的聚类（例如 K-means、GMM、字典学习）更加高效。

      > 假如每桶有n个filter，那么可设置index = f(angle, strength, coherence)的散列函数，在f(0,0,0)=0，f(1,1,1)=n。
      >
      > 我们有trick避免冲突。

   3. 如若上采样+滤波，滤波器组只需一个(但尺寸要求较大)。如若滤波+pixelshuffle，滤波器组需要上采样倍数^2个。

      1. 上采样+滤波，会更复杂的有Pixel-Type，这是由于插值方法的特性（像素的来源不同）导致的。
      2. 滤波+pixelshuffle，不需要Pixel-Type，或者说Pixel-Type隐含在了我们有4组滤波器，即4种Type。

   4. 所有滤波器权重均由学习得到。

4. 优化点：

   1. 考虑许多上采样kernel中心对称的特性。可以减少一半冗余。
   2. 考虑使用量化。RAISR只有单层，不存在误差累积。



#### :page_with_curl: BLADE: Best Linear Adaptive Enhancement

**分析现有方法：**

> 深度学习：DL methods can <font color="red">trade</font> quality <font color="red">vs.</font> computation time and memory costs through considered choice of network architecture.
>
> Deep networks are hard to analyze, however, which makes failures challenging to diagnose and fix.
>
> These problems motivate us to take a lightweight, yet effective approach that is <font color="red">trainable</font> but still <font color="red">computationally simple</font> and <font color="red">interpretable</font>.

相关工作：

A Deep Convolutional Neural Network with Selection Units for Super-Resolution

**本文方法：**

我们的方法可以看作是一个浅层的双层网络，其中第一层是预先确定的，第二层是经过训练的。 我们表明，这种简单的网络结构允许推理计算效率高、易于训练、解释，并且足够灵活以在广泛的任务中表现良好。

**滤波器选择：**

我们发现结构张量分析（structure tensor analysis）是一个特别好的选择：它是稳健的、相当有效的，并且适用于我们测试过的一系列任务。 结构张量分析是局部梯度的主成分分析（PCA）

**图像结构张量：**

From the eigensystem, we define the features:

- <font color="purple">$\text{orientation} = \arctan w_2/w_1$</font>, is the predominant local orientation of the gradient;

- <font color="purple">$\text{strength} = \sqrt{\lambda_1}$</font>, is the local gradient magnitude; and

- <font color="purple">$\text{coherence} = \frac{\sqrt{\lambda_1} - \sqrt{\lambda_2}}{\sqrt{\lambda_1} + \sqrt{\lambda_2}}$</font>, which characterizes the amount of anisotropy in the local structure.



**阅读评价：**

和eSR-MAX某种意义上是等价，“This spatially-adaptive filtering is equivalent to passing image through a linear filterbank and then for each spatial position selecting one filter output”，但提前逐像素进行了模板选择，减少了计算资源的浪费。这篇文章提到，“three-dimensional index”，和我的思路是不谋而合的。



#### :page_with_curl: Multiscale PCA for Image Local Orientation Estimation

[MS-PCA]()

**思想：**

PCA（主成分分析）+ multiscale（多尺度金字塔）进行图像局部方向估计

**主成分分析：**

PCA ① 寻找能尽可能体现差异的属性，② 寻找能够尽可能好地重建原本特性的属性。两个目标是等效的，所以PCA可以一箭双雕。这个属性我们一般称为主成分、特征。Q：为什么两个目标等效 A：假设是一个假设新的特征是线性组合的直线。

**多尺度：**



#### :page_with_curl: edge–SR: Super–Resolution For The Masses

[edge-SR]()

WACV 2022 *BOE*

**分析现有方法：**

1. 超分辨率的历史

- - 传统插值算法：linear或者bicubic上采样，在低分辨率图像上插0然后低通滤波得到，对应pytorch、tf中的反卷积(*strided transposed convolutional layer*)
  - 先进的上采样算法：利用几何原理提升质量，自适应上采样和滤波为主
  - 深度学习：用CNN的SRCNN、用ResNets的EDSR、用DenseNets的RDN、用attention的RCAN、用非局部attention的RNAN、用transformer的SwinIR等

2. FSRCNN 和 ESPCN 都在未来的 SR 研究中留下了深刻的印记，这些研究经常以低分辨率执行计算并使用pixel-shuffle layers上采样。

**该文贡献：**

提出单层架构超分，详尽比较速度-效果权衡，对单层架构中的自注意力策略的分析和解释。

**该文观点：**

传统插值的上下采样是等效于：filter–then–downsampling和upsampling–then–filter。张量处理框架则使用跨步转置卷积层实现这种上采样。

传统插值上采样图示：

<img src="../../images/typora-images/image-20220916141622033.png" alt="image-20220916141622033" style="zoom:35%;" />

但图中的定义的升频(upscaling)显然效率低下，因为上采样(upsampling)引入了许多零，当乘以滤波器系数时会浪费资源。 一个众所周知的优化，广泛用于经典升频器的实际实现中，是将插值滤波器从图中的大小 sk×sk 拆分或解复用为 s^2 所谓的大小为 k × k 的高效滤波器。 然后，s^2 个滤波器的输出通过Pixel-Shuffle操作进行多路复用，以获得放大后的图像。

上采倍数越高，s越大，意味着实现**1**的核大小越大，或实现**2**的卷积通道数越大。

**提出模型：**

- eSR：

  卷积输出s^2个通道，最后pixel-shuffle得到超分结果。

- eSR-MAX：卷积输出C * s^2个通道，pixel-shuffle后，每C个通道取一个最大值。

  思想源自Maxout networks。

- eSR-TM：卷积输出2 * C * s^2个通道，pixel-shuffle后，前C个通道softmax，得到概率对后C个通道加权平均。

  属于自注意力，思想是Template Matching。

- eSR-TR

- eSR-CNN

**阅读评价：**

有启发意义，但实际由于硬件支持（比如耗时的softmax），并不如我们ZoomSR速度快。效果也是锯齿比较严重。另外，eSR-TM不一定得用softmax，用tanh，或者啥也不用应该也是有理由的。比如NAFNet的非线性激活就是个例子：

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
```

#### **:page_with_curl:SamplingAug: Patch Sampling Augmentation for SISR**

[SamplingAug](https://github.com/littlepure2333/SamplingAug)

**分析现有方法：**

现有方法大多在均匀采样的 LR-HR 补丁对上训练 DNN，这使得
 他们未能充分利用图像中的一些提供有用信息的补丁。

最近去噪领域也有工作提出训练一个额外的 PatchNet 来选择更多信息的补丁样本进行去噪网络的训练，能显著提高网络性能。

**该文方法：**

启发式度量来评估每个补丁对的信息重要性，再据此进行采样。

<img src="../../images/typora-images/image-20221027144430792.png" alt="image-20221027144430792" style="zoom:30%;" />

具体启发是：

能被线性函数超分的patch通常提供的重要信息更少。所以本文设计的度量方式是linearSR (如bilinear插值) 和HR之间的PSNR。

做法：

- 由于对所有overlapped patch都度量，计算复杂度会高。所以这里只对整个的积分图做linearSR。

- 考虑三种采样策略：

  1. 采样根据度量前p%的informative patches

  2. NMS(Non-Maximum Suppression)方法

     > 非极大抑制，是在重叠度较大的候选框中只保留一个置信度最高的。（Fast R-CNN中提出的）。

  3. TD(Throwing-Dart)采样策略

     > dart throwing （像一个人蒙上眼睛胡乱扔飞镖的样子）常用于渲染随机均匀的点组成的图案。每次在区域内随机选择一个点，并检查该点与所有已经得到的点之间是否存在“冲突”。若该点与某个已得到的点的最小距离小于指定的下界，就抛弃这个点，否则这就是一个合格的点，把它加入已有点的集合。重复这个操作直到获得了足够多的点。

  b、c会导致非重叠的采样，a、b、c都可以提升性能，其中策略a居然是效果最好的。说明信息量较少的样本可能对 SISR 的性能没有贡献。

- 积分图（integral image）又称总和面积表（summed area table）是一个快速且有效的对一个网格的矩形子区域中计算和的数据结构和算法。（就是junlin在面试时考我的数据结构）。

- 所有patchsize训练都会受益，不过小patchsize更明显；

  小模型的采样比例p越小效果越好，大模型的p可以稍大一点。这是由模型容量决定的；

  SamlingAug不同于<font color="brown">困难样本挖掘</font>（OHEM），OHEM只反传每个batch中最高的损失。用于SR时难以收敛。

  > Adobe的超分方法也有类似idea：
  >
  > A second key piece is that we focused our training efforts on “challenging” examples — image areas with lots of texture and tiny details, which are often susceptible to artifacts after being resized.
  
  



**阅读评价：**

2021、2022年SR领域提出多篇不同复杂度的网络处理不同难度的patch的工作，出发点一致，只是方法侧重略有不同。然而现有的inference engine对这类方法还不够友好。

而SamlingAug从训练样本的角度出发，则更为有意思一点。感觉和<font color="brown">困难样本挖掘</font>的思路有点类似，但绕了一道，不是直接SR和HR之间loss来决定是否反传。而是linearSR和HR之间的loss来决定是否反传。就成了与当前网络无关的独立因素了，避免了训练不稳定。

> 我这种思路，再结合topk的代码，还是挺好实现的。

#### :page_with_curl: Pixel-Adaptive Convolutional Neural Networks

[PAC](https://suhangpro.github.io/pac/)

**分析现有方法：**

权重共享的标准卷积是content-agnostic的。这个劣势可以通过学习大量滤波器来解决，但这样做增加了待学习的参数量，需要更大的内存占用和更多的标注数据。 另一种是content-adaptive的卷积：

- 一类是使传统的图像自适应滤波器可微分，例如双边滤波器（bilateral filters ）、导向滤波器（guided image filters ）、非局部均值（non-local means ）、传播图像滤波（propagated image filtering  ），并将它们用作 CNN 的层。 这些内容自适应层通常设计用于增强 CNN 结果，但不能替代标准卷积。 
- 另一类是内容自适应网络，使用单独的子网络学习特定位置的核，该子网络预测每个像素的卷积滤波器权重。 这些被称为“动态滤波网络”（DFN），也称为交叉卷积（cross-convolution）或核预测网络（KPN）。尽管 DFN 是通用的并且可作为标准卷积层的替代品，但很难扩展到具有大量滤波器组的整个网络。

**本文方法：**

标准卷积：

$\mathbf{v}'_i = \sum_{j \in \Omega(i)} \mathbf{W}\left[\mathbf{p}_i - \mathbf{p}_j\right] \mathbf{v}_j + \mathbf{b} $  (1)

这里$\mathbf{p}_i$是像素坐标。稍微滥用了符号，本文使用$[\mathbf{p}_i - \mathbf{p}_j]$来表示2D的空间维度
偏移量的索引。公式(1)可见，权重仅取决于像素位置，因此与图像内容无关。 换句话说，权重在空间上是共享的，是image-agnostic的。

现有动态卷积：

$\mathbf{v}'_i = \sum_{j \in \Omega(i)} \mathbf{W}\left(\mathbf{f}_i - \mathbf{f}_j\right) \mathbf{v}_j + \mathbf{b}$ (2)

使卷积操作内容自适应的一种直观方法，而不仅仅是基于像素位置，是将 W 泛化为依赖于像素特征。它的缺点是 ① 计算开销(overhead)大 ② 特征f很难学。需要手动指定特征空间，例如位置和颜色特征f = (x, y, r, g, b)。③ 我们必须限制特征的维度 d（例如，< 10），因为投影的输入图像在高维空间中可能变得过于稀疏。 ④ 此外，标准卷积的权值共享带来的优势在高维滤波中消失了。

像素自适应卷积：

$\mathbf{v}'_i = \sum_{j \in \Omega(i)} K\left(\mathbf{f}_i, \mathbf{f}_j\right) \mathbf{W}\left[\mathbf{p}_i - \mathbf{p}_j\right] \mathbf{v}_j + \mathbf{b}$  (3) 

其中 K 是具有固定参数形式的核函数，例如高斯：$K(\mathbf{f}_i, \mathbf{f}_j)=\exp(-\frac{1}{2}(\mathbf{f}_i-\mathbf{f}_j)^\intercal (\mathbf{f}_i-\mathbf{f}_j))$

因为 K 具有预定义的形式，我们可以在 2D 网格本身上执行此过滤，而无需移动到更高维。 我们将上述滤波操作（等式 3）称为“像素自适应卷积”（PAC），因为标准空间卷积 W 通过内核 K 使用像素特征 f 在每个像素处进行自适应。我们将这些像素特征 f 称为“自适应特征 ”，内核 K 为“自适应内核”。 **自适应特征 f** 可以是手动指定的，例如位置和颜色特征 f = (x, y, r, g, b)，也可以是端到端学习的深度特征。

<font color="red">PAC可以看作几种广泛使用的滤波操作的泛化形式：</font>

- 标准卷积可以看作PAC的特例，$K(\mathbf{f}_i, \mathbf{f}_j) = 1$。这可以通过设置常数自适应特征 f 来实现，即$\mathbf{f}_i = \mathbf{f}_j, \forall i,j$。

- 双边滤波也可看作PAC的特例，其中W 也有固定的参数形式，例如 2D 高斯滤波器：$\mathbf{W}\left[\mathbf{p}_i - \mathbf{p}_j\right]=\exp(-\frac{1}{2}(\mathbf{p}_i-\mathbf{p}_j)^\intercal\Sigma^{-1}(\mathbf{p}_i-\mathbf{p}_j))$.

  > 双边滤波（Bilateral filter）是**一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的**。

- 池化操作也可以由 PAC 建模。平均池化对应PAC的特例，

  $K(\mathbf{f}_i, \mathbf{f}_j)=1,\; \mathbf{W}=\frac{1}{s^2}\cdot\mathbf{1}$，最近提出的细节保持池化，对应$K(\mathbf{f}_i, \mathbf{f}_j)=\alpha + \left(|\mathbf{f}_i-\mathbf{f}_j|^2+\epsilon^2\right)^\lambda$

**阅读评价：**

文章提到了自注意力，说概念类似，但是PAC只关注local，不需要很高的复杂度。但比较好奇这样和空间注意力有啥区别？可能唯一区别是多了img2col。在卷积操作时起作用。

https://www.yuque.com/lart/architecture

#### :page_with_curl:Null-Space Learning for Consistent SR

零空间学习

#### :page_with_curl:IQA: Unifying Structure and Texture Similarity

#### :page_with_curl:Super Resolution for Compressed Screen Content Video

屏幕内容超分

现有超分失效的原因：

1. 数据gap：自然场景内容相对平滑，带有附加传感器噪声。 相比之下，屏幕内容视频可能具有锐利的边缘、高对比度的纹理和无噪声的内容。

   经过编码的屏幕内容：局部切入和截断在屏幕内容视频中很常见，这可能会由于缺少有利的参考而导致编码质量下降。

2. 其次，现有方法采用连续的相邻帧作为输入，而不考虑场景瞬间切换或局部突变。

3. 大多现有方面没有考虑压缩失真。

相关工作：

SSIM把两幅图的相似性按三个维度进行比较：亮度，对比度和结构。公式的设计遵循三个原则：对称性s(x,y)=s(y,x)、有界性s(x,y)<=1、极限值唯一s(x,y)=1当且仅当x=y。

比如，亮度相似度：$l(x,y)=\frac{2\mu_{x}\mu_{y} + C_{1}}{\mu_{x}^{2}+\mu_{y}^{2}+C_{1}} $

本文贡献：

- luminance-sharpness similarity作为loss，

  就是SSIM中亮度维度的相似度 + 提出的一个锐度的相似度。

- 网络结构主要就是利用3个输入：当前帧、前一帧、两帧差异的指数

数据有网页、游戏画面、动画和中英文文档等等。

评价：感觉从网络结构来说有些道理，但损失函数其实不一定比4个方向的梯度损失好，就是尺寸大一点，方向多一点。

#### :page_with_curl:CUF: Continuous Upsampling Filters

Nerf的思想和上采样任务结合：将上采样内核参数化为神经场(Neural fields)。

文章内容：

神经场表示具有基于坐标的神经网络的信号，它已经在许多领域找到了应用，包括 3D 重建、视点合成等。

最近的研究调查了神经场在单图超分辨率背景下的使用（[LIIF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf)和[LTE](https://github.com/jaewon-lee-b/lte)）。这些模型基于以编码器产生的潜在表示为条件的多层感知器。 虽然这种架构允许连续尺度的超分辨率，但它们需要在目标分辨率下为每个像素执行条件神经场，这使得它们不适合计算资源有限的应用程序。 此外，与亚像素卷积等经典超分架构相比，性能的提高并不能证明如此大量使用资源是合理的。

总而言之，神经领域尚未得到广泛采用，因为经典解决方案 1.实施起来更容易，2效率更高。 

在本文中，我们专注于克服这些限制，同时注意到（回归）超分辨率性能处于饱和状态（即，如果不依赖生成模型，图像质量的进一步改进似乎不太可能，而 PSNR 的微小提升不一定与主观一致）。

我们的动机假设是超分辨率卷积滤波器在空间和跨尺度上都是高度相关的。 因此，在<font color="purple">条件神经场的潜在空间中表示此类滤波器可以有效地捕获和压缩此类相关性</font>。



#### :page_with_curl:Implicit Transformer Network for Screen Content Super-Resolution

包括ITSRN和ITSRN++两篇文章。

训练集使用的SCI1K-train。

测试[数据集](https://drive.google.com/drive/folders/1uTQ2FAAUz5l-rtP35_fUhByRXxO25IFW)用到了SIQAD(20张600×800)、SCID(40张720p)、SCI1K-test。

相关工作：[Meta-SR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf)、 [LIIF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf)和[LTE](https://github.com/jaewon-lee-b/lte)，这些工作基本都属于连续超分（任意倍数）的范畴。这个领域最近关注度比较高。

#### :page_with_curl:LAU-Net: Latitude Adaptive Upscaling Network for ODI SR

全景图像(ODI)超分

ODI 的特征：不同纬度区域具有不均匀分布的像素密度和纹理复杂度。

本文思想：提出纬度自适应的超分网络，它允许不同纬度的像素采用不同的放大因子。

<img src="../../images/typora-images/image-20230201111115714.png" alt="image-20230201111115714" style="zoom:50%;" />

相关工作：PanoSwin: A Panoramic Shift Windowing Scheme for Panoramic Tasks

#### :page_with_curl:SwinIR

三部分：浅层特征提取+深层特征提取+图像重建

SwinIR普遍适用于各类图像复原任务，无需改动特征提取模块，对不同的任务仅仅是使用不同的重建模块。

<img src="../../images/typora-images/image-20230207160250936.png" alt="image-20230207160250936" style="zoom:50%;" />

**1.浅层特征提取：**采用一层卷积，根据论文[Early Convolutions Help Transformers See Better](https://proceedings.neurips.cc/paper/2021/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf)，得到特征$F_{0}$

**2.深层特征提取：**集联K个residual Swin Transformer blocks (RSTB)，最后接一层卷积。得到特征$F_{DF}$

>在特征提取的最后使用卷积层可以将卷积操作的归纳偏置带入基于Transformer的网络中，为后期浅层和深层特征的聚合打下更好的基础。

**3.图像重建：**聚合浅层和深层特征 $I_{RHQ} = H_{REC} (F_{0} + F_{DF} )$, 浅层特征主要包含低频，而深层特征侧重于恢复丢失的高频。 通过long skip connection，SwinIR可以将低频信息直接传递给重建模块，帮助深度特征提取模块专注于高频信息，稳定训练。 (1) 对于超分任务，使用亚像素卷积层对特征进行上采样。(2) 对于不需要上采样的任务，单层卷积就可以完成重建。另外SwinIR的输出会和低质量图像相加，形成残差学习。





#### :page_with_curl:Restormer

#### :computer: Anime4K

这是一个热门[项目](https://bloc97.github.io/Anime4K/)

[reddit探讨](https://www.reddit.com/r/compsci/comments/cq9n8t/anime4k_a_realtime_anime_upscaling_algorithm_for/)

[Preprint](https://github.com/h5mcbox/anime4k/blob/master/Preprint.md)

现有方法：

1. 核方法，比如Bicubic或xBR是为其他内容设计，倾向于软化边缘。不适合用于动漫内容。

2. 传统的去模糊和锐化方法，会导致overshoot(过冲，之后通常发生振铃效应)，出现在边缘附近。这会分散观看者的注意力并降低图片的感知质量。

3. 基于学习的方法（例如waifu2x、EDSR等）对于实时（<30ms）应用程序来说太慢了几个数量级。

   >虽然waifu2x 或 EDSR 等算法的性能大大优于任何其他通用上采算法。
   >
   >然而，我们将利用我们的上采样算法只需要处理单一类型的内容（动画）这一事实的优势，因此我们可能有机会匹配（甚至超越）基于学习的算法。

本文方法：

Anime4K的出发点是寻找一种好的边缘细化算法，而不是寻求通用的放大算法。 与恢复纹理等小细节相比，清晰的边缘对于动漫升级更为重要。

算法实现：

主要目标是修改$LR_U$（模糊图像）直到其残差变得最薄，从而为我们提供可能的 $HR$（清晰）图像之一。

我们的算法将简单地将 $LR_U$ 及其残差作为输入，推动残差的像素，使残差线变得更细。 对于对残差执行的每个“推”操作，我们对彩色图像执行相同的操作。 残差将作为推动的指引。 这具有迭代最大化图像梯度的效果，这在数学上等同于最小化模糊，但没有传统“去模糊”和“锐化”方法中常见的过冲或振铃伪影。

伪代码：

```cpp
for each pixel on the image:
  for each direction (north, northeast, east, etc.):
    using the residual, if an edge is found:
      push the residual pixel in the current direction
      push the color pixel in the current direction
```

我们的算法用来提高性能的一个技巧是使用 sobel 滤波器来近似图像的残差，而不是使用高斯滤波器计算残差，因为计算高斯核的成本更高。 此外，最大化 sobel 梯度在数学上类似于（但不等同！）最小化残差厚度。 这种修改在目视检查中没有产生质量下降。

该算法的一个优点是它与尺度无关。 动漫可能事先被错误地放大了（双倍放大，甚至先缩小后放大），并且该算法仍然会检测到模糊边缘并对其进行细化。 因此，可以使用用户喜欢的任何算法（双线性、Jinc、xBR 甚至 waifu2x）提前对图像进行放大，然后该算法将正确地细化边缘并消除模糊。 在 900p 的动漫上运行此算法，使结果看起来像真正的 1080p 动漫。 为了获得更强的去模糊效果，我们只需再次运行该算法。 该算法迭代地锐化图像。

然而，对于 2倍 上采样，我们注意到线条通常太粗并且看起来不自然（因为模糊通常向外散布暗线，使它们变粗），因此我们为细线添加了预通道。 此通道不是算法的组成部分，如果用户希望保留粗线，则可以安全地删除它。

我们已经在 Java 和 HLSL/C 中实现了这个算法。

[Python版](https://github.com/TianZerL/Anime4KPython/blob/master/Anime4KPython/Anime4K.py)

```python
# Process anime4k
def process(self):
  for i in range(self.passes): # 迭代的遍数
    self.getGray() # 得到Y通道
    self.pushColor() # 分上下，左右，正对角，负对角，然后两侧的强弱情况，一共4x2=8种情形，判断是否执行推的操作。
    self.getGradient()
    self.pushGradient()
```

推的具体过程：

```python
def getLightest(mc, a, b, c):
            mc[R] = mc[R] * (1 - self.sc) + (a[R] / 3 + b[R] / 3 + c[R] / 3) * self.sc
            mc[G] = mc[G] * (1 - self.sc) + (a[G] / 3 + b[G] / 3 + c[G] / 3) * self.sc
            mc[B] = mc[B] * (1 - self.sc) + (a[B] / 3 + b[B] / 3 + c[B] / 3) * self.sc
            mc[A] = mc[A] * (1 - self.sc) + (a[A] / 3 + b[A] / 3 + c[A] / 3) * self.sc

'''
比如
tl tc tr
ml mc mr
bl bc br

如果
maxD = max(tc[A], mc[A], ml[A])
minL = min(mr[A], br[A], bc[A])
minL > maxD，就是正对角的下方比上方强。
此时会调用getLightest(mc, mr, br, bc)，
mc[R] = 
mc[R] * (1 - α) + (mr[R] / 3 + br[R] / 3 + bc[R] / 3) * α，
α是可调节的'推'的强度。
'''


```

总结一下

`getGray`：计算图像的灰度并将其存储到 Alpha 通道

`pushColor`：会在 Alpha 通道的灰度引导下使图像的线条变细

具体就是两侧比较的各3个像素，哪边强，当前位置就和哪边靠近。比如，左比右强，mc和左面一列的均值加权更新。

`getGradient`：计算图像的梯度并将其存储到 Alpha 通道

用的sobel滤波，快慢模式的区别仅仅是①abs和②平方和开方。

`pushGradient`：将在 Alpha 通道中的梯度引导下使图像的线条更锐利

具体操作和pushColor类似，只是引导由灰度图变为了梯度。

参数推荐：<font color="lighblue">passes=1,strengthColor=0.1,strengthGradient=0.8,fastMode=True</font>

评价：挺不错的，可以结合RAISR的统计信息分析，对部分像素使用。

#### :page_with_curl:Neural Preset for Color Style Transfer

文章有不错的应用价值。作者原来是专门研究半监督/自监督学习的，最近开始将自监督引入一些CV任务中，提出了MODNet和本文的Neural Preset。比较有意思的是，Neural Preset不需要fine-tuneing就能用在暗光增强、水下图像增强、去雾和图像和谐化等多个任务中。

> 术语了解：艺术风格迁移 不同于 颜色风格转换（也称写实风格迁移）。
>
> artistic style transfer：改变纹理和颜色。
>
> color style transfer (*aka* photoreal- istic style transfer)：仅改变颜色。

**现有方法**通常在实践中受到三个明显的限制：

1. 伪影（例如，扭曲的纹理或不和谐的颜色），因为它们执行基于卷积模型的颜色映射，卷积模型对图像块进行操作，对于相同像素值的输入可能具有不一致的输出。
2. 运行时内存占用巨大，它们无法处理高分辨率（例如8K）图像。
3. 它们在切换样式时效率低下，因为它们将颜色样式转换作为单阶段过程进行，每次都需要运行整个模型。

本文方法：

- 确定性神经颜色映射 (<font color="brown">DNCM</font>)：相同颜色的像素转换到相同的输出。但只需极少内存，不像3D-LUT那样耗内存。

  > 手工滤波器和3D查找表也属于确定性颜色映射。但手工滤波器功能性弱，只能处理基本的颜色调整。通过预测系数，融合LUT模板的方法，则具有大量可学习参数，难以优化。

- <font color="brown">两阶段</font>：

  第一阶段由输入建立nDNCM，用于将输入归一化到仅表示“图像内容”的空间；

  第二阶段由风格图建立sDNCM，用于将归一化图像转化到目标风格。

  **两方面好处**，①可以存储sDNCM，作为预设滤镜。②可以一次nDNCM后快速切换不同滤镜。

- 由于难以获取配对数据，使用了<font color="brown">自监督</font>学习。

  > 自监督学习（SSL）已被广泛探索用于预训练 [4、5、19、21、25、26、58–60]。 一些作品还通过 SSL [31、37、42] 解决特定的视觉任务。本文应该属于后者。

无需工程技巧，就可以在3090显卡上实时增强4K视频，且具有帧间一致性。

DNCM：

<img src="../../images/typora-images/image-20230407135638072.png" alt="image-20230407135638072" style="zoom:50%;" />

两阶段：

在两阶段的方式中，编码器 E 被修改以输出 d 和 r，它们分别用作 nDNCM 和 sDNCM 的参数。E()共享权重，但nDNCM和sDNCM有不同的投影矩阵 P 和 Q。

<img src="../../images/typora-images/image-20230407141029803.png" alt="image-20230407141029803" style="zoom:50%;" />

自监督：

本文提出一种自监督策略：

<img src="../../images/typora-images/image-20230408163706000.png" alt="image-20230408163706000" style="zoom:36%;" />

由于难以获取GT的风格化图像，我们有输入图像$I$制作两个伪风格图像。具体方式是通过对$I$色彩的扰动，获得两个不同色彩风格的数据增强样本$I_i$和$I_j$。扰动可以由色彩滤镜或LUT完成。

损失函数细节：

内容一致损失用L2，$\mathcal{L}_{con} = || \mathbf{Z}_{i} - \mathbf{Z}_{j} ||_{2}$，

风格重建损失用L1，$\mathcal{L}_{rec} = || \mathbf{Y}_{i} - \mathbf{I}_{i} ||_{1} + || \mathbf{Y}_{j} - \mathbf{I}_{j} ||_{1}$，

最终损失函数为：$\mathcal{L} = \mathcal{L}_{rec} + \lambda \, \mathcal{L}_{con}$，$\lambda$是一个可控的权重。

#### :page_with_curl:Color Image Enhancement with Saturation Adjustment Method



### 智能编码系列

[2016~2022](./CLIC.html)

### TalkingHead

#### :page_with_curl:Perceptual Head Generation with Regularized Driver and Enhanced Renderer

黄哲威的工作，基于俞睿师兄的PIRenderer

#### :page_with_curl:PIRenderer: Controllable Portrait Image Generation

俞睿师兄的工作，旨在得到具有语义和完全分离的参数——使用三维可变形面部模型（3DMM）的参数来控制面部运动。

> 提供细粒度控制，直观且易于使用。这是FOMM等工作所不能达成的，它们阻碍了模型以直观方式编辑肖像的能力。

结合技术的先验知识，人们可以期望控制类似于图形渲染处理的照片般逼真的肖像图像的生成。

> 为了实现直观控制，运动描述符应该在语义上有意义，这需要将面部表情、头部旋转和平移表示为完全分离的变量。

<img src="../../images/typora-images/image-20230314105220551.png" alt="image-20230314105220551" style="zoom:36%;" />

我们表明，我们的模型不仅可以通过使用用户指定的动作编辑目标图像来实现<font color="brown">直观的（top）</font>图像控制，而且还可以在<font color="brown">间接（middle）</font>肖像编辑任务中生成逼真的结果，其中目标是模仿另一个人的动作。 此外，我们通过进一步扩展模型来解决音频驱动的面部重现任务，展示了我们模型作为高效神经渲染器的潜力。 由于高层完全分离的参数化，我们可以从<font color="brown">“弱”控制音频（bottom）</font>中提取令人信服的动作。

>3DMM参数：人脸三维网格、模型在图像中的位置，以及当前人脸的形状特征参数和blendshape表情参数。（即 pose（包括旋转rotation和平移translation）, identity, expression）
>
>现在有些固定了pose和identity的应用场景，则只需要blendshape。

PIRenderer包含三部分网络：

> 0. 采用现成的 3D 人脸重建模型从真实世界的人像图像中提取相应的 3DMM 系数进行训练和评估；
>
>    注：为了缓解系数提取误差和噪声带来的影响，将具有**连续帧的窗口**的系数用作中心帧的运动描述符。

1. Mapping Network：将3DMM参数映射为隐向量（a latent vector）；
2. Warping Network：在隐向量的指导下，估计source和所需target之间的光流，并通过用估计的变形来warp源得到粗略的生成结果；
3. Editing Network：从粗略的结果到精细的生成结果。

<img src="../../images/typora-images/image-20230314142526847.png" alt="image-20230314142526847" style="zoom:50%;" />

[Face3D 拓展](https://zhuanlan.zhihu.com/p/530830577)

#### :page_with_curl:Responsive Listening Head Generation: A Benchmark Dataset and Baseline

生成聆听者的头部视频的任务。

任务：聆听头部生成将来自说话者的音频和视觉信号作为输入，并以实时方式提供非语言反馈（例如，头部运动、面部表情）。

本文主要是提供数据集，和尝试基于PIRender的baseline。

#### :page_with_curl:Structure-Aware Motion Transfer with Deformable Anchor Model

阿里妈妈的工作，基于FOMM。👉🏻[PR](https://blog.csdn.net/alimama_Tech/article/details/125419491)

FOMM的缺陷：

**源图像**和**驱动图像**关键点检出的对应关键点并没有指向同一个真实部位时，输出结果中这个区域的就有较强的结构模糊。另外，还有一个观察是通常这样的不匹配都源自关键点检测没有击中合理的部位，甚至都在人体/脸部以外了。

本文改进：

推理不变，训练则引入结构先验（或叫位置先验），关键点的对应点既可以通过根节点光流图计算得到，又可以通过模型直接直接预测出来。两路结果求Loss，就惩罚了不符合根节点先验约束的关键点位置预测。

效果比较：

FOMM vs. RegionMM vs. This paper
