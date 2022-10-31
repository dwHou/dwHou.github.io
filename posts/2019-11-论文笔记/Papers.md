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

   3. 如若上采样+滤波，滤波器组只需一个(但尺寸要求较大)。如若滤波+pixelshuffle，滤波器组需要上采样倍数^2个。

      1. 上采样+滤波，会更复杂的有Pixel-Type，这是由于插值方法的特性（像素的来源不同）导致的。
      2. 滤波+pixelshuffle，不需要Pixel-Type，或者说Pixel-Type隐含在了我们有4组滤波器，即4种Type。

   4. 所有滤波器权重均由学习得到。

4. 优化点：

   1. 考虑许多上采样kernel中心对称的特性。可以减少一半冗余。
   2. 考虑使用量化。RAISR只有单层，不存在误差累积。





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

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20221027144430792.png" alt="image-20221027144430792" style="zoom:30%;" />

具体启发是：

能被线性函数超分的patch通常提供的重要信息更少。所以本文设计的度量方式是linearSR和HR之间的PSNR。

做法：

- 由于对所有overlapped patch都度量，计算复杂度会高。所以这里只对整个的积分图做linearSR。

- 考虑三种采样策略：

  1. 采样根据度量前p%的informative patches

  2. NMS(Non-Maximum Suppression)方法

     > 非极大抑制，是在重叠度较大的候选框中只保留一个置信度最高的。（Fast R-CNN中提出的）。

  3. TD(Throwing-Dart)采样策略

     > dart throwing （像一个人蒙上眼睛胡乱扔飞镖的样子）常用于渲染随机均匀的点组成的图案。每次在区域内随机选择一个点，并检查该点与所有已经得到的点之间是否存在“冲突”。若该点与某个已得到的点的最小距离小于指定的下界，就抛弃这个点，否则这就是一个合格的点，把它加入已有点的集合。重复这个操作直到获得了足够多的点。

  b、c会导致非重叠的采样，a、b、c都可以提升性能，其中策略a居然是效果最好的。说明信息量较少的样本可能对 SISR 的性能没有贡献。

- 



**阅读评价：**

2021、2022年SR领域提出多篇不同复杂度的网络处理不同难度的patch的工作，出发点一致，只是方法侧重略有不同。然而现有的inference engine对这类方法还不够友好。

而SamlingAug从训练样本的角度出发，则更为有意思一点。同时和<font color="brown">困难样本挖掘</font>也有点关联。
