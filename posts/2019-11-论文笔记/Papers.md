#### Multi-Stage Progressive Image Restoration

[MPRNet](https://arxiv.org/pdf/1803.05407.pdf)

CVPR 2021 *Inception Institute of AI, UAE*

###### 分析现有方法：

1. encoder-decoder结构 ：<b>优：</b>更宽的上下文信息，<b>劣：</b>保持细节不足。
2. single-scale pipeline ：<b>优：</b>保持空间上的精确细节，<b>劣：</b>语义上不太可靠。

###### 该文观点：

1. 在multi-stage结构中，结合encoder-decoder与single-scale pipeline，是必要的。
2. multi-stage不仅仅是上一阶段的输出，也作为下一阶段的输入。
3. 每一阶段都用上监督信息是很重要的，渐进式学习。由此，设计supervised attention module (SAM)模块。
4. 该文提出了将上一阶段特征(contextualized)传递给下一阶段。由此，设计cross-stage feature fusion (CSFF)方法。

###### 网络结构：

<img src="MPRNet.png" style="width:80%; height: 80%;">



#### Series-Parallel Lookup Tables

[SPLUT]()

ECCV 2022 *Tsinghua University*

###### 分析现有方法：

1. LUT通过快速内存访问代替耗时的计算，具有实用性。
2. 但是大多数现有的基于 LUT 的方法只有一层 LUT。 如果使用 n 维 LUT并且用于查询v个可能值，则 LUT 的尺寸有 v^n。 因此，通常将 v 和 n 设置为较小的值以避免无法承受的大 LUT，这严重限制了效果。



#### RAISR: Rapid and Accurate Image Super Resolution

[RAISR]()

TCI 2016 *Google*

###### 分析现有方法：

1. 传统插值方法，是内容无关的线性方法，表达能力不足。

###### 该文观点：

1. example-based方法，即使用外部数据集学习LR patch到HR patch的映射。





#### edge–SR: Super–Resolution For The Masses

[edge-SR]()

WACV 2022 *BOE*

###### 分析现有方法：

1. 超分辨率的历史

- - 传统插值算法：linear或者bicubic上采样，在低分辨率图像上插0然后低通滤波得到，对应pytorch、tf中的反卷积(*strided transposed convolutional layer*)
  - 先进的上采样算法：利用几何原理提升质量，自适应上采样和滤波为主
  - 深度学习：用CNN的SRCNN、用ResNets的EDSR、用DenseNets的RDN、用attention的RCAN、用非局部attention的RNAN、用transformer的SwinIR等

2. FSRCNN 和 ESPCN 都在未来的 SR 研究中留下了深刻的印记，这些研究经常以低分辨率执行计算并使用pixel-shuffle layers上采样。

###### 该文贡献：

提出单层架构超分，详尽比较速度-效果权衡，对单层架构中的自注意力策略的分析和解释。

###### 该文观点：

传统插值的上下采样是等效于：filter–then–downsampling和upsampling–then–filter。张量处理框架则使用跨步转置卷积层实现这种上采样。

传统插值上采样图示：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220916141622033.png" alt="image-20220916141622033" style="zoom:35%;" />

但图中的定义的升频(upscaling)显然效率低下，因为上采样(upsampling)引入了许多零，当乘以滤波器系数时会浪费资源。 一个众所周知的优化，广泛用于经典升频器的实际实现中，是将插值滤波器从图中的大小 sk×sk 拆分或解复用为 s^2 所谓的大小为 k × k 的高效滤波器。 然后，s^2 个滤波器的输出通过Pixel-Shuffle操作进行多路复用，以获得放大后的图像。

上采倍数越高，s越大，意味着实现**1**的核大小越大，或实现**2**的卷积通道数越大。

###### 提出模型：

