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

1. example-based方法，即使用外部数据集学习
