[TOC]



## Transformer逐段精读

**Attention Is All You Need**

作者排名随机

**任务：**序列转录（机器翻译）

**架构：**简单（在深度学习领域成为褒义词），仅仅倚赖于注意力机制

**现有：**RNN并行度差。CNN难以长距离建模。

**借鉴：**卷积比较好的地方是 可以做多个输出通道，可以认为每个输出通道识别不一样的模式。所以提出多头（Multi-Head）注意力机制来模拟这样一个效果。

**展望：**作者非常前瞻地预言了Transformer可以用于图像、视频。











## ViT逐段精读

**An Image Is Worth 16x16 Words**

**任务：**挖出视觉甚至多模态领域的大坑

**现有：**Transformer成为NLP领域里的一个标准。但在视觉领域，自注意力要么跟CNN一起用，要么仅仅替换CNN里的个别组件。



















## Swin Transformer逐段精读





















## Tokens-to-Token精读

