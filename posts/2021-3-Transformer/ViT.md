## Transformer

[TOC]

**by** <font color="brown">**[Jay Alammar](http://jalammar.github.io/)**</font>

#### 前言

Transformer的重要意义。

Transformer由Encoder、Decoder构成，其中的基础模块称作encoder block和decoder block。

语言模型里GPT-2、GPT-3用到了decoder block（GPT-2用36），Bert则是用了encoder block(24个)。





# Vistion Transformer

[TOC]

## Transformer在CV上的应用前景

研究方向主要两个：

1. **作为convolution的补充**

   

2. **替代convolution：**ViT中不同的query是share key set的，这会使得内存访问非常友好而大幅度提速。一旦解决了速度问题，self-attention module在替代conv的过程中就没有阻力了。

   注：ViT是2020年10月挂在Arxiv上，2021年发表。

   

ViT的特性：

1. long range带来的全局特性：

   从浅层到深层，都比较能利用全局的有效信息。multi-head机制保证了网络可以关注到多个discriminative parts，其实每一个head都是一个独立的attention。

   

2. 更好的多模态融合能力：

   CNN擅长的是解构图像的信息，卷积核就是以前传统数字图像处理中的滤波操作。而Transformer中，不需要保持H*W*C的feature map结构。就类似position embedding，只要你能编码的信息，都可以非常轻松地利用进来。

   

3. Multiple tasks能力。

   

4. 更好的表征能力。

   

   总体而言，Transformer是NLP给CV的一个输出，我们可以去学习Transformer的长处，至于未来是否会替换CNN，或者Transformer与CNN共存，甚至互相弥补，这个还是靠整个学界去决定。CV的任务很多很难，无论是CNN还是Transformer都不会是CV的终点，保持学习、保持接纳、保持探究。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013112858069.png" alt="image-20211013112858069" style="zoom:10%;" />

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013113017587.png" alt="image-20211013113017587" style="zoom:10%;" />

​														这里的xi到zi是都是共享参数的全连接层。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013112754034.png" alt="image-20211013112754034" style="zoom:10%;" />

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013113211338.png" alt="image-20211013113211338" style="zoom:10%;" />

​						                          transformer encoder的输出中，有用的是向量c0。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013114049198.png" alt="image-20211013114049198" style="zoom:10%;" />

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013114807750.png" alt="image-20211013114807750" style="zoom:50%;" />

Transformer的缺点：

需要大量的数据进行训练。ViT论文里研究了三个数据集，其中JFT甚至有3亿张图片。可惜JFT是谷歌私有的一个数据集，不对外公开。

实验结果表示，如果用ImageNet（small）预训练，ViT表现不如ResNet。用ImageNet-21K（medium），ViT表现与ResNet相当。只有在使用JFT数据集后，ViT才超越了ResNet。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013114951914.png" alt="image-20211013114951914" style="zoom:50%;" />

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20211013122242576.png" style="zoom:50%;" />

而且实验的迹象表明，即便是3亿张图片的JFT也不够大，继续增大数据集，ViT的优势还会进一步增大。反观ResNet，预训练的数据量是1亿还是3亿张图片时区别不大。

