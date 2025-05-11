https://github.com/ZJU-LLMs/Foundations-of-LLMs

[浙江大学-大模型原理与技术](https://www.bilibili.com/video/BV1PB6XYFET2/)

毛玉仁

## 第一章 语言模型基础

#### 00 序言

**语言之于智能**：在认知层面，语言与智能紧密相连，语言是智能的载体。

**如何建模语言**：将语言建模为一系列词元（Token）组成的序列数据。其中，词元是不可再拆分的最小语义单位。

**语言模型**：语言模型旨在预测一个词元或词元序列出现的概率。现有语言模型通常基于规则、统计或学习来构建。

`{我，为，什么，要，选，这，门，课}` → 语言模型 → 0.66666

<img src="../../images/typora-images/image-20250425062027391.png" alt="image-20250425062027391" style="zoom:50%;" />

语言模型的概率预测与<font color="seablue">上下文</font>和<font color="seablue">语料库</font>息息相关。

上下文

{这，课，好，难，`我，为，什么，要，选，这，门，课`} → 语言模型 → 0.9

{这，课，好，简，单，`我，为，什么，要，选，这，门，课`} → 语言模型 → 0.2

语料库

普通话语料库：`{我，为，什么，要，选，这，门，课}` → 语言模型 → 0.6

四川话语料库：

`{我，为，什么，要，选，这，门，课}` → 语言模型 → 0.2

`{我，为，啥子，要，选，这，门，课}` → 语言模型 → 0.6

综合以上两点，我们可以用条件概率的链式法则对语言模型的概率进行建模。

**条件概率链式法则**

设词元序列为$\{w_1, w_2, ..., w_N\}$，其概率可由条件概率的链式法则进行计算。

$P(\{w_1, w_2, ..., w_N\}) = P(w_1) \cdot P(w_2|w_1) \cdot  P(w_3|w_1,w_2) ... P(w_N|w_1,w_2, ... , w_{N-2}, w_{N-1})$

**n-阶马尔科夫假设**

当前状态只与<font color="seablue">前面n个状态</font>有关。

对序列$\{w_1, w_2, ..., w_N\}$，当前状态$w_N$出现的概率只与前n个状态$\{w_{N-n},... ,w_{N-1}\}$有关，即：

$P(w_N|w_1,w_2, ... , w_{N-1}) \approx P(w_N|w_{N-n},... ,w_{N-1})$

<img src="../../images/typora-images/image-20250425065726030.png" alt="image-20250425065726030" style="zoom:50%;" />

#### 01 基于统计的语言模型

###### 1.1 n-grams 语言模型

n-grams 语言模型中的n-gram 指的是长度为n 的词序列。n-grams 语言模型通过依次统计文本中的n-gram 及其对应的(n-1)-gram 在语料库中出现的相对频率来

计算文本$w_{1:N}$ 出现的概率。

> 经典的n-grams语言模型，被工业界沿用至今。

<img src="../../images/typora-images/image-20250425071942493.png" alt="image-20250425071942493" style="zoom:50%;" />

n-grams语言模型中，<font color="brwon">n为变量</font>，当n=1时，称之为<font color="brwon">unigram</font>，其不考虑文本的上下文关系。当n=2时，称之为<font color="brwon">bigrams</font>，其对前一个词进行考虑。当n=3时，称之为<font color="brwon">trigrams</font>，其对前两个词进行考虑。以此类推。

bigrams的例子：

<img src="../../images/typora-images/image-20250425072659867.png" alt="image-20250425072659867" style="zoom:50%;" />

虽然“长颈鹿脖子长”并没有直接出现在语料库中，但是bigrams 语言模型仍可以预测出“长颈鹿脖子长”出现的概率有 2/15。由此可见，n-grams具备<font color="brwon">对未知文本的泛化能力</font>。

###### 1.2 n-grams中的n

$P_{trigrams}(长颈鹿, 脖子, 长) = \frac{C(长颈鹿, 脖子, 长)}{C(长颈鹿, 脖子)} = 0$

n的选择会影响n-grams模型的<font color="brwon">泛化性能</font>和<font color="brwon">计算复杂度</font>。实际中n通常<font color="brwon">小于等于5</font>。

泛化性：在n-grams 语言模型中，<font color="brwon">n 代表了拟合语料库的能力与对未知文本的泛化能力之间的权衡</font>。当n 过大时，语料库中难以找到与n-gram 一模一样的词序列，可能出现大量“零概率”现象；在n 过小时，n-gram 难以承载足够的语言信息，不足以反应语料库的特性。

计算量：随着n的增大，n-gram模型的参数呈指数级增长。假设语料库中包含1000个词汇，则unigram的参数量为1000，而bigrams的参数量则为1000*1000。

**n-grams中的统计学原理**

n-grams语言模型是在n阶马尔可夫假设下，对语料库中出现的<font color="brwon">长度为n的词序列出现概率</font>的<font color="brwon">极大似然估计</font>。

###### 1.3 n-grams语料及数据

n-gram的效果与语料库息息相关。Google在2005年开始Google Books Library Project项目，试图囊括自现代印刷术发明以来的全世界所有的书刊。其提供了unigram到5-gram的数据。

**n-grams的应用**

n-gram不仅在输入法、拼写纠错、机器翻译等任务上得到广泛应用。其还推动了Culturomics（文化组学）的诞生。

**n-grams的缺点**

n-gram因为观测长度有限，无法捕捉长程依赖。此外，其是逐字匹配的，不能很好地适应语言的复杂性。

<img src="../../images/typora-images/image-20250425164032067.png" alt="image-20250425164032067" style="zoom:50%;" />



#### 02 基于学习的语言模型

###### 2.0 学习与统计的区别

统计：设计模型，描摹已知。

学习：找到模型，预测未知。

<img src="../../images/typora-images/image-20250425164211005.png" alt="image-20250425164211005" style="zoom:50%;" />

###### 2.1 机器学习的过程

机器学习的过程：在某种<font color="brwon">学习范式</font>下，基于<font color="brwon">训练数据</font>，利用<font color="brwon">学习算法</font>，从受<font color="brwon">归纳偏置</font>限制的<font color="brwon">假设类</font>中选取可以达到<font color="brwon">学习目标</font>的假设，该假设可以<font color="brwon">泛化</font>到未知数据上。

假设类：

<img src="../../images/typora-images/image-20250427141627827.png" alt="image-20250427141627827" style="zoom:50%;" />

归纳偏置：

<img src="../../images/typora-images/image-20250427141809153.png" alt="image-20250427141809153" style="zoom:50%;" />

学习范式：

<img src="../../images/typora-images/image-20250427141911240.png" alt="image-20250427141911240" style="zoom:50%;" />

学习目标：

<img src="../../images/typora-images/image-20250427141949855.png" alt="image-20250427141949855" style="zoom:50%;" />

损失函数：

<img src="../../images/typora-images/image-20250427142019106.png" alt="image-20250427142019106" style="zoom:50%;" />

学习算法：

1阶优化：目前最常用的梯度下降。

0阶优化：对梯度进行模拟，用估计出来的梯度来对模型进行优化。

<img src="../../images/typora-images/image-20250427142132603.png" alt="image-20250427142132603" style="zoom:50%;" />

泛化误差：

<img src="../../images/typora-images/image-20250427142512867.png" alt="image-20250427142512867" style="zoom:50%;" />

泛化误差界的公式来自<font color="brwon">概率近似正确</font>（PAC，Probably Approximately Correct）理论。

PAC Learning为机器学习提供了对机器学习方法进行定量分析的理论框架，可以为设计机器学习方法提供理论指导。

> Leslie Valiant由该理论，获得2010年图灵奖。

<img src="../../images/typora-images/image-20250427143613330.png" alt="image-20250427143613330" style="zoom:50%;" />

###### 2.2 机器学习的发展历程

<img src="../../images/typora-images/image-20250427150044567.png" alt="image-20250427150044567" style="zoom:50%;" />

<img src="../../images/typora-images/image-20250427150311713.png" alt="image-20250427150311713" style="zoom:50%;" />

#### 03 RNN与Transformer

###### 3.1 RNN

RNN 是一类<font color="brwon">网络连接中包含环路的神经网络的总称</font>。

<img src="../../images/typora-images/image-20250428110728884.png" alt="image-20250428110728884" style="zoom:50%;" />

RNN 在串行输入的过程中，前面的元素会被循环编码成<font color="brwon">隐状态</font>，并<font color="brwon">叠加到当前的输入上面</font>。是在时间维度上嵌套的复合函数。

在训练RNN时，涉及大量的矩阵联乘操作，容易引发<font color="brwon">梯度衰减</font>或<font color="brwon">梯度爆炸</font>问题。

**LSTM**

为解决经典RNN的梯度衰减/爆炸问题，带有<font color="brwon">门控机制</font>的LSTM被提出。

LSTM将经典RNN中的通过复合函数传递隐藏状态的方式，解耦为<font color="brwon">状态累加</font>。隐藏状态通过<font color="brwon">遗忘门</font>、<font color="brwon">输入门</font>来实现合理的状态累加，通过<font color="brwon">输出门</font>实现合理整合。

- LSTM中采用遗忘门来适度忘记“往事”。
- LSTM中采用输入门来对“新闻”进行选择性聆听。
- 将“往事”与“新闻”相加得到当前状态。
- LSTM采用输出门，考虑“人情世故”，将当前状态适度输出。

GRU为降低LSTM的计算成本，GRU将遗忘门与输入门进行合并。

###### 3.2 Transformer

典型的支持<font color="brwon">并行输入</font>的模型是Transformer，其是一类基于注意力机制的<font color="brwon">模块化</font>构建的神经网络结构。

两种主要模块

**(1) 注意力模块**

注意力模块负责对<font color="brwon">上下文</font>进行通盘考虑。

注意力模块由自<font color="brwon">注意力层</font>、<font color="brwon">残差连接</font>和<font color="brwon">层正则化</font>组成。

<img src="../../images/typora-images/image-20250504171111974.png" alt="image-20250504171111974" style="zoom:50%;" />

**(2) 全连接前馈模块**

全连接前馈模块占据了Transformer近三分之二的参数，掌管着Transformer模型的<font color="brwon">记忆</font>。

1. 注意力层

<img src="../../images/typora-images/image-20250504171823386.png" alt="image-20250504171823386" style="zoom:50%;" />

说白了是加权输出的机制，而权重是通过$W_q$、$W_k$两个矩阵学出来的。

<font color="brwon">加权平均</font>：原值是$v$，权重是当前位置的$q$和上下文的$k$的相似度。

2. 层正则化与残差连接

层正则化用以加速神经网络训练过程并取得更好的泛化性能；引入残差连接可以有效解决梯度消失问题。

> 残差连接加在LN前叫pre-LN，加在LN后叫post-LN。不同的模型里，两种加法性能各有优劣。

**自回归 vs. Teacher Forcing**

自回归面临着错误级联放大和串行效率低两个主要问题。为了解决上述两个问题，Teacher Forcing在语言模型预训练过程中被广泛应用。

<img src="../../images/typora-images/image-20250504173734109.png" alt="image-20250504173734109" style="zoom:50%;" />

###### 3.3 训练RNN/Transformer的过程

<img src="../../images/typora-images/image-20250504174229743.png" alt="image-20250504174229743" style="zoom:50%;" />

<img src="../../images/typora-images/image-20250504174246875.png" alt="image-20250504174246875" style="zoom:50%;" />

但Teacher Forcing的训练方式将导致<font color="brwon">曝光偏差</font>（Exposure Bias）：训练模型的过程和模型在推理过程存在差异。其易导致<font color="brwon">模型幻觉</font>问题。

#### 04 语言模型采样与评测

###### 4.1 语言模型的采样

语言模型每轮预测输出的是一个概率向量。我们需要<font color="brwon">根据概率值从词表中选出本轮输出的词元</font>。选择词元的过程被称为<font color="brwon">采样</font>。

两类主流的采样方法可以总结为 

(1). 概率最大化方法

最大化$P(w_{N+1:N+M}) = \prod_{i=N}^{N+M-1} P(w_{i+1}|w_{1:i}) = \prod_{i=N}^{N+M-1}o_i(w_{i+1})$

假设生成M个词元，概率最大化方法的搜索空间为$M^D$，是<font color="brwon">NP-Hard问题</font>。

- 贪心搜索法

  贪心搜索在每轮预测中都选择概率最大的词

  

- 波束搜索法

  波束搜索在每轮预测中都先保留b个可能性最高的词，在结束搜索事，得到M个集合。找出最优组合使得联合概率最大。

  <img src="../../images/typora-images/image-20250504185235651.png" alt="image-20250504185235651" style="zoom:50%;" />

但概率最大的文本通常是<font color="brwon">最为常见的文本</font>。这些文本会略显平庸。用于生成代码还行。但在开放式文本生成中，贪心搜索和波束搜索都容易生成一些<font color="brwon">“废话文学”</font>——重复且平庸的文本。 

(2). 随机采样方法

在每轮预测时，其先选出<font color="brwon">一组</font>可能性高的候选词，然后按照其概率分布进行随机采样。

- Top-k采样方法

  在每轮预测中都选取K个概率最高的词作为本轮的候选词集合。

  缺点：受候选词分布的方差的影响，方差大时可能“胡言乱语”，方差小时，候选集不够丰富。

- Top-P采样方法：

  Top-P设定阈值p来对候选集进行选取。

**Temperature机制**

Top-K采样和Top-P采样的随机性由语言模型输出的概率决定，不可自由调整。但在<font color="brwon">不同场景</font>中，我们<font color="brwon">对于随机性的要求可能不同</font>。引入Temperature机制可以对解码随机性进行调节。

<img src="../../images/typora-images/image-20250509113132427.png" alt="image-20250509113132427" style="zoom:50%;" />

Temperature越高随机性越高。可以看到T无穷大时，会变成概率为$\frac{1}{K} $或$\frac{1}{|S_p|} $的均匀分布。反之，T趋近于0时，也会将概率值大的输出通过指数放缩得更大，再归一化。

###### 4.2 语言模型评测

（1）内在评测：测试文本通常由与预训练中所用的文本独立同分布的文本构成，不依赖于具体任务。

<font color="brwon">困惑度（Perplexity）</font>，$PPL(s_{test})=P(w_{1:N})^{-\frac{1}{N}}=\sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(w_i|w_{<i})} } $

困惑度减小也意味着熵减，意味着模型“胡言乱语”的可能性降低。

（2）外在评测：测试文本通常包括该任务上的问题和对应的标准答案，其依赖于具体任务。

- 基于统计指标的评测

  <font color="brwon">BLEU </font>(BiLingual Evaluation Understudy)：计算n-gram匹配精度的一种指标

  

  示例：“大语言模型”翻译成英文，生成的翻译为“big  language models”，而参考文本为“large language models”。

  当n=1时，$Pr(g_1)=\frac{2}{3}$。

  当n=2时，$Pr(g_2)=\frac{1}{2}$。

  当N=2时，$BLEU = \sqrt{\frac{1}{2}\cdot\frac{2}{3}}=\sqrt{\frac{1}{3}}$ 

  

- 基于语言模型的评测

  从语义理解的层面进行评测

  1. 基于上下文词嵌入：上下文词嵌入（Contextual Embeddings）向量的相似度。

  2. 基于生成模型：直接利用提示词工程引导LLM输出评测分数。属于无参评价。

     <img src="../../images/typora-images/image-20250511203742088.png" alt="image-20250511203742088" style="zoom:50%;" />

###### 4.3 语言模型的应用

- 直接应用

  语言模型输出的概率值可以直接应用于输入法、机器翻译、对话等任务。

- 间接应用

  语言模型中间产出的文本嵌入可以应用于实体识别、实体检测、文本检索等任务。

## 第二章 大语言模型架构

#### 10 模型架构概览

**涌现能力**：实验发现<font color="brwon">新能力</font>随着<font color="brwon">模型规模</font>提升<font color="brwon">凭空自然涌现</font>出来，因此将其称为<font color="brwon">涌现能力</font>（Emergent Abilities），例如上下文学习、逻辑推理和常识推理等能力。

**扩展法则**：GPT系列模型的性能提升，有着一系列关于<font color="brwon">模型能力与参数/数据规模之间的定量关系</font>作为理论支撑，即<font color="brwon">扩展法则</font>（Scaling Law）。其中以OpenAI提出的Kaplan-McCandlish法则以及DeepMind提出的Chinchilla法则最为著名。

**模型基础**：Transformer灵活的并行架构为<font color="brwon">训练数据和参数的扩展</font>提供了<font color="brwon">模型基础</font>，推动了本轮大语言模型的法则。

> [!NOTE]
>
> 如上一章所述Transformer是模块化的模型

###### 1.1 基于Transformer的三种架构

在<font color="brwon">Transformer</font>的基础上衍生出了<font color="brwon">三种主流模型架构</font>。

- Encoder-only架构

  只选用Transformer中的<font color="brwon">Encoder部分</font>，代表模型为BERT系列。

  <img src="../../images/typora-images/image-20250511212256145.png" alt="image-20250511212256145" style="zoom:50%;" />

- Encoder-Decoder架构

  同时选用Transformer中的<font color="brwon">Encoder和Decoder部分</font>，代表模型为T5、BART等。

  <img src="../../images/typora-images/image-20250511212430573.png" alt="image-20250511212430573" style="zoom:50%;" />

- Decoder-only架构

  只选用Transformer中的<font color="brwon">Decoder部分</font>，代表模型为GPT和LLaMA系列。

  <img src="../../images/typora-images/image-20250511212313060.png" alt="image-20250511212313060" style="zoom:50%;" />

###### 1.2 三种架构对比

<img src="../../images/typora-images/image-20250511212904788.png" alt="image-20250511212904788" style="zoom:50%;" />



#### 11 基于Encoder-only架构的大语言模型

#### 12 基于Encoder-Decoder架构的大语言模型

#### 13 基于Decoder-only架构的大语言模型

#### 14 Mamba原理

## 第三章 Prompt工程

## 第四章 参数高效微调

## 第五章 模型编辑

## 第六章 检索增强生成