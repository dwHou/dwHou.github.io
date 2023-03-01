# DDPM

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

在这篇博文中，我们将更深入地研究去噪扩散概率模型（也称为 DDPM、扩散模型、基于分数的生成模型或简称为自编码器），因为研究人员已经能够使用它们进行高质量的条件(or 非条件)图像/音频/视频生成。 

我们将重温 (Ho et al., 2020) DDPM的原始论文，基于 Phil Wang 的[实现](https://github.com/lucidrains/denoising-diffusion-pytorch)在 PyTorch 中逐步实现它——(原论文为TensorFlow 实现)。 

> 请注意，生成建模的扩散思想实际上已经在 (Sohl-Dickstein et al., 2015) 中引入。 然而，直到 (Song et al., 2019)（斯坦福大学），和之后的(Ho et al., 2020)（谷歌大脑）才独立改进了该方法。

请注意，扩散模型有多种[视角](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw)解读。 在这里，我们采用离散时间（潜在变量模型）视角，但有必要理解其他视角。

好吧，让我们开始吧！



### 一.什么是扩散模型

如果将（去噪）扩散模型与其他生成模型（例如归一化流、GAN 或 VAE）进行比较，它并没有那么复杂：它们都将噪声从一些简单的分布转换为数据样本。 这也是神经网络学习从纯噪声开始逐渐对数据进行去噪的情况。

具体来说包括 2 个过程：

1. 我们选择的固定（或预定义）前向扩散过程 $q$，逐渐将高斯噪声添加到图像中，直到最终得到纯噪声。
2. 学习到的反向去噪扩散过程 $p_θ$，其中训练神经网络从纯噪声开始逐渐对图像进行去噪，直到最终得到实际图像。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230224162710062.png" alt="image-20230224162710062" style="zoom:50%;" />

由 $t$ 索引的前向和反向过程都发生在一定数量的有限时间步长 $T$内（DDPM 作者使用 $T=1000$）。 你从$t=0$开始，在那里你采样了一个真实的图像$\mathbf{x}_0$，来自您的数据分布（假设是来自 ImageNet 的***猫***图像），前向过程在每个时间步 $t$从高斯分布中采样一些噪声，这些噪声被添加到前一个时间步的图像中。 给定足够大的$T$和在每个时间步适宜地添加噪声，您最终会通过在$t=T$处得到所谓的各向同性高斯分布。

### 二.数学形式

因为需要损失函数来定义优化目标，所以我们以更公式化的方式解读扩散过程。

#### 1.前向扩散过程：$q(x_{t}∣x_{t−1})$

基本上，每个新（稍微嘈杂的）图像$\mathbf{x}_t$都是从具有 $\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} $与$\sigma^2_t = \beta_t$ 的<font color="brown">条件高斯分布</font>中提取的。可以通过$\mathbf{x}_{t}=\sqrt{1−β_{t}}*\mathbf{x}_{t-1}+\sqrt{β_t}*ϵ$，其中$\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$实现，两种形式是等价的。

> 这是添加的信号相关噪声。

请注意，$β_{t}$ 在每个时间步长 $t$ 都不是常数（因此也有下标）——事实上，它定义了一个所谓的<font color="brown">“方差表（variance schedule）”</font>，它可以是线性的、二次的、余弦的，等等，我们将进一步看到 （有点像学习率表）。

> 感觉和GDN（广义分歧归一化）有点像。

如果方差表设置得合适，$\mathbf{x}_T$可以是完全的高斯噪声。

> 如果我们知道条件分布$p(\mathbf{x}_{t-1}∣\mathbf{x}_{t})$，然后我们可以反向运行这个过程：通过采样一些随机的高斯$\mathbf{x}_T$，然后不断去噪，最终得到一个真实分布的样本$\mathbf{x}_0$。
>
> 然而，我们不知道 $p(\mathbf{x}_{t-1}∣\mathbf{x}_{t})$。这很棘手，因为它需要知道所有可能图像的分布才能计算这个条件概率。因此，我们将利用<font color="brown">神经网络</font>。

#### 2.反向扩散过程：$p_{\theta}(x_{t-1}∣x_{t})$

我们将利用神经网络来近似（学习）反向的条件概率分布，我们称之为 $p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)$, 其中 $θ$ 是神经网络的参数，通过梯度下降更新。

如果我们假设这个逆向过程也是高斯分布，那么回想一下任何高斯分布都由 2 个参数定义：均值$\mu_\theta$和方差$\Sigma_\theta$

我们可以将该过程参数化为：

$p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t)$，

其中均值和方差也以噪声水平 $t$为条件。

因此，我们的神经网络需要学习/表示均值和方差。 然而，DDPM 作者决定保持方差固定，让神经网络只学习（表示）均值 $\mu_\theta$。

> 后续有论文也学习方差$\Sigma_\theta$。

### 三.定义目标函数

（通过重参数化均值）

为了导出目标函数来学习反向过程的均值$\mu_\theta$，作者观察到 $q$ 和 $p_\theta$ 的组合可以看作是变分自动编码器 ([VAE](https://arxiv.org/abs/1312.6114))。 因此，<font color="brown">变分下界</font>（也称为 ELBO）可用于最小化关于GT数据样本 $\mathbf{x}_0$ 的负对数似然（关于 ELBO 的详细信息，我们参考 VAE 论文）。 事实证明，这个过程的 ELBO 是每个时间步 t 的损失总和，$ L = L_0 + L_1 + ... + L_T$ .

通过前向$q$和后向过程$p_\theta$的构建，损失的每一项（$L_0$除外）实际上是**2个高斯分布**之间的KL散度，可以明确地写成关于均值的$l2$损失！

如 Sohl-Dickstein 等人所示，构造正向过程 $q$ 的直接结果是我们可以以 $\mathbf{x}_0$ 为条件在任意噪声水平下对 $\mathbf{x}_t$ 进行采样（因为高斯的和 也是高斯分布）。 这非常方便：我们不需要重复应用 $q$ 来对 $\mathbf{x}_t$ 进行采样。我们有：

​	                                  $q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$

也就是$T$步的$q$实际一步到位。其中 $\alpha_t := 1 - \beta_t$ ，$ \bar{\alpha}_t := \Pi_{s=1}^{t} \alpha_s$。

> 我们认为这个等式有着“nice property”。 
>
> 1. 这意味着我们可以对高斯噪声进行采样并对其进行适当缩放，并将其添加到 $\mathbf{x}_0$ 中以直接得到 $\mathbf{x}_t$。 请注意，$\bar{\alpha}_t$ 是已知 方差表$\beta_t$ 的函数，因此也是已知的并可以预先计算。 然后，这允许我们在训练期间优化损失函数 $L$ 的随机项（或者换句话说，在训练期间随机采样 $t$ 并优化 $L_t$）。
>
> 2. 如 Ho 等人所示，此属性的另一个优点 是可以（经过一些数学运算后）重新参数化均值以使神经网络学习（预测）在噪声水平 $t$ 添加的噪声（通过网络 $\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)$ )。
>
>    这意味着我们的神经网络成为噪声预测器，而不是均值预测器。 均值可以计算如下：
>    
>    $\mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)$

最终的目标函数 $L_t$ 就会看起来如下：

$\| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t) } \mathbf{\epsilon}, t) \|^2$. 

这里, $\mathbf{x}_0$是初始（真实的、未损坏的）图像，我们看到固定前向过程给出的直接噪声水平 $t$ 样本。 $\mathbf{\epsilon}$ 是在时间步长 $t$ 采样的纯噪声，而 $\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)$ 是我们的神经网络。 神经网络使用真实和预测高斯噪声之间的简单均方误差 (MSE) 进行优化。

训练算法现在可以总结如下：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230301141221490.png" alt="image-20230301141221490" style="zoom:50%;" />

换句话说：

1. 我们从真实未知且可能复杂的数据分布 $q(\mathbf{x}_0)$ 中随机抽取样本 $\mathbf{x}_0$
2. 我们在 $1$ 和 $T$ 之间均匀采样噪声水平 $t$ （即随机时间步长）
3. 我们从高斯分布中采样一些噪声，并在 $t$ 级通过该噪声破坏输入（使用上面提到的“nice property”）
4. 神经网络被训练为根据损坏的图像 $\mathbf{x}_t$（即根据已知时间表 $\beta_t$ 应用于$ \mathbf{x}_0$的噪声）预测此噪声

实际上，所有这些都是在批量数据上完成的，因为人们使用随机梯度下降来优化神经网络。

### 四.神经网络

神经网络需要在特定时间步长处接收噪声图像并返回预测噪声。 请注意，预测噪声是与输入图像具有相同大小/分辨率的张量。 所以从技术上讲，网络接收和输出相同形状的张量。 我们可以为此使用哪种类型的神经网络？

这里通常使用的与自动编码器非常相似，您可能还记得典型的“深度学习入门”教程。 自动编码器在编码器和解码器之间有一个所谓的“瓶颈”层。 编码器首先将图像编码成称为“瓶颈”的较小隐藏表示，然后解码器将该隐藏表示解码回实际图像。 这迫使网络只在瓶颈层保留最重要的信息。

在架构方面，DDPM 作者选择了<font color="brown"> U-Net</font>。 这个网络，像任何自动编码器一样，中间有一个瓶颈，确保网络只学习最重要的信息。 重要的是，它在编码器和解码器之间引入了残差连接，极大地改善了梯度流（受 ResNet [He et al., 2015] 启发）。

### 五.位置向量

由于神经网络的参数跨时间共享（噪声水平），作者受 Transformer 的启发，采用正弦<font color="brown">位置向量（position embeddings）</font>对 $t$ 进行编码。 这使得神经网络“知道”它在哪个特定时间步长（噪声水平）运行。

SinusoidalPositionEmbeddings 模块将形状为 (batch_size, 1) 的张量作为输入（即一批中几个噪声图像的噪声水平），并将其转换为形状为 (batch_size, dim) 的张量，其中 dim 是图像的维数 位置嵌入。 然后将其添加到每个残差块中，我们将进一步介绍。

```Python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```



应用：https://cloud.tencent.com/developer/article/2090159

# 扩散模型综述

参考曾面试的候选人的一篇调研。

## 横向比较：

- GANs 生成对抗网络：

  > 采用生成器合成图像和鉴别器来区分真图像和假图像。

- ARs 自回归模型：

  > 逐像素生成图像，就像在 NLP 中下一个单词预测任务一样。

- flows 流模型：

  > 使用可逆函数 𝑓 将输入图像编码为潜在变量，并通过反向 𝑓 解码潜在变量。

- VAEs 变分自动编码器：

  > 将输入图像编码为遵循高斯分布的潜在变量。

- DMs 扩散模型：

  > 逐步学习将潜在变量解码为图像。 在每一步中，新的潜在变量都从依赖于当前潜在变量的高斯分布中采样。

  <img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230301143418221.png" alt="image-20230301143418221" style="zoom:36%;" />

## 纵向比较：

### DDPM

**前向过程：**<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230301131220228.png" alt="image-20230301131220228" style="zoom:50%;" />

> DDPM 将给定图像$𝑥_0 $逐步编码为高斯噪声$𝑥_𝑇$，这被称为前向过程。

**反向过程：**<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230301131242203.png" alt="image-20230301131242203" style="zoom:50%;" />

:warning:这里$p(x_{t-1}|x_t)$误写为$q(x_{t-1}|x_t)$

> 解码阶段，也是一个反向过程，逆向进行。在这个过程中，𝜇和Σ通过神经网络建模。
>
> 反向过程从 $\cal{N}(0,\mathbf{I})$ 开始。 在每一步中，根据当前噪声图像 $𝑥_𝑡$ 推导高斯分布。 (𝑡 − 1) 步图像是从新分布中采样的。 因此，图像合成过程实际上转化为去噪过程，正如名称：去噪扩散模型。

在反向过程中，𝜇 和 Σ 应该通过神经网络推断。 然而，在前向过程中，细心的读者应该注意到，$𝑞(𝑥_T|𝑥_0)$实际上是一系列已知高斯分布$\cal{N}(\sqrt{1-{\beta}_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$的组合。因此，$𝑞(𝑥_T| 𝑥_0)$也是已知的高斯分布：

​				                                     	$\alpha_t=1 - \beta_t$ ,

​                                                         $\bar{\alpha}_t= \Pi_{s=1}^{t} \alpha_s$ ,

​                                        $q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$ .

**训练：**训练损失很简单：

​                         $\| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t) } \mathbf{\epsilon}, t) \|^2$

> $𝐸$ 是一个噪声估计器，通常是一个由 $𝑡$ 调节的 U-Net。 𝐸旨在预测高斯噪声𝜖。



 
