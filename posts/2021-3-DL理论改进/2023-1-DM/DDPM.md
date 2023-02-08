# 扩散模型

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

在这篇博文中，我们将更深入地研究去噪扩散概率模型（也称为 DDPM、扩散模型、基于分数的生成模型或简称为自编码器），因为研究人员已经能够使用它们进行高质量的条件(or 非条件)图像/音频/视频生成。 

我们将重温 (Ho et al., 2020) DDPM的原始论文，基于 Phil Wang 的[实现](https://github.com/lucidrains/denoising-diffusion-pytorch)在 PyTorch 中逐步实现它——(原论文为TensorFlow 实现)。 

> 请注意，生成建模的扩散思想实际上已经在 (Sohl-Dickstein et al., 2015) 中引入。 然而，直到 (Song et al., 2019)（斯坦福大学），和之后的(Ho et al., 2020)（谷歌大脑）才独立改进了该方法。

请注意，扩散模型有多种[视角](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw)解读。 在这里，我们采用离散时间（潜在变量模型）视角，但一定要检查其他视角。

好吧，让我们开始吧！



### 一.什么是扩散模型

如果将（去噪）扩散模型与其他生成模型（例如归一化流、GAN 或 VAE）进行比较，它并没有那么复杂：它们都将噪声从一些简单的分布转换为数据样本。 这也是神经网络学习从纯噪声开始逐渐对数据进行去噪的情况。

