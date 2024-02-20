按论文的进化顺序来介绍

#### DDPM

DDPM是比较早的diffusion模型，奠定了基础框架，主要做生成任务，不涉及跨模态。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20240220111615058.png" alt="image-20240220111615058" style="zoom:50%;" />

可以看到生成任务是通过去噪完成的，在N个step里渐步地预测噪声，来进行去噪。这在扩散模型里也称为逆过程（去噪过程）。

> 它的速度比较慢，比现在的stable diffusion慢很多，因为它是直接在图像空间的一个降噪。

至于step怎么取，论坛里基于数学推导和经验，贡献了五花八门不少的sampler。一般是递减的，也有复杂、经过精巧设计的。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20240220112230407.png" alt="image-20240220112230407" style="zoom:50%;" />