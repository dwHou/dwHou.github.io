背景：越来越多的场景用Rectified Flow代替DDPM。

https://event.baai.ac.cn/activities/814

在这次演讲中，我将讨论矫正流(Rectified Flow)。这个算法出奇的简单,它解决了使用非配对数据点学习两个分布之间传输映射的问题。这种问题包括生成式模型和无监督数据迁移。矫正流符合一个常微分方程(ODE),它被训练尽可能沿着直线路径前进,仅使用监督学习和L2目标函数。矫正流有一种特殊的操作，称为重流(Reflow)。它通过迭代地拉直概率流轨迹,同时改善噪声和数据点之间的配对来提高生成速度。此外,我还将分享我们如何将矫正流框架扩展到大规模LAION数据集和文生图模型Stable Diffusion,从而得到一个强大的一步式文生图模型,名为InstaFlow。这显示了矫正流框架在训练基础模型方面的潜力。



## 利用直线概率流加速SD的训练和推理

《Ultra-Fast Stable Diffusion from Straight Probability Flows》

<img src="/Users/devonn.hou/Library/Application Support/typora-user-images/image-20250212164506003.png" alt="image-20250212164506003" style="zoom:50%;" />

### 算法框架RF

DDPM、Score-based Model是SDE；

DDIM等提出了曲线的ODE（向随机过程中增加确定项，而从SDE转换而来）；

RF进一步利用直线的生成式ODE。



### RF应用