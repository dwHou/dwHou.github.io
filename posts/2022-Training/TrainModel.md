---
layout: post
category: "machinelearning"
title:  "深度学习调参技巧"
tags: [python, machine learning]
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### 目录

- TOC
{:toc}

---

### 调参

* 调参：
	* trial-and-error
	* 没有捷径可走。有人思考后再尝试，有人盲目尝试。
	* 快速尝试：调参的关键

---

### 大方向

* 【1】**刚开始，先上小规模数据，模型往大了放**（能用256个filter就别用128个），直接奔着过拟合去（此时都可不用测试集验证集）
	* 验证训练脚本的流程。小数据量，速度快，便于测试。
	* 如果小数据、大网络，还不能过拟合，需要检查输入输出、代码是否错误、模型定义是否恰当、应用场景是否正确理解。比较神经网络没法拟合的问题，这种概率太小了。
* 【2】**loss设计要合理**
	* 分类问题：softmax，回归：L2 loss。
	* 输出也要做归一化。如果label为10000，输出为0，loss会巨大。
	* 多任务情况时，各个loss限制在同一个量级上。
* 【3】**观察loss胜于观察准确率**
	* 优化目标是loss
	* 准确率是突变的，可能原来一直未0，保持上千代迭代，突变为1
	* loss不会突变，可能之前没有下降太多，之后才稳定学习
* 【4】**确认分类网络学习充分**
	* 分类 =》类别之间的界限
	* 网络从类别模糊到清晰，可以看softmax输出的概率分布。刚开始，可能预测值都在0.5左右（模糊），学习之后才慢慢移动到0、1的极值
* 【5】学习速率是否设置合理
	* 太大：loss爆炸或者nan
	* 太小：loss下降太慢
	* 当loss在当前LR下一路下降，但是不再下降了 =》可进一步降低LR
* 【6】**对比训练集和验证集的loss**
	* 可判断是否过拟合
	* 训练是否足够
	* 是否需要early stop
* 【7】**清楚receptive field大小**
	* CV中context window很重要
	* 对模型的receptive field要有数
* 【8】**在验证集上调参**

---

### 数据

---

#### 预处理

* -mean/std zero center已然足够，PCA、白化都用不上
* 注意shuffle

---

### 模型本身

* 理解网络的原理很重要，CNN的卷积这里，得明白sobel算子的边界检测
* CNN适合训练回答是否的问题
* google的Inception论文，结构要掌握
* 理想的模型：高高瘦瘦的，很深，但是每层的卷积核不多。很深：获得更好的非线性，模型容量指数增加，但是更难训练，面临梯度消失的风险。增加卷积核：可更好的拟合，降低train loss，但是也更容易过拟合。
* 如果训练RNN或者LSTM，务必保证梯度的norm（归一化的梯度）被约束在15或者5
* 限制权重大小：可以限制某些层权重的最大范数以使得模型更加泛化

---

#### 参数初始化方法

* 用高斯分布初始化
* 用xavier
* word embedding：xavier训练慢结果差，改为uniform，训练速度飙升，结果也飙升。
* 良好的初始化，可以让参数更接近最优解，这可以大大提高收敛速度，也可以防止落入局部极小。
* relu激活函数：初始化推荐使用He normal
* tanh激活函数：推荐使用xavier（Glorot normal）

---

#### 隐藏层的层数

---

#### 节点数目

---

#### filter

* 用3x3大小
* 数量：2^n
* 第一层的filter数量不要太少，否则根本学不出来（底层特征很重要）

#### 激活函数的选取

* 给神经网络加入一些非线性因素，使得网络可以解决较为复杂的问题
* 输出层：
	* 多分类任务：softmax输出
	* 二分类任务：sigmoid输出
	* 回归任务：线性输出
* 中间层：优先选择relu，有效的解决sigmoid和tanh出现的梯度弥散问题
* CNN：先用ReLU
* RNN：优先选用tanh激活函数

---

#### dropout

* 可防止过拟合，人力成本最低的Ensemble
* 加dropout，加BN，加Data argumentation
* dropout可以随机的失活一些神经元，从而不参与训练
* 例子【[Dropout 缓解过拟合](https://morvanzhou.github.io/tutorials/machine-learning/torch/5-03-dropout/)】：
	* 任务：拟合数据点（根据x值预测y值）
	* 构建过拟合网络，比如这里使用了2层，每层节点数=200的网络
	* 使用dropout和不使用dropout，看拟合的效果
	* 可以看到，对于过拟合（模型对训练集拟合得很好）的情况下，使用dropout，能够降低在测试集上的loss，和真实值预测的更贴近。[![20191018141520](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191018141520.png)](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191018141520.png)

---

### 损失函数

---

### 训练相关

---

#### 学习速率

* 在优化算法中更新网络权重的幅度大小
* 可以是恒定的、逐渐下降的、基于动量的或者是自适应的
* 优先调这个LR：会很大程度上影响模型的表现
* 如果太大，会很震荡，类似于二次抛物线寻找最小值
* 一般学习率从0.1或0.01开始尝试
* 通常取值[0.01, 0.001, 0.0001]
* 学习率一般要随着训练进行衰减。衰减系数设0.1，0.3，0.5均可，衰减时机，可以是**验证集准确率不再上升时**，或**固定训练多少个周期以后自动进行衰减**。
* 有人设计学习率的原则是监测一个比例：每次更新梯度的norm除以当前weight的norm，如果这个值在10e-3附近，且小于这个值，学习会很慢，如果大于这个值，那么学习很不稳定。
* ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191015000514.png)
	* 红线：初始学习率太大，导致振荡，应减小学习率，并从头开始训练
	* 紫线：后期学习率过大导致无法拟合，应减小学习率，并重新训练后几轮
	* 黄线：初始学习率过小，导致收敛慢，应增大学习率，并从头开始训练

---

#### batch size大小

* 每一次训练神经网络送入模型的样本数
* 可直接设置为16或者64。通常取值为：[16, 32, 64, 128]
* CPU讨厌16，32，64，这样的2的指数倍（为什么？）。GPU不会，GPU推荐取32的倍数。

---

#### momentum大小

* 使用默认的0.9

---

#### 迭代次数

* 整个训练集输入到神经网络进行训练的次数
* 当训练集错误率和测试错误率想相差较小时：迭代次数合适
* 当测试错误率先变小后变大：迭代次数过大，需要减小，否则容易过拟合

---

#### 优化器

* 自适应：Adagrad, Adadelta, RMSprop, Adam
* 整体来讲，Adam是最好的选择
* SGD：虽然能达到极大值，运行时间长，可能被困在鞍点
* Adam: 学习速率3e-4。能快速收敛。

---

#### 残差块和BN

* 残差块：可以让你的网络训练的更深
* BN：加速训练速度，有效防止梯度消失与梯度爆炸，具有防止过拟合的效果
* 构建网络时最好加上这两个组件

---

### 案例：人脸特征点检测

这篇文章([Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/))是14年获得kaggle人脸特征点检测第二名的post，详细的描述了如何构建模型并一步步的优化，里面的一些思路其实很直接干脆，可以参考一下。

* 任务：人脸特征点检测
* 样本：7049个96x96的灰度图像，15个特征点坐标（15x2=30个坐标值）
* 探索：
	* 全部特征点都有label的样本有多少？
	* 每个特征点对应的有label的样本有多少？
* 预处理：
	* 灰度图像归一化：[0,1]
	* 坐标值(y值)归一化：[-1,1]
* 构建简单的单层网络【net1】，发现是过拟合的（训练集的loss比验证集的loss低很多）
* 网络过于简单？使用卷积神经网络【net2】。卷积神经网络效果好很多此时对应的训练集和验证集的loss都比【net1】低很多，且loss比较平滑，但是模型还是过拟合的。
	* ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191016001832.png)
* 预防过拟合？使用更多的数据，这里是kaggle提供的数据集，没办法获得更多的数据，又是图像数据，可以使用数据增强【net3】。
	* 比如最简单的是图像水平翻转
	* 这里检测特征点，注意翻转前后特征点也要变化，比如原来的左眼左角，水平翻转之后变成了右眼的右角。
	* 这里的翻转不需要重新生成数据集，可在load batch数据时做 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191016002714.png)transformation转换(很方便)
* 网络本身的可调参数，比如学习率或者动量大小，之前用的是默认值，是否还有优化的余地？学习率和动量的动态变化，对于上面的网络都进行这两个参数修改的尝试：卷积+学习率/动量【net4】，卷积+数据增强+学习率/动量【net5】
	* 学习率：开始学习使用大的学习率，然后在训练过程中慢慢较小。比如：回家先坐火车，快到家门时改用走路。
	* 动量大小：在训练过程中增加动量的大小 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191016002623.png)
* dropout【net6】是否能进一步减小过拟合？
	* 一定要先训练得好，最好过拟合，然后再做正则化
	* 注意此时训练集和验证集的loss的比值，前者是带有dropout计算出来的loss，后者的loss是没有的
* 增大训练的次数似乎可以降低loss【net7】？
	* dropout之后，其实loss可以进一步减小的
	
```
Name  |   Description    |  Epochs  |  Train loss  |  Valid loss
-------|------------------|----------|--------------|--------------
 net1  |  single hidden   |     400  |    0.002244  |    0.003255
 net2  |  convolutions    |    1000  |    0.001079  |    0.001566
 net3  |  augmentation    |    3000  |    0.000678  |    0.001288
 net4  |  mom + lr adj    |    1000  |    0.000496  |    0.001387
 net5  |  net4 + augment  |    2000  |    0.000373  |    0.001184
 net6  |  net5 + dropout  |    3000  |    0.001306  |    0.001121
 net7  |  net6 + epochs   |   10000  |    0.000760  |    0.000787
```

* 单独的训练？
	* 上面训练时是并行的，同时检测15个特征点
	* 只使用了15个特征点都有坐标值的样本，其他数据（70%）都被丢掉了，这是很影响最后模型的泛化效果的，尤其是当去预测测试集样本的时候
	* 可以每个特征点单独进行分类器训练，就使用上面最好的模型，此时对应的样本数目也是更多的
	* 这里用了一个训练时设置的字典型变量，遍历特征点训练，可以参考
	* 此时，对于每个特征，可能需要训练的次数不用那么多，那么可以设置early stopping（这里n=200）
	* 最后拿到了2.17的RMSE，直接飙升到底2名 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191016003534.png)
* 使用pre-train的模型？
	* 网络参数的初始化也很重要
	* 可以使用上面训练好的网络，比如net6或者net7的参数进行初始化
	* 这也是一个正则化的效果
	* 此时的RMSE降低了一点到2.13，仍然是第2名 ![](https://raw.githubusercontent.com/Tsinghua-gongjing/blog_codes/master/images/20191016003820.png)

---

## 参考

* [总结知乎深度学习调参技巧](https://blog.csdn.net/m0_37644085/article/details/88956758)
* [深度学习调参技巧](https://zhuanlan.zhihu.com/p/51556033)
* [A Recipe for Training Neural Networks @Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
* [Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](https://pdfs.semanticscholar.org/688a/3745525ad64b72e14dce36df44d69637fda0.pdf)
* [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
* [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)


---

