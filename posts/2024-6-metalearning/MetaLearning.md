# 元学习

> [!NOTE]
>
> - [火炉课堂 | 元学习(meta-learning)到底是什么鬼？-_bilibili](https://link.juejin.cn?target=https%3A%2F%2Fwww.bilibili.com%2Fvideo%2FBV1sX4y1w7Tz%2F%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D98399e311a67b69a21656bbd6fc533d0)
> - [李宏毅 | 元学习 meta Learning & few-shot learning 少样本学习 - bilibili](https://link.juejin.cn?target=https%3A%2F%2Fwww.bilibili.com%2Fvideo%2FBV1KF41167VZ%2F%3Fspm_id_from%3D333.337.search-card.all.click)

元学习通俗的来说，就是**去学习如何学习（Learning to learn）,掌握学习的方法**，有时候掌握学习的方法比刻苦学习更重要！

## 1. 概念

何为“元”（What is Meta?  ）

When meta is used as a prefix, meta-X means "beyond-X", "after-X", or <font color="blue">"X about X"</font>. 

时间线：传统机器学习 < model learning > →  深度学习 < joint feature and model learning> → 元学习 < joint feature, model, and algorithm learning >

> 不断解放程序员的手工设计劳动

和AutoML的区别：元学习可以是AutoML的一个手段。但元学习侧重learning（优化），AutoML可以只是tuning（试错）。

>  [!IMPORTANT]
>
> 元学习 (Meta-Learning) 通常被理解为“学会学习 (Learning-to-Learn)”， 指的是在多个学习阶段改进学习算法的过程。 在基础学习过程中， 内部（或下层/基础）学习算法解决由数据集和目标定义的任务。 在元学习过程中，外部（或上层/元）算法更新内部学习算法，使其学习的模型改进外部目标

### 1.1 定义

两类：单任务元学习 & 多任务元学习

1. **单任务元学习：为任务找到一个最佳的算法$F_\omega$**

比如下图中$A$是元学习算法，对于训练数据最佳的模型$f_\theta$。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20240629184751344.png" alt="image-20240629184751344" style="zoom:50%;" />

>$\omega$ 表示算法里面可学的参数，比如用哪种基础网络结构、什么超参或哪个优化器等等，$\omega$ 的玩法非常多。

2. **多任务元学习：找到能适用于新任务的$F_\omega$**

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20240629185842675.png" alt="image-20240629185842675" style="zoom:50%;" />

> 学习处理新任务的能力。

>  [!TIP] 
>
> 可以理解成 传统学习目标是 generalizing over data, 多任务元学习是 generalizing over tasks.

>  [!IMPORTANT]
>
> 元学习的含义有两层， 第一层是让机器学会学习，使其具备分析和解决问题的能力， 机器通过完成任务获取经验，<font color="purple">**提高**</font>完成任务的能力; 第二层是让机器学习模型可以更好地<font color="purple">**泛化**</font>到新领域中， 从而完成差异很大的新任务。
>
> Few-Shot Learning 是 Meta-Learning 在监督学习领域的应用。 在 Meta-training 阶段， 将数据集分解为不同的任务，去学习类别变化的情况下模型的泛化能力。 在 Meta-testing 阶段， 面对全新的类别，不需要变动已有的模型，只需要通过一部或者少数几步训练，就可以完成需求。

### 1.2 元学习单位

>  [!IMPORTANT]
>
> 元学习的基本单元是任务，任务结构如图1所示。 元训练集 (Meta-Training Data)、元验证集 (Meta-Validation Data) 和元测试集 (Meta-Testing Data) 都是由抽样任务组成的任务集合。 元训练集和元验证集中的任务用来训练元学习模型， 元测试集中的任务用来衡量元学习模型完成任务的效果。
>
> 在元学习中，之前学习的任务称为元训练任务 (meta-train task)， 遇到的新任务称为元测试任务 (meta-test task)。 每个任务都有自己的训练集和测试集， 内部的训练集和测试集一般称为支持集 (Support Set) 和查询集 (Query Set)。 支持集又是一个 N-Way K-Shot 问题，即有 N 个类别，每个类有 K 个样例。



<p align="center"> <img src="https://paddlepedia.readthedocs.io/en/latest/_images/Task.png" alt="Markdown Logo" style="zoom:100%;> <figcaption style="text-align:center;">图1 任务结构</figcaption> </p>

### 1.3 基学习器和元学习器

>  [!IMPORTANT]
>
> 元学习本质上是层次优化问题 (双层优化问题 Bilevel Optimization Problem)， 其中一个优化问题嵌套在另一个优化问题中。 外部优化问题和内部优化问题通常分别称为上层优化问题和下层优化问题， 如图2所示的MAML。

<p align="center"> <img src="https://paddlepedia.readthedocs.io/en/latest/_images/BilevelOptimization.png" alt="Markdown Logo" style="zoom:70%;> <figcaption style="text-align:center;">图2 双层优化元学习 MAML</figcaption> </p>