## NeRF原理理解

### 通俗版

核心逻辑：无非就是camera pose作为输入，real image作为输出监督，从而得到一个场景的隐式表示。

### 自学版

[youtube link](https://www.youtube.com/watch?v=TQj-KUQophI&t=6s)

[bilibili link](https://www.bilibili.com/read/cv23979250?spm_id_from=333.999.0.0)

#### 一.概念

NeRF是视角生成算法，从少数相片（real camera）中学习生成新视角（virtual camera）。

可以实现5D的神经辐射场（照射野），其中3维用来表示我们在哪，2维表示我们看向哪。

> 3 dimensions for location and 2 for view direction.  

对应神经网络的输入也就是这5个维度，输出是图片。

#### 二.难点

主要是那些稍许改变视角，就会大幅变化的事物：

- non-Lambertian材质：无光泽的物体比较简单，因为它们从不同方向看都是相似的。而有光泽的物体就难多了，因为视角变化时它们的表面也会大幅变化，基本是训练数据中没见过的。
- 遮挡：需要学习到准确的深度信息。

#### 三.知识系统

##### 1.概览

<img src="../../../images/typora-images/image-20230706170431287.png" alt="image-20230706170431287" style="zoom:50%;" />

计算机视觉输入是一张图片，然后通过各种方法，来了解和认识这张图片。

计算机图形学的目标则恰恰相反，已经对一个场景有认识了，然后想通过渲染获得图片。

##### 2.应用

- 渲染
- 导出其他格式，比如mesh
- 参与其他的计算机视觉任务，比如已有场景建模，再做深度估计、目标检测、反求观测者位置（iNeRF）

##### 3.体积渲染

在NeRF里可微的渲染很重要，而我们介绍的工作用到的都是可微的体积渲染（Volume Rendering）

<img src="../../../images/typora-images/image-20230711141527195.png" alt="image-20230711141527195" style="zoom:50%;" />

猪骨密度>猪脂肪/皮肤>空气，这里把密度再理解为透光率：

<img src="../../../images/typora-images/image-20230711142005506.png" alt="image-20230711142005506" style="zoom:50%;" />

<img src="../../../images/typora-images/image-20230711142026375.png" alt="image-20230711142026375" style="zoom:50%;" />

光路可逆，这里假设相机接收光，或者发射光是一样的结论。某束光线沿直线传播，透射比（transmittance）就会在经过猪骨头时大幅下降。

体积渲染最后时用离散的采样点代替了连续的积分。

##### 4.NeRF

<img src="../../../images/typora-images/image-20230711142503862.png" alt="image-20230711142503862" style="zoom:50%;" />

神经网络形成了场景表示。

<img src="../../../images/typora-images/image-20230711143052516.png" alt="image-20230711143052516" style="zoom:50%;" />

其实在NeRF之前，NIPS2020的一篇[论文](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)有这样一种尝试：

希望通过一个神经网络，就是由全连接层和非线性函数组成，建模一张图片的表示：坐标到rgb的映射关系。相当于背下来了这张图。

但是直接这种<font color="brown">背图片</font>，大概只能恢复成下方图片的样子，很糊，还有很多三角形的artifacts（大概是因为非线性函数用的ReLU，如果换一种非线性函数，会是另一种风格）。

> 一般的神经网络，如果没有经过专门设计，只能学到一些非常低频的信息。

<img src="../../../images/typora-images/image-20230711151348423.png" alt="image-20230711151348423" style="zoom:50%;" />