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

###### 4.1 Coord MLP

<img src="../../../images/typora-images/image-20230711143052516.png" alt="image-20230711143052516" style="zoom:50%;" />

其实在NeRF之前，NIPS2020的一篇[论文](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)有这样一种尝试：

希望通过一个神经网络，就是由全连接层和非线性函数组成，建模一张图片的表示：坐标到rgb的映射关系。相当于背下来了这张图。

但是直接这种<font color="brown">背图片</font>，大概只能恢复成下方图片的样子，很糊，还有很多三角形的artifacts（大概是因为非线性函数用的ReLU，如果换一种非线性函数，会是另一种风格）。

> 一般的神经网络，如果没有经过专门设计，只能学到一些非常低频的信息。

<img src="../../../images/typora-images/image-20230711151348423.png" alt="image-20230711151348423" style="zoom:50%;" />

**论文解读** | Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains [*](https://zhuanlan.zhihu.com/p/452324858)

首先理解傅里叶变换：

<video src="./d223de46-23a7-11eb-95ec-fe27a5b7ef44.mp4"></video>

理解神经正切核(neural tangent kernel)：

一句话总结：<font color="brown">NTK衡量的是，在使用SGD优化参数下，其对应的随机到样本 $x^′$ ，在参数更新非常一小步 $ \eta$ 后， $f(x)$ 的变化。</font>

比如$f(x) = \theta_1x + \theta_2$, 初始化参数$\theta_1=3$, $\theta_2=1$，

和$f(x) = \theta_1x + 10*\theta_2$, 初始化参数$\theta_1=3$, $\theta_2=0.1$，

看起来二者初始化时对于某个样本的Loss一样，但SGD优化参数后，f(x)的变化是不一样的：

假设有一个样本是 (x,y)=(10,50)，学习率0.1，使用squared error loss : 

$((10*3+1) - 50 )^2$求导 = 2*(31 - 50) = -38

更新后的f(10) = 31 - 0.1*38 = 34.8，且根据求导关系，$\theta_1$更新的变化比 $\theta_2$大十倍。

$(3+10*\delta)*10+1+\delta=34.8, \delta=0.03762376$

新的$\theta_1=3.3762376$, $\theta_2=1.03762376$，如下图：

![img](https://www.inference.vc/content/images/2020/11/download-37.png)

而$f(x) = \theta_1x + 10*\theta_2$，初始化参数$\theta_1=3$, $\theta_2=0.1$，则根据$(3+10*\delta)*10+10*(0.1+10*\delta)=34.8, \delta=0.019$，

新的$\theta_1=3.19$, $\theta_2=0.29$，如下图：

![img](https://www.inference.vc/content/images/2020/11/download-39.png)

GeoGebra作图：

<img src="./NTK.png" alt="NTK" style="zoom:50%;" />

也就是说，NTK对参数是敏感的。

再来举一个小型神经网络的例子，同样的，在样本点 (x,y)=(10,50) 更新这么一个函数，我们得到函数的变化为：

![img](https://pic4.zhimg.com/80/v2-44a87cc7237eb09dae6513e8a4e23bbf_1440w.webp)

显然，我们发现，在靠近0附近它的变化是很小的，而在10附近它的变化是很大的，之前说过，NTK就是刻画这种变化的，因此，我们可以把NTK画出来：

![img](https://www.inference.vc/content/images/2020/11/download-43.png)

值得一提的是，虽然样本是 $x=10$ 的点，但是变化最大的地方其实是在 $x=7$ 的地方。

那如果我们不停的更新参数会怎样？以下是更新15次的图：

![img](https://www.inference.vc/content/images/2020/11/download-45.png)

![img](https://www.inference.vc/content/images/2020/11/download-46.png)

显然，随着参数的变化，kernel大小也在变化，而且越来越平滑，这意味着函数在每个取值下的变化越来越一致。

NTK的形式把loss function的作用和NN结构的作用分离开了，NN的结构贡献给了NTK。

现在正式阅读这篇==Coord MLP==:

