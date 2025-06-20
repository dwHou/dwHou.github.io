## NeRF原理理解

### 通俗版

核心逻辑：无非就是camera pose作为输入，real image作为输出监督，从而得到一个场景的隐式表示。

### 自学版

[youtube link](https://www.youtube.com/watch?v=TQj-KUQophI&t=6s)

[bilibili link](https://www.bilibili.com/read/cv23979250?spm_id_from=333.999.0.0)

#### 一.概念

NeRF是视角生成算法，从少数相片（real camera）中学习生成新视角（virtual camera）。

可以实现5D的神经辐射场（照射野），其中3维用来表示采样点在哪，2维表示从哪看向采样点（光路可逆）。

> 3 dimensions for location and 2 for view direction.  

对应神经网络的输入也就是这5个维度，输出是图片。

#### 二.难点

主要是那些稍许改变视角，就会大幅变化的事物：

- 非朗伯（non-Lambertian）材质：无光泽的物体比较简单，因为它们从不同方向看都是相似的。而有光泽的（non-Lambertian）物体就难多了，因为视角变化时它们的表面也会大幅变化，基本是训练数据中没见过的。
- 遮挡：需要学习到准确的深度信息。

#### 三.知识系统

##### 1.概览

<img src="../../../images/typora-images/image-20230706170431287.png" alt="image-20230706170431287" style="zoom:50%;" />

计算机视觉输入是一张图片，然后通过各种方法，来了解和认识这张图片。

计算机图形学的目标则恰恰相反，已经对一个场景有认识了，然后想通过渲染获得图片。

##### 2.应用

- 渲染

- 导出其他格式，比如mesh

  > 现实中的3D数据主要有面数据、点数据、体数据，所以对应催生了一些Mesh、Point Cloud、Volume等中间表示。
  >
  > NeRF选择Volume作为场景的中间表达

- 参与其他的计算机视觉任务，比如已有场景建模，再做深度估计、目标检测、反求观测者位置（iNeRF）

##### 3.体积渲染

在NeRF里可微的渲染很重要，而我们介绍的工作用到的都是可微的体积渲染（Volume Rendering）

<img src="../../../images/typora-images/image-20230711141527195.png" alt="image-20230711141527195" style="zoom:50%;" />

猪骨密度>猪脂肪/皮肤>空气，这里把密度再理解为透光率：

<img src="../../../images/typora-images/image-20230711142005506.png" alt="image-20230711142005506" style="zoom:50%;" />

<img src="../../../images/typora-images/image-20230711142026375.png" alt="image-20230711142026375" style="zoom:50%;" />

光路可逆，这里假设相机接收光，或者发射光是一样的结论。某束光线沿直线传播，透射比（transmittance）就会在经过猪骨头时大幅下降。

体积渲染最后时用离散的采样点代替了连续的积分。

##### 4.NeRF之位置编码

<img src="../../../images/typora-images/image-20230711142503862.png" alt="image-20230711142503862" style="zoom:50%;" />

神经网络形成了场景表示。

> <font color="brown">Original Paper:</font> transform input 5D coordinates with a positional encoding that enables the MLP to represent higher frequency functions.

###### 4.1 Coord MLP

<img src="../../../images/typora-images/image-20230711143052516.png" alt="image-20230711143052516" style="zoom:50%;" />

其实在NeRF之前，NIPS2020的一篇[论文](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)有这样一种尝试：

希望通过一个神经网络，就是由全连接层和非线性函数组成，建模一张图片的表示：坐标到rgb的映射关系。相当于背下来了这张图。

但是直接这种<font color="brown">背图片</font>，大概只能恢复成下方图片的样子，很糊，还有很多三角形的artifacts（大概是因为非线性函数用的ReLU，如果换一种非线性函数，会是另一种风格）。

> 一般的神经网络，如果没有经过专门设计，只能学到一些非常低频的信息。

<img src="../../../images/typora-images/image-20230711151348423.png" alt="image-20230711151348423" style="zoom:50%;" />

解决办法是引入对坐标的编码（<font color="brown">positional encoding</font>）。

> 我们需要这样一种位置表示方式，满足于：
> （1）它能用来表示一个token在序列中的绝对位置
> （2）在序列长度不同的情况下，不同序列中token的相对位置/距离也要保持一致
> （3）可以用来表示模型在训练过程中从来没有看到过的句子长度。



###### 4.2 PE of Transformer

> Transfomer里input = input_embedding + positional_encoding，这里，input_embedding是通过常规embedding层，将每一个token的向量维度从vocab_size映射到d_model，由于是相加关系，自然而然地，这里的positional_encoding也是一个d_model维度的向量。（在原论文里，d_model = 512）
>
> 直觉上，位置编码的高冗余特征实际是一种纠错编码，靠高度冗余保证位置信息在多个信息通道上稳定灌入网络而不至于被网络抛弃。

直观地想法是用d_model长度的二进制编码位置。

<img src="../../../images/typora-images/image-20230724161739747.png" alt="image-20230724161739747" style="zoom:30%;" />

但这样编码出来的位置向量，处在一个离散的空间中，不同位置间的变化是不连续的。

所以我们相当用周期函数（sin）来表示位置：

$PE_t = [sin(\frac{1}{2^0}t),sin(\frac{1}{2^1}t),...,sin(\frac{1}{2^{i-1}}t),...,sin(\frac{1}{2^{d_{model}}-1}t)]$

但$y＝sin(ωx＋θ)+K$的周期$T=2\pi/ω$，如果我们频率$f=1/T$偏大，引起波长偏短，则不同t下的位置向量可能出现重合的情况。比如在下图中(d_model = 3），图中的点表示每个token的位置向量，颜色越深，token的位置越往后，在频率偏大的情况下，位置响亮点连成了一个闭环，靠前位置（黄色）和靠后位置（棕黑色）竟然靠得非常近：

<img src="https://picx.zhimg.com/80/v2-a76698f2225c5ed7bc49b2259548cf2a_1440w.webp?source=1940ef5c" alt="img" style="zoom:50%;" />

为了避免这种情况，我们尽量将函数的波长拉长。一种简单的解决办法是同一把所有的频率都设成一个非常小的值。因此在transformer的论文中，采用了 $\frac{1}{10000^{i/(d_{model}-1)}}$这个频率。

总结一下，到这里我们把位置向量表示为：

$PE_t = [sin(ω_0t),sin(ω_1t),...,sin(ω_{i-1}t),...,sin(ω_{d_{model}-1}t)]$，其中$ω_i=\frac{1}{10000^{i/(d_{model}-1)}}$，频率被调到非常小（只用看最右边的，最小频率足够小了，周期=10000，而机器翻译一般最大输入文本词量也不超过5000）。

> 这儿也能看出位置编码是冗余的，实际上只需要找到周期最大的sin、cos，就能找到对应的x、y。

> Attention is all you need文章里表示，他们也尝试了sequence2sequence里基于学习的位置编码，但最终发现几乎是等价的。
>
> **从方法的可理解性上**，Learned Positional Embedding更加易于理解。**从参数维度上**，使用Sinusoidal Position Encoding不会引入额外的参数。

那现在我们对位置向量再提出一个要求，**不同的位置向量是可以通过线性转换得到的**。这样，我们不仅能表示一个token的绝对位置，还可以表示一个token的相对位置，即我们想要：

$PE_{t+\bigtriangleup t}=T_{\bigtriangleup t}*PE_t$

1. 绝对位置0的编码是固定的：PE(p=0)=(0,1,0,1,.....) ，所以谁都知道想参考句首的信息应该怎么办

2. 固定的相对距离，位置编码的相似度是固定的：DotProduct(PE(i), PE(i+k)) = DotProduct(PE(j), PE(j+k)) for any i,j,k

3. 位置编码的相似度随着距离的变化单调变化，如DotProduct(PE(i), PE(i+k1)) > DotProduct(PE(i), PE(i+k2)) for any k1<k2

4. 任意两个位置编码之间都可以用一个线性变化互相求得。

   $\begin{aligned}PE_{pos+k,2i} &= \sin(\omega_i pos + \omega_i k) \\ &= \cos(\omega_i pos)\sin(\omega_i k) + \sin(\omega_i pos)\cos(\omega_i k) \\ &= (PE_{pos,2i+1})\mu  + (PE_{pos,2i})\nu \\ &= (PE_{pos,2i}, PE_{pos,2i+1})(\nu, \mu)\end{aligned}$

   

   $\begin{aligned}PE_{pos+k,2i+1} &= \cos(\omega_i pos + \omega_i k) \\ &= \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k) \\ &= (PE_{pos,2i+1})\nu  - (PE_{pos,2i})\mu \\ &= (PE_{pos,2i}, PE_{pos,2i+1})(-\mu, \nu)\end{aligned}$

   

   把上面的式子变为矩阵形式：

   $\begin{pmatrix}
     PE_{pos+k,2i} \\
     PE_{pos+k,2i+1}
   \end{pmatrix} = \begin{pmatrix}
     \nu & \mu \\
     -\mu & \nu
   \end{pmatrix} \times \begin{pmatrix}
     PE_{pos,2i} \\
     PE_{pos,2i+1}
   \end{pmatrix} = \begin{pmatrix}
     \cos & \sin \\
     -\sin & \cos
   \end{pmatrix} \times \begin{pmatrix}
     PE_{pos,2i} \\
     PE_{pos,2i+1}
   \end{pmatrix}$​

   **由sin/cos构成的这个矩阵是个标准的二维旋转矩阵!**

   PE这里用“角度”替代了“距离”来表述“差异”

   > 记得线性代数里是如何表述高维向量之间的差异么？两个矢量的[点积](https://www.zhihu.com/search?q=点积&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2931272773})，反映了cos(angle)的大小，也就等价于反映了angle的大小，也就等价于反映了两个矢量的相似度（差异程度）。

   <img src="https://pic2.zhimg.com/80/v2-a19cc7611d847582f3575d5fdc90afb1_1440w.webp" alt="img" style="zoom:50%;" />

   PE沿着position滑动，就变成了矢量沿着单位圆匀速转动而已。

   这个精巧的设计满足了transfomer里对Positional Encoding的<font color="brown">几项要求</font>：

   1. dimension的范围可以自由改变，向右追加或者减少频率更慢的轮子
   2. position的范围可以自由改变，增大或者减小轮子的半径
   3. 字间距离单位不受position范围的变化而变化，每个轮子的转速恒定
   4. 这么多速度不一的轮子，可以组合出很多唯一的编码

   最后Word Embedding + Position Embedding当作attention的输入，PE+WE很类似于在通过傅立叶变换在频域里添加水印的过程。

   >**这样的编码方案下，Attention就具备了考虑位置的能力。既能搞定长程依赖，也能搞定短程依赖，所以Attention is all you need!**

   >你可以把一个输入的句子拆开成每个token，把每个单独的token看成图里的一个node。如果只有attention机制但是没有PE的话，相当于是在每个node之间都连了边，连成了一张全连接图。所以只看输入输出的句子的话，实际上整个句子（Graph）所携带的信息和每个词（node）的顺序是无关的。因此我们要加入PE来让整个输入的信息是order variant的。并且这个PE需要具有<font color="brown">平移不变性，单调性和对称性</font>。

   > 2017年Attention is all you need paper里用的sinusoidal PE是用了三角函数的周期性来实现这三个性质的。后面其实还有了PE的很多变种，比如BERT就直接通过学习来实现adaptive encoding了（learnable PE）。

参考：[1](如何理解Transformer论文中的positional encoding，和三角函数有什么关系？ - 猛猿的回答 - 知乎 https://www.zhihu.com/question/347678607/answer/2301693596), [2](https://zhuanlan.zhihu.com/p/621307748), [3](哪位大神能讲一下Transformer的Positional embedding为什么有用？ - Olivia的回答 - 知乎 https://www.zhihu.com/question/385895601/answer/1924217518), [4-en-source](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#the-intuition), [5-收音机类比](https://zhuanlan.zhihu.com/p/644616193)

$PE(pos, k)=\left \{ 
\begin{array}{rcl}
\sin(\omega_i pos) & & k=2i\\ 
\cos(\omega_i pos) & & k=2i+1\\ 
\end{array}
\right .$,  其中$\omega_i=\frac{1}{10000^{2i/d_{model}}}$

<font color="brown">和傅立叶变换的区别和相似：</font>

把PE的颜色恢复成三维空间的波形幅度后，它就非常接近傅立叶变换里的多列波形图，这两张图有些区别，这个傅立叶变换图中波的频率是从最右列向最最左列逐渐降低，而上面的PE图则是频率从最左列向最右列逐渐降低，另外，傅立叶变换图里的每条波的幅度范围不同，而PE里都是相同的。

<img src="https://pic4.zhimg.com/80/v2-41113c5e88ff429c3a013ab8752f369b_1440w.webp" alt="img" style="zoom:50%;" />

**PE性质验证代码：**

```python
def pe(p, d):
    e = [.0] * d
    for i in range(d//2):
        e[2 * i] = math.sin(p / 10000 ** (2 * i / d))
        e[2*i+1] = math.cos(p / 10000 ** (2 * i / d))
    return e


d = 64; el=[]
for p in range(0,d*2):
    e = pe(p, d)
    el += [e]
    e = [f"{x:+.2f}" for x in e]
    print(f"d={d} p={p:2d} e={e}")

#el的元素是每个位置的位置向量e
for i,ei in enumerate(el):
    sl = []
    #ei和位置间隔1~12的位置向量的相似度
    for j,ej in enumerate(el[i+1:i+12]):
        s = sum([xi*xj for xi,xj in zip(ei,ej)])
        sl += [s]
    sl = [f"{s:+.2f}" for s in sl]
    print(f"i={i:2d} sl={sl}")
```

<font color="darkblue">你可以用这段代码反推一下，如果不用sin和cos交替来表示位置会怎么样，如果频率设置大了会怎么样。</font>

最终位置编码可视化出来是（序列长度p_max为50，位置编码维度d_model为128）：

<img src="https://d33wubrfki0l68.cloudfront.net/ef81ee3018af6ab6f23769031f8961afcdd67c68/3358f/img/transformer_architecture_positional_encoding/positional_encoding.png" alt="Sinusoidal position encoding" style="zoom:50%;" />

> 越往右的位置，频率越小，波长越长，所以不同的t对最终的结果影响不大。

**Transformer中有两个关键的基础组件：1. Multi-Head Attention， 2. Positional Encoding。**

Multi-Head Attention给予了Transfomer以捕获长程依赖（Long-range dependencies）和内容依赖的能力。Positional Encoding给予了Transfomer捕获短程依赖和位置依赖的能力。由于Positional Encoding也是被合并到Embeding中之后通过Multi-Head Attention起作用的，所以Transfomer的论文，题为*Attention is All You Need*。

4.1之 **论文解读** | Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains [*](https://zhuanlan.zhihu.com/p/452324858) 

和NeRF是同一批作者。

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

（这里把NTK除以NTK在10处的值，有点像normalize，只是为了scale一下y，方便观察。）

值得一提的是，虽然样本是 $x=10$ 的点，但是变化最大的地方其实是在 $x=7$ 的地方。

那如果我们不停的更新参数会怎样？以下是更新15次的图：

![img](https://www.inference.vc/content/images/2020/11/download-45.png)

![img](https://www.inference.vc/content/images/2020/11/download-46.png)

显然，随着参数的变化，kernel大小也在变化，而且越来越平滑，这意味着<font color="brown">函数在每个取值下的变化越来越一致</font>。

NTK的形式把loss function的作用和NN结构的作用分离开了，NN的结构贡献给了NTK。

现在正式阅读这篇==Coord MLP==:

在实验中，MLP总是倾向于学习更光滑的结果，很难复原高频细节。这种现象有理论解释，那就是网络自身的谱偏置（spectral bias）。

标准MLP的学习过程可以用一个带neural tangent kernel（NTK）的kernel regression[1]来刻画。而NTK理论揭示了，MLP它所对应的kernel有一个性质，那就是随着频率增加，它会快速下降[2]。这一性质阻止了MLP去学习到一个高频信号。

> [1]对于MLP的NTK，<font color="brown">它的特征值会随着频率的变大而迅速衰减，这意味着高频信号的特征值小</font>，故而在每次参数更新中受到的惩罚小，收敛很慢很慢，甚至几乎学不出来。
>
> [2] 我其实并不知道这直观地意味着什么，但我猜想和上面NTK的性质有关系？NTK会趋向平滑，一个smooth的NTK或许很难最终塑造出抖动很厉害、很sharp的函数。

这篇文章则主要在讲如何<font color="brown">克服谱偏置</font>，提出了一种将网络input从低维map到高维的一种方式，让MLP在不增加容量的情况下也能够学好高频信号。

设计原则：

- 让kernel的带宽可调（减缓spectral falloff）
- 让NTK在定义域内shift-invariant（也就是只和点点之间的差有关，而不关心点的绝对坐标）

傅里叶变换方法可以同时满足这两个目标。

首先，它的kernel function可以变换成只和Δv有关的函数，证明了这种mapping方式是shift-invariant。

因为 $\cos(\alpha-\beta) = \cos \alpha \cos \beta + \sin \alpha \sin \beta$, 核函数可以映射为:

$k_\gamma(v_1, v_2) = \gamma(v_1)^T \gamma(v_2)  = \sum_{j=1}^m a_j^2 \cos\left(2 \pi \mathbf b_j^T\left(v_1 - v_2 \right) \right) = h_\gamma(v_1 - v_2)$

   $ \textrm{where } h_\gamma(v_\Delta) \triangleq \sum_{j=1}^m a_j^2 \cos(2 \pi \mathbf b_j^T v_\Delta) \, .$

其次，通过调整变换中的参数a（ $a_j = 1/j^p$ ）和b（$b_j = j$ ），可以调节kernel的带宽。

###### 4.3 PE of NeRF

<img src="../../../images/typora-images/image-20230724202625603.png" alt="image-20230724202625603" style="zoom:50%;" />

同研究Coord MLP得到的结论类似，没有应用Positional Encoding时辐射场渲染结果看起来更“糊”。

PE把这些坐标投射到高维空间变成高维空间的basis，满足了平移不变性，单调性和对称性。

##### 5.NeRF

###### 5.1 原理

参考[1](https://zhuanlan.zhihu.com/p/622380174)

对于NeRF，直观地看，输入是五维（3维表示位置，2维表示视角or入射光线角度or经纬度）。重建颜色和密度。

> <font color="brown">Original Paper:</font> regressing from a single 5D coordinate (*x, y, z,* *θ*, *φ*) to a single volume density and view-dependent RGB color.
>
> That is
>
> 1.  (a sampled set of 3D points + 2D viewing directions)  → (an output set of colors and densities)
>
>    这里的color具体可以称作emitted color，即某个视角反射光线发射出的颜色。
>
> 2. use classical volume rendering techniques to accumulate those colors and densities into a 2D image, which is naturally differentiable.

<img src="../../../images/typora-images/image-20230725151450468.png" alt="image-20230725151450468" style="zoom:50%;" />

但我们不希望密度受视角的影响，一般的材料没有这种性质；（也不希望视角太多地影响颜色）：

<img src="../../../images/typora-images/image-20230725152329052.png" alt="image-20230725152329052" style="zoom:50%;" />

所以$(\theta, \phi)$只通过一层网络影响$(r,g,b)$

> <font color="brown">Original Paper:</font>  *σ* as a function of only the location **x**, while allowing the RGB color **c** to be predicted as a function of both location and viewing direction

img by EG3D:

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230908111239763.png" alt="image-20230908111239763" style="zoom:36%;" />

如果输入没有$(\theta, \phi)$ ，模型会难以表示镜面反射，没法表示非朗伯效应。

对视角相关的发射光辉（emitted radiance）的可视化：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20230907154813096.png" alt="image-20230907154813096" style="zoom:50%;" />

可以看到上图展示了来自两个不同相机位置的两个固定 3D 点的外观：一个位于船侧（橙色小图），一个位于水面（蓝色小图）。 NeRF能够预测这两个 3D 点的镜面反射外观的变化，并且 (c) 中展示了这种性质（<font color="brown">非朗伯</font>）在整个视角半球上都是连续、具有泛化性的。

采样也是采用了由粗到精的方式降低成本：

<img src="../../../images/typora-images/image-20230725152855615.png" alt="image-20230725152855615" style="zoom:50%;" />

###### 5.2 代码

[repo URL](https://colab.research.google.com/drive/1oRnnlF-2YqCDIzoc-uShQm8_yymLKiqr)

> 原理上，NeRF的核心idea就是x,d->sigma,color，为了解决loss弄出了基于体素渲染的可微分渲染，为了方便输入特征可学习加入了positional encoding。实现上，代码注释后很容易抓住重点。代码引入相机位姿、ray、chunk等概念。

5.2.1 加载原始数据

```python
# 加载ShapeNet数据，repo已经提供
data_f = "66bdbc812bd0a196e194052f3f12cb2e.npz"
data = np.load(data_f)

# 图像归一化到[0,1]
images = data["images"] / 255
```

5.2.2 准备生成光线的基准位置和朝向

```python
img_size = images.shape[1]
# 注意: 图像上每个像素都对应一条光线
# 以图像中心作为坐标原点，这里是要为后续的光线的基准位置和朝向作准备
xs = torch.arange(img_size) - (img_size / 2 - 0.5)
ys = torch.arange(img_size) - (img_size / 2 - 0.5)

# 这里需要注意，coords错了会训练不出来，主要是meshgrid的indexing方式
parms = inspect.signature(torch.meshgrid).parameters.keys()
# torch.meshgrid的indexing参数官方一直变来变去，需要检查一下。目前版本的参数列表是['tensors', 'indexing']
# 这里的-y是因为世界坐标系的y轴向上为正
if 'indexing' in parms:
    (xs, ys) = torch.meshgrid(xs, -ys, indexing='xy')
else:
    (ys, xs) = torch.meshgrid(-ys, xs)

# 这里拿到相机的焦点位置
focal = float(data["focal"])
# 这里就是基准的相机平面，在世界坐标系中，注意z是-focal
# （这里相机的基准为平面，后续进行旋转、缩放，就能还原真实的相机位姿）
pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
camera_coords = pixel_coords / focal # 相机平面缩放到z=-1位置
# 光线的基准朝向 shape: [100, 100, 3]
init_ds = camera_coords.to(device)
# 光线的基准位置 shape: [3,]
init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)
```

> 生成光线：需要注意的是，训练集中每张图片都有其对应的相机姿态信息标注

```python
# 从训练集中随机挑一组相机位姿
poses = data["poses"]
target_img_idx = np.random.randint(images.shape[0])
# [800, 4, 4]
target_pose = poses[target_img_idx].to(device)

# 这里R就是熟悉的变换矩阵，看到3*3就知道里面只有旋转和缩放
'''
如何理解矩阵相乘的几何意义或现实意义？ - DBinary的回答 - 知乎
https://www.zhihu.com/question/28623194/answer/1486711889
''' 
R = target_pose[:3, :3]
R = torch.Tensor(R)

# 翻译一下,对于100*100的平面上每个三维坐标，都用R进行变换
# 最终的ds就是旋转后的光线方向向量
ds = torch.einsum("ij,hwj->hwi", R, init_ds)
# 因为位置是单独一个向量，直接左乘变换矩阵即可
# @ 表示常规的数学上定义的矩阵相乘；* 表示两个矩阵对应位置处的两个元素相乘
os = (R @ init_o).expand(ds.shape)

# 对照原论文中光线公式Section 4: r(t) = o + t * d.
# 此时我们就已经准备好了模型的输入o和d
```

> **两个矩阵相乘的意义是将右边矩阵中的每一列列向量变换到左边矩阵中每一行行向量为基所表示的空间中去**。更抽象的说，一个矩阵可以表示一种线性变换。

<img src="../../../images/typora-images/v2-bddb06555c7eb1d535091fdf047cc2a5_1440w.png" alt="image-20230725152855615" style="zoom:50%;" />

> 和代码对应：
>
> Eye at z = 0 ------ <font color="brown">os</font> = R @ init_o （光源：光线基准位置经过R变换）
>
> near plane ------ <font color="brown">ds</font> = torch.einsum("ij,hwj->hwi", R, init_ds) （视角：光线基准朝向经过R变换）

1. 已知光源os，视角方向ds
2. MLP输出point处颜色和密度
3. 的颜色和密度通过体积渲染得到，而整个3D points集合，就可渲染到2D图像。

> 处理为batch rays: 对照原文的section 5.3

```python
# 设置每张图片采样64*64条光线
batch_img_size = 64
n_batch_pix = batch_img_size**2
​
# 使每个像素对应光线的采样概率一致
pixel_ps = torch.full((n_pix,), 1 / n_pix).to(device)
# 拿到被采样到的光线的index
pix_idxs = pixel_ps.multinomial(n_batch_pix, False)
# 计算每个光线对应的row和col
pix_idx_rows = pix_idxs // img_size
pix_idx_cols = pix_idxs % img_size
​
# 从img_size*img_size的光线中拿出对应光线
ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
    batch_img_size, batch_img_size, -1
)
os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
    batch_img_size, batch_img_size, -1
)
```



###### 5.3 整体代码

[paper]()

[Supplementary Materials](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392-supp.pdf)

[pytorch-nerf codes](https://colab.research.google.com/drive/1oRnnlF-2YqCDIzoc-uShQm8_yymLKiqr#scrollTo=VWextW26Fvem)

0. 数据准备

```shell
!wget "https://github.com/airalcorn2/pytorch-nerf/blob/master/66bdbc812bd0a196e194052f3f12cb2e.npz?raw=True" -O 66bdbc812bd0a196e194052f3f12cb2e.npz
```

1. 函数定义

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim

'''第一遍均匀采样'''
def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os):
    # Sample depths (t_is_c). See Equation (2) in Section 4. 
    u_is_c = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds)
    t_is_c = t_i_c_bin_edges + u_is_c * t_i_c_gap
    # Calculate the points along the rays (r_ts_c) using the ray origins (os), sampled
    # depths (t_is_c), and ray directions (ds). See Section 4: r(t) = o + t * d.
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
    return (r_ts_c, t_is_c)

'''第二遍精细采样'''
def get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds):
    # See text surrounding Equation (5) in Section 5.2 and:
    # https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html#discrete_distributions.

    # Define PDFs (pdfs) and CDFs (cdfs) from weights (w_is_c).
    w_is_c = w_is_c + 1e-5
    pdfs = w_is_c / torch.sum(w_is_c, dim=-1, keepdim=True)
    cdfs = torch.cumsum(pdfs, dim=-1)
    cdfs = torch.cat([torch.zeros_like(cdfs[..., :1]), cdfs[..., :-1]], dim=-1)

    # Get uniform samples (us).
    us = torch.rand(list(cdfs.shape[:-1]) + [N_f]).to(w_is_c)

    # Use inverse inverse transform sampling to sample the depths (t_is_f).
    idxs = torch.searchsorted(cdfs, us, right=True)
    t_i_f_bottom_edges = torch.gather(t_is_c, 2, idxs - 1)
    idxs_capped = idxs.clone()
    max_ind = cdfs.shape[-1]
    idxs_capped[idxs_capped == max_ind] = max_ind - 1
    t_i_f_top_edges = torch.gather(t_is_c, 2, idxs_capped)
    t_i_f_top_edges[idxs == max_ind] = t_f
    t_i_f_gaps = t_i_f_top_edges - t_i_f_bottom_edges
    u_is_f = torch.rand_like(t_i_f_gaps).to(os)
    t_is_f = t_i_f_bottom_edges + u_is_f * t_i_f_gaps

    # Combine the coarse (t_is_c) and fine (t_is_f) depths and sort them.
    (t_is_f, _) = torch.sort(torch.cat([t_is_c, t_is_f.detach()], dim=-1), dim=-1)
    # Calculate the points along the rays (r_ts_f) using the ray origins (os), depths
    # (t_is_f), and ray directions (ds). See Section 4: r(t) = o + t * d.
    r_ts_f = os[..., None, :] + t_is_f[..., :, None] * ds[..., None, :]
    return (r_ts_f, t_is_f)

'''体积渲染'''
def render_radiance_volume(r_ts, ds, chunk_size, F, t_is):
    # Use the network (F) to predict colors (c_is) and volume densities (sigma_is) for
    # 3D points along rays (r_ts) given the viewing directions (ds) of the rays. See
    # Section 3 and Figure 7 in the Supplementary Materials.
    r_ts_flat = r_ts.reshape((-1, 3))
    ds_rep = ds.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
    ds_flat = ds_rep.reshape((-1, 3))
    c_is = []
    sigma_is = []
    # The network processes batches of inputs to avoid running out of memory.
    for chunk_start in range(0, r_ts_flat.shape[0], chunk_size):
        r_ts_batch = r_ts_flat[chunk_start : chunk_start + chunk_size]
        ds_batch = ds_flat[chunk_start : chunk_start + chunk_size]
        preds = F(r_ts_batch, ds_batch)
        c_is.append(preds["c_is"])
        sigma_is.append(preds["sigma_is"])

    c_is = torch.cat(c_is).reshape(r_ts.shape)
    sigma_is = torch.cat(sigma_is).reshape(r_ts.shape[:-1])

    # Calculate the distances (delta_is) between points along the rays. The differences
    # in depths are scaled by the norms of the ray directions to get the final
    # distances. See text following Equation (3) in Section 4.
    delta_is = t_is[..., 1:] - t_is[..., :-1]
    # "Infinity". Guarantees last alpha is always one.
    one_e_10 = torch.Tensor([1e10]).expand(delta_is[..., :1].shape)
    delta_is = torch.cat([delta_is, one_e_10.to(delta_is)], dim=-1)
    delta_is = delta_is * ds.norm(dim=-1).unsqueeze(-1)

    # Calculate the alphas (alpha_is) of the 3D points using the volume densities
    # (sigma_is) and distances between points (delta_is). See text following Equation
    # (3) in Section 4 and https://en.wikipedia.org/wiki/Alpha_compositing.
    alpha_is = 1.0 - torch.exp(-sigma_is * delta_is)

    # Calculate the accumulated transmittances (T_is) along the rays from the alphas
    # (alpha_is). See Equation (3) in Section 4. T_i is "the probability that the ray
    # travels from t_n to t_i without hitting any other particle".
    T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
    # Guarantees the ray makes it at least to the first step. See:
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L142,
    # which uses tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True).
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0

    # Calculate the weights (w_is) for the colors (c_is) along the rays using the
    # transmittances (T_is) and alphas (alpha_is). See Equation (5) in Section 5.2:
    # w_i = T_i * (1 - exp(-sigma_i * delta_i)).
    w_is = T_is * alpha_is

    # Calculate the pixel colors (C_rs) for the rays as weighted (w_is) sums of colors
    # (c_is). See Equation (5) in Section 5.2: C_c_hat(r) = Σ w_i * c_i.
    C_rs = (w_is[..., None] * c_is).sum(dim=-2)

    return (C_rs, w_is)


def run_one_iter_of_nerf(
    ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, chunk_size, F_c, N_f, t_f, F_f
):
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    (C_rs_c, w_is_c) = render_radiance_volume(r_ts_c, ds, chunk_size, F_c, t_is_c)

    (r_ts_f, t_is_f) = get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds)
    (C_rs_f, _) = render_radiance_volume(r_ts_f, ds, chunk_size, F_f, t_is_f)

    return (C_rs_c, C_rs_f)


class NeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of encoding functions for positions. See Section 5.1.
        self.L_pos = 10
        # Number of encoding functions for viewing directions. See Section 5.1.
        self.L_dir = 4
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        in_feats = pos_enc_feats
        net_width = 256
        early_mlp_layers = 5
        early_mlp = []
        for layer_idx in range(early_mlp_layers):
            early_mlp.append(nn.Linear(in_feats, net_width))
            early_mlp.append(nn.ReLU())
            in_feats = net_width

        self.early_mlp = nn.Sequential(*early_mlp)

        in_feats = pos_enc_feats + net_width
        late_mlp_layers = 3
        late_mlp = []
        for layer_idx in range(late_mlp_layers):
            late_mlp.append(nn.Linear(in_feats, net_width))
            late_mlp.append(nn.ReLU())
            in_feats = net_width

        self.late_mlp = nn.Sequential(*late_mlp)
        self.sigma_layer = nn.Linear(net_width, net_width + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_enc_feats + net_width, net_width // 2), nn.ReLU()
        )
        self.final_layer = nn.Sequential(nn.Linear(net_width // 2, 3), nn.Sigmoid())

    def forward(self, xs, ds):
        # Encode the inputs. See Equation (4) in Section 5.1.
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2 ** l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2 ** l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2 ** l_dir * torch.pi * ds))
            ds_encoded.append(torch.cos(2 ** l_dir * torch.pi * ds))

        ds_encoded = torch.cat(ds_encoded, dim=-1)
        
        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays. See
        # Section 3 and Figure 7 in the Supplementary Materials.
        outputs = self.early_mlp(xs_encoded)
        outputs = self.late_mlp(torch.cat([xs_encoded, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])
        outputs = self.pre_final_layer(torch.cat([ds_encoded, outputs[:, 1:]], dim=-1))
        c_is = self.final_layer(outputs)
        return {"c_is": c_is, "sigma_is": sigma_is}
```

2. 训练	

```python
# Set seed.
seed = 9458
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize coarse and fine MLPs.
device = "cuda:0"
F_c = NeRFMLP().to(device)
F_f = NeRFMLP().to(device)
# Number of query points passed through the MLP at a time. See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L488.
chunk_size = 1024 * 32
# Number of training rays per iteration. See Section 5.3.
batch_img_size = 64
n_batch_pix = batch_img_size ** 2

# Initialize optimizer. See Section 5.3.
lr = 5e-4
optimizer = optim.Adam(list(F_c.parameters()) + list(F_f.parameters()), lr=lr)
criterion = nn.MSELoss()
# The learning rate decays exponentially. See Section 5.3
# See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L486.
lrate_decay = 250
decay_steps = lrate_decay * 1000
# See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L707.
decay_rate = 0.1

# Load dataset.
data_f = "66bdbc812bd0a196e194052f3f12cb2e.npz"
data = np.load(data_f)

# Set up initial ray origin (init_o) and ray directions (init_ds). These are the
# same across samples, we just rotate them based on the orientation of the camera.
# See Section 4.
images = data["images"] / 255
img_size = images.shape[1]
xs = torch.arange(img_size) - (img_size / 2 - 0.5)
ys = torch.arange(img_size) - (img_size / 2 - 0.5)
(xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
focal = float(data["focal"])
pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
# We want the zs to be negative ones, so we divide everything by the focal length
# (which is in pixel units).
camera_coords = pixel_coords / focal
init_ds = camera_coords.to(device)
init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

# Set up test view.
test_idx = 150
plt.imshow(images[test_idx])
plt.show()
test_img = torch.Tensor(images[test_idx]).to(device)
poses = data["poses"]
test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
test_os = (test_R @ init_o).expand(test_ds.shape)

# Initialize volume rendering hyperparameters.
# Near bound. See Section 4.
t_n = 1.0
# Far bound. See Section 4.
t_f = 4.0
# Number of coarse samples along a ray. See Section 5.3.
N_c = 64
# Number of fine samples along a ray. See Section 5.3.
N_f = 128
# Bins used to sample depths along a ray. See Equation (2) in Section 4.
t_i_c_gap = (t_f - t_n) / N_c
t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

# Start training model.
train_idxs = np.arange(len(images)) != test_idx
images = torch.Tensor(images[train_idxs])
poses = torch.Tensor(poses[train_idxs])
n_pix = img_size ** 2
pixel_ps = torch.full((n_pix,), 1 / n_pix).to(device)
psnrs = []
iternums = []
# See Section 5.3.
num_iters = 300000
display_every = 100
F_c.train()
F_f.train()
```

```shell
!nvidia-smi
```

```python
for i in range(num_iters):
    # Sample image and associated pose.
    target_img_idx = np.random.randint(images.shape[0])
    target_pose = poses[target_img_idx].to(device)
    R = target_pose[:3, :3]

    # Get rotated ray origins (os) and ray directions (ds). See Section 4.
    ds = torch.einsum("ij,hwj->hwi", R, init_ds)
    os = (R @ init_o).expand(ds.shape)

    # Sample a batch of rays.
    pix_idxs = pixel_ps.multinomial(n_batch_pix, False)
    pix_idx_rows = pix_idxs // img_size
    pix_idx_cols = pix_idxs % img_size
    ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
        batch_img_size, batch_img_size, -1
    )
    os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
        batch_img_size, batch_img_size, -1
    )

    # Run NeRF.
    (C_rs_c, C_rs_f) = run_one_iter_of_nerf(
        ds_batch,
        N_c,
        t_i_c_bin_edges,
        t_i_c_gap,
        os_batch,
        chunk_size,
        F_c,
        N_f,
        t_f,
        F_f,
    )
    target_img = images[target_img_idx].to(device)
    target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_f.shape)
    # Calculate the mean squared error for both the coarse and fine MLP models and
    # update the weights. See Equation (6) in Section 5.3.
    loss = criterion(C_rs_c, target_img_batch) + criterion(C_rs_f, target_img_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Exponentially decay learning rate. See Section 5.3 and:
    # https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/.
    for g in optimizer.param_groups:
        g["lr"] = lr * decay_rate ** (i / decay_steps)

    if i % display_every == 0:
        F_c.eval()
        F_f.eval()
        with torch.no_grad():
            (_, C_rs_f) = run_one_iter_of_nerf(
                test_ds,
                N_c,
                t_i_c_bin_edges,
                t_i_c_gap,
                test_os,
                chunk_size,
                F_c,
                N_f,
                t_f,
                F_f,
            )

        loss = criterion(C_rs_f, test_img)
        print(f"Loss: {loss.item()}")
        psnr = -10.0 * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(C_rs_f.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        plt.show()

        F_c.train()
        F_f.train()

print("Done!")
```



##### 6.Plenoxels

无需网络，既得益于方法创新，也得益于工程上的CUDA优化。

体素+球面谐波系数（只用到金字塔的前三层）

<img src="../../../images/typora-images/image-20230725154550204.png" alt="image-20230725154550204" style="zoom:50%;" />

体素中每一项不是常规的存rgb+$\delta$，而是存rgb上的球面谐波系数（以及密度），可视化就是下面这27个球：

<img src="../../../images/typora-images/image-20230725154517994.png" alt="image-20230725154517994" style="zoom:50%;" />

这个和NeRF的区别就类似于SRCNN和SRLUT的区别。

中间非整数坐标的体素也是通过三线性插值（Trilinear Interpolation）得到。

> 镜面反射、金属的高光学不了，但一般有材质感的反射效果都还是可以学出来的。

存储时，体素剪枝：密度为0的点（如空气），和光线射到该点时能量很弱的点（如物体内部），可以被剪枝掉。

[笔记1](https://zhuanlan.zhihu.com/p/549260115), [教程1](https://zhuanlan.zhihu.com/p/481275794), [教程2](https://zhuanlan.zhihu.com/p/482154458), [教程3](https://blog.csdn.net/minstyrain/article/details/123858806), [教程4](https://blog.csdn.net/qq_44324007/article/details/129998545?spm=1001.2014.3001.5502)

[代码解读1](https://liwen.site/archives/2302), [代码解读2](https://zhuanlan.zhihu.com/p/524523357)

#### 四.改进的工作