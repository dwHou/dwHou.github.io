StatQuest with Josh Starmer

李航《统计学习方法》

### PCA

理解主成分分析法（PCA）和构成分析法的奇异值分解（SVD）

http://blog.codinglabs.org/articles/pca-tutorial.html

#### 二维：

**PC1是一条通过原点（由均值x_mean, y_mean得到，即数据中心化），并且综合所有样本点，有着最大的投影点到原点距离平方和的线。（除以样本数目后，也可以说方差最大）**

<font color="brown">术语总结：</font>

> 主成分：PC1、PC2这些就属于
>
> 线性组合：就是PC1、PC2的配方
>
> **奇异向量（特征向量）：PC1、PC2上的单位向量**
>
> 载荷得分：单位向量在各个变量的投影（<1），和配方意义类似

PCA calls the SS(distances) for the best fit line the Eigenvalue for PC1

PCA把最佳拟合线距离的平方和 称为 PC1的特征值

PC1的特征值的平方根 称为 PC1的奇异值

<font color="brown">数学定义：</font>

> Ax=cx，其中A是矩阵，c是特征值，x是特征向量
>
> Ax矩阵相乘的含义就是，矩阵A对向量x进行一系列的变换(旋转或者拉伸)，其效果等于一个常数c乘以向量x。
>
> 通常我们求特征值和特征向量是想知道，矩阵能使哪些向量(当然是特征向量)只发生拉伸，其拉伸程度如何(特征值的大小)。这个真正的意义在于，是为了让我们看清矩阵能在哪个方向(特征向量)产生最大的变化效果。

**PC2就是垂直于PC1的一条通过原点的线。**

再把整个坐标空间旋转，使得PC1和PC2分别为横轴纵轴。

SS(distances for PC1) / (n - 1) = Variation for PC1 

（这就是PC1方差最大的那个方差。）

SS(distances for PC2) / (n - 1) = Variation for PC2

Variation for PC1 / (Variation for PC1 + Variation for PC2)  是PC1的差异率。

碎石图（Scree Plot）是一种图像呈现方式，用来描绘每个PC所占的差异率。

#### 三维：

和二维类似，找到贯穿原点的最佳拟合线是PC1。只不过现在PC1的配方包含三种成分了。

然后找到穿过原点且和PC1垂直的条件下的最佳拟合线是PC2。

最后，和PC1、PC2所在平面垂直的为PC3。

#### 以此类推：

只不过超过3维，我们没法用几何理解了。

理论上，每个变量有一个PC。但实际上PC的数量是min{变量的个数，样本数}。

一旦确定了所有主成分，就可以用特征值，即距离的平方和确定每个PC占的差异率。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20221008212023745.png" alt="image-20221008212023745" style="zoom:30%;" />

比如这里PC1是79%，PC2是15%，PC3是6%。则将3D图转换为二维的PCA图，仍可以表示94%的差异率。

依次类推，虽然更高维度我们画不了图，但不影响我们做数学和画碎石图。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20221008212722867.png" alt="image-20221008212722867" style="zoom:30%;" /><img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20221008213844987.png" alt="image-20221008213844987" style="zoom:30%;" />

如果碎石图看起来像上图，其中PC3和PC4占差异相当大的比例。

那么仅使用前两个PC不能很精确地代表我们的数据。但是，即使像这样不清晰的PCA图（noisy PCA plot）也可以用来数据聚类。

