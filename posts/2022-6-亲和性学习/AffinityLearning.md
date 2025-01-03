# 深度图增强/补全

[TOC]

[RGB+ToF Depth Completion](https://codalab.lisn.upsaclay.fr/competitions/4956) 

[KITTI Depth Completion(DC) Dataset BenchMark ](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

## 应用场景

<font color="orange">**室内：**</font>Time-of-Flight (ToF)，智能手机的深度传感器，精确度低；距离有限

​            structured light，如微软的Kinect（只需做深度图增强，也叫hole-filling），对光线敏感；耗电

​            stereo cameras，需要大量的计算；无特征区域失败

<font color="orange">**室外：**</font>LiDARs，昂贵；庞大

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220606162928564.png" alt="image-20220606162928564" style="zoom:50%;" />

## 无引导的方法

## RGB 引导的方法

### 1.融合置前的模型

### 2.融合置后的模型

### 3.显式3D建模的模型

### 4.基于空间传播（SPN）的模型

#### 4.1 Anisotropic Diffusion



#### 4.2 SPN

**Learning Affinity via. Spatial Propagation Network** 

##### 概念

**亲和性**描述了像素或区域之间的强成对关系。与距离、颜色、纹理或结构相关。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531104029824.png" alt="image-20220531104029824" style="zoom:50%;" />

##### 核心思想

在图像密集预测类型的任务例如深度估计、语义分割中，如果利用每个点与相邻点的预测之间的关系进行优化，往往可以得到更为准确而精细的结果。

以往工作，DenseCRF就是如此。

相同思路的方法还有SPN，通过四个方向的空间传播收集周围点的相似（Affinity）信息。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531110131267.png" alt="image-20220531110131267" style="zoom:50%;" />



#### 4.3 CSPN

##### 原理

主要贡献：

1. 对SPN进行改进提出CSPN，使用卷积提取像素点周围8个邻接像素点的相似信息，更快速更准确的对该点的深度预测值进行优化。

2. CSPN由于其可以进行相似信息预测，可以进行深度补全任务，在将稀疏深度图转为密集深度图可以保持邻域之间的圆滑。

CSPN 即基于 SPN 做出的改进，用卷积核的方式直接得到8个方向上的信息，另外通过循环迭代，得到更大相邻半径的信息：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531110347154.png" alt="image-20220531110347154" style="zoom:50%;" />

对于一个深度估计的网络而言，输出大小为 MxNx1，对应图片每个像素的深度。CSPN 则在其基础上多输出一个 MxNx8 的分支，其中 8 表示对于每个像素的8个邻接点与其的相似性（Affinity）信息。

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531110545312.png" alt="image-20220531110545312" style="zoom:50%;" />

得到额外的MxNx8的相似性信息后，首先对其进行正则化，目的是将其规范到（-1,1）范围内，并在正则化过程中生成3x3卷积核，对原始MN1的深度估计结果进行卷积运算，输出最终的优化后的深度估计，针对不同数据集或场景的需要，可以进行多次迭代，扩大邻接像素点的范围，进行更大半径的邻接信息优化。正则化方法如下：

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531111024815.png" alt="image-20220531111024815" style="zoom:50%;" />

> **<font color="lighblue">原文：</font>**其中变换核 $\hat{k}_{i,j} \in R^{k\times k\times c} $ 是亲和网络的输出，它取决于输入图像。 核尺寸 k 通常设置为奇数（一般是3），以便围绕像素$(i, j)$ 的计算上下文是对称的。 $⊙$ 是Hadamard乘积（逐项乘积）。 和SPN一样，CSPN将核的权重正则化到 $(-1, 1)$ 范围，这样有助于模型稳定收敛。最后，CSPN执行此迭代 $N$ 步以达到稳定的分布。



##### 源码简析

参考CSPN的[源码](https://github.com/XinJCheng/CSPN/blob/master/cspn_pytorch/models/torch_resnet_cspn_nyu.py)，从网络输出角度可以将算法结构简单的看做UNet的改进，对于原始输出为MxNx1的UNet，额外新增一个MxNx8的分支，用来输出邻接像素点的相似性信息。

**关键代码**

0.接口代码

```python
class Affinity_Propagate(nn.Module):
  def __init__(self,
                   prop_time,
                   prop_kernel,
                   norm_type='8sum'):
          """
          Inputs:
              prop_time: how many steps for CSPN to perform
              prop_kernel: the size of kernel (currently only support 3x3)
              way to normalize affinity
                  '8sum': normalize using 8 surrounding neighborhood
                  '8sum_abs': normalization enforcing affinity to be positive
                              This will lead the center affinity to be 0
          """
      self.in_feature = 1
      self.out_feature = 1
  def forward(self, guidance, blur_depth, sparse_depth=None)
```

可以看到Affinity_Propagate的特征为1通道。

由关联矩阵形成的sum_conv的参数会设置requires_grad = False。



1.正则化代码

```python
gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                     gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum
```

对8通道的邻接像素的相似性信息进行正则化与拼接求和，输入后续卷积操作。

2.深度信息优化代码

```python
if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth
```

3.损失函数代码

```python
class Wighted_L1_Loss(torch.nn.Module):
    def __init__(self):
        super(Wighted_L1_Loss, self).__init__()

    def forward(self, pred, label):
        label_mask = label > 0.0001
        _pred = pred[label_mask]
        _label = label[label_mask]
        n_valid_element = _label.size(0)
        diff_mat = torch.abs(_pred-_label)
        loss = torch.sum(diff_mat)/n_valid_element
        return loss
```

从损失函数可以看出，CSPN的损失函数仅有深度信息平均误差，对于邻接像素点的相似性信息，并无标签进行学习，缺失也难以获得相似性信息的标签，不过可以参考SPN中数学推导的方法得到相似性信息，或许可以当做标签，但可能并不准确。

##### 性能效果

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20220531132527429.png" alt="image-20220531132527429" style="zoom:50%;" />





1. Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image
2. Learning Affinity via Spatial Propagation Networks
3. Learning Depth with Convolutional Spatial Propagation Network
4. Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network

#### 4.4 CSPN++

##### 原理

为了解决确定传播卷积核大小和迭代次数的困难，Cheng 等人进一步提出 CSPN++。

主要贡献：

#### 4.5 DynSPN

##### **原理**

**现状分析：**当前的大多数 SPNs都基于线性传播模型。 也就是affinity matrix的值在传播过程中不会改变，这样会限制表达能力。

**核心思想：**为了解决上述问题，受动态滤波器 (Jia et al. 2016) 的启发，本文提出了一种新的非线性传播，称为DySPN。它可以学习自适应affinity matrix。 为了实现这种设计，采用了注意力机制。 更具体地说，在传播开始时，远邻居提供的信息来快速填充初始深度图的洞，在深度图逐渐稠密时，更关注近邻居提供的信息来保边。

> Inspired by the dynamic filters described above, we propose a DySPN with a non-linear propagation model, which applies <font color="red">spatial and sequential attention</font> to generate a series of <font color="red">adaptive affinity matrices</font>.







#### 4.6 Guided Filter

和FastGuidedFilter进行比较。

**原理**



```python
from guided_filter_pytorch.guided_filter import FastGuidedFilter

hr_y = FastGuidedFilter(r, eps)(lr_x, lr_y, hr_x)

# 分析一下前向过程：
def forward(self, lr_x, lr_y, hr_x):
  1.取各输入(引导lr，引导hr，目标lr)的N,C,H,W
  2.首先判断运行条件，
  	N相同，
    高低分辨率引导的C相同，
    引导lr的通道为1或和目标lr相同
    引导lr和目标lr的长宽一致
    引导lr的长宽大于直径(2r+1)
    assert n_lrx == n_lry and n_lry == n_hrx
    assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
    assert h_lrx == h_lry and w_lrx == w_lry
    assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1
  3.
        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b
```





### 5.基于残差深度的模型





多尺度Guide网络

BTS（From Big to Small）

