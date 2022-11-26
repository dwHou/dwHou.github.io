[CompressAI](https://interdigitalinc.github.io/CompressAI/) (compress-ay) 是用于端到端压缩研究的 PyTorch 库和评估平台。

## Introduction

**概念**

CompressAI 建立在 PyTorch 之上并提供：

- 用于基于深度学习的数据压缩的自定义操作、层和模型
- 官方 [TensorFlow 压缩库](https://github.com/tensorflow/compression)的部分端口
- 用于学习图像压缩的预训练端到端压缩模型
- 将学习模型与经典图像/视频压缩编解码器进行比较的评估脚本

CompressAI 旨在通过提供资源来研究、实施和评估基于机器学习的压缩编解码器，让更多的研究人员为学习图像和视频压缩领域做出贡献。

**模型Zoo**
CompressAI 包括一些用于压缩任务的预训练模型。 有关更多文档，请参阅 Model Zoo 部分。

在不同比特率失真点和不同指标下训练的可用模型列表预计在未来会增加。

## Installation

```shell
pip install compressai
```

## TUTORIALS

### 训练

运行 train.py –help 以列出可用选项。

```python
  -h, --help 显示帮助信息
  -m, --model 模型结构{bmshj2018-factorized,bmshj2018-hyperprior,mbt2018-mean,mbt2018,cheng2020-anchor,cheng2020-attn}                   
  -d, --dataset 训练集
  -e, --epochs 训练总轮次
  -lr, --learning-rate 学习率
  -n, --num-workers 数据加载线程数
  --lambda Bit-rate distortion参数
  --batch-size 训练批大小
  --test-batch-size 测试批大小
  --aux-learning-rate 辅助损失学习率
  --patch-size 训练裁剪的patch大小                       
  --cuda 使用cuda
  --save 模型保持到硬盘
  --seed SEED 随机种子
  --clip_max_norm 梯度裁剪最大范数
  --checkpoint 保存点的路径
                     
```

示例：

```shell
python3 examples/train.py -m mbt2018-mean -d /path/to/image/dataset --batch-size 16 -lr 1e-4 --save --cuda
```

#### 模型更新

训练模型后，您需要运行 update_model 脚本来更新熵瓶颈（bottlenecks）的内部参数：

```shell
python -m compressai.utils.update_model --architecture ARCH checkpoint_best_loss.pth.tar
```

注：在 examples/ 路径下执行该命令行

这将修改与执行实际熵编码所需的学习累积分布函数 (CDF) 相关的缓冲区。

> 运行 `python -m compressai.utils.update_model --help `获取完整的选项列表。

或者，您可以在训练脚本结束时调用 CompressionModel 或 EntropyBottleneck 实例的 update() 方法，然后再保存模型检查点。

#### 模型评估

更新模型检查点后，您可以使用 eval_model 获取其在图像数据集上的性能：

```shell
python -m compressai.utils.eval_model checkpoint /path/to/image/dataset -a ARCH -p path/to/checkpoint-xxxxxxxx.pth.tar
```

> 运行 `python -m compressai.utils.eval_model --help` 获取完整的选项列表。

#### 熵编码

默认情况下，CompressAI 使用范围不对称数字系统 (ANS, Asymmetric Numeral Systems ) 熵编码器。 您可以使用 `compressai.available_entropy_coders()` 获取已实现的熵编码器的列表，并通过 `compressai.set_entropy_coder()` 更改默认的熵编码器。

1.将图像张量压缩为比特流：

```python
x = torch.rand(1, 3, 64, 64)
y = net.encode(x)
strings = net.entropy_bottleneck.compress(y)
```

2.将比特流解压为图像张量：

```python
shape = y.size()[2:]
y_hat = net.entropy_bottleneck.decompress(strings, shape)
x_hat = net.decode(y_hat)
```

### 自定义模型

训练自己的模型
在本教程中，我们将使用 CompressAI 中预定义的一些模块和层来实现自定义的auto encoder结构。

#### 搭建模型

让我们使用 `EntropyBottleneck` 模块构建一个简单的auto encoder，包含编码器的 3 个卷积、解码器的 3 个转置反卷积和 GDN 激活函数：

```python
import torch.nn as nn
import torch

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

class Network(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
            GDN(N), # 广义分歧归一化
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2),
        )

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods

if __name__ == '__main__':
 		net = Network()
    net = net.cuda()
    # 查看entropy_bottleneck层有哪些参数
    for n, p in net.named_parameters():
        print(n)
    
    x = torch.randn((8, 3, 360, 640)).cuda()
    x_hat, y_likelihoods = net(x)
    print(x_hat.shape)
    print(y_likelihoods.shape)
```

步长卷积减少张量的空间维度，同时增加通道数量（这有助于学习更好的潜在（latent）表示）。 瓶颈模块([论文](https://arxiv.org/pdf/1802.01436.pdf))用于在训练时获得latent的可微熵估计。

> 也可以用`CompressionModel `基类来实现网络

#### 损失函数

<font color="brown">**1.率失真损失**</font>

我们将定义一个简单的率失真损失，它最大化 PSNR 重建 (RGB) 并最小化量化latent (y_hat) 的长度（以bit为单位）。

标量$\lambda$用于平衡重建质量和比特率（如 JPEG 质量参数，或 HEVC 的 QP）：

$$ L = Q + \lambda * R $$

```python
import math
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(1, 3, 64, 64)
net = Network()
x_hat, y_likelihoods = net(x)

# bitrate of the quantized latent
N, _, H, W = x.size()
num_pixels = N * H * W
bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels) 
# 算术编码器是一种接近最优的熵编码器，这使得在训练期间使用y的熵作为码率估计成为可能。观测的y_likelihoods越大（不确定性越小，相同维度输入经过熵编码后码字越少），算术编码用的码字越少。

# mean square error
mse_loss = F.mse_loss(x, x_hat)

# final loss term
loss = mse_loss + lmbda * bpp_loss
```

> 可以训练可以处理多个比特率失真点的架构，但这超出了本教程的范围。 请参阅这篇[论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf)。

<font color="brown">**2.辅助损失**</font>

需要训练熵瓶颈的参数以最小化latent元素的密度模型评估。 辅助损失可通过 `entropy_bottleneck` 层访问：

```python
aux_loss = net.entropy_bottleneck.loss()
```

最小化辅助损失是必要的，可以网络训练期间或之后进行。

#### 优化器

为了同时训练压缩网络和熵瓶颈密度估计，我们需要两个优化器。 

```python
import torch.optim as optim

parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
```

> 也可以通过 `torch.optim.Optimizer` [parameter groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options)来定义一个优化器。

#### 训练循环

如下：

```python
x = torch.rand(1, 3, 64, 64)
for i in range(10):
  optimizer.zero_grad()
  aux_optimizer.zero_grad()

  x_hat, y_likelihoods = net(x)
  # ...
  # compute loss as before
  # ...
  loss.backward()
  optimizer.step()

  aux_loss = net.aux_loss()
  aux_loss.backward()
  aux_optimizer.step()
```

## LIBRARY API

#### compressai

compressai.**available_entropy_coders**() ：返回可用的熵编码器列表。

compressai.**get_entropy_coder**() ：返回用于编码比特流的默认熵编码器的名称。

compressai.**set_entropy_coder**(*entropy_coder*) ：指定用于对比特流进行编码的默认熵编码器。

#### compressai.ans

范围不对称数字系统 (rANS) 绑定。 rANS 可用于替代传统范围编码器。

编码器：*class* compressai.ans.**RansEncoder**

解码器：*class* compressai.ans.**RansDecoder**

#### compressai.datasets

*class* compressai.datasets.**ImageFolder**(*root*, *transform=None*, *split='train'*)

加载图像文件夹数据库，训练和测试图像样本分别存放在不同的目录中。

*class* compressai.datasets.**VideoFolder**(*root*, *rnd_interval=False*, *rnd_temp_order=False*, *transform=None*, *split='train'*)

加载视频文件夹数据库。 训练和测试视频剪辑存储在包含许多子目录的目录中。

#### compressai.entropy_models

**熵瓶颈：**

*class* compressai.entropy_models.**EntropyBottleneck**(*channels: int*)

**高斯条件：**

*class* compressai.entropy_models.**GaussianConditional**()

均来自[论文](https://arxiv.org/abs/1802.01436)

#### compressai.layers

MaskedConv2d: 屏蔽未来“看不见的”像素, 用于构建自回归网络组件，参见[论文](https://arxiv.org/abs/1606.05328)。

GDN: Generalized Divisive Normalization layer，参见[论文](https://arxiv.org/abs/1511.06281)。

GDN1: 简化的GDN层，参见[论文](https://arxiv.org/abs/1912.08771)。

AttentionBlock：自注意力模块，参见[论文](https://arxiv.org/abs/2001.01568)。

QReLU: 根据给定位深clamp输入，参见[论文](https://openreview.net/pdf?id=S1zz2i0cY7)。

> GDN, GDN1, QReLU的作者是有重合的。

#### compressai.models

CompressionModel：用于构造具有至少一个熵瓶颈模块的auto-encoder的基类。

```python
class compressai.models.CompressionModel(entropy_bottleneck_channels, init_weights=None)
```

2018：FactorizedPrior，ScaleHyperprior[论文](https://arxiv.org/abs/1802.01436)，MeanScaleHyperprior，JointAutoregressiveHierarchicalPriors

2020：Cheng2020Anchor，Cheng2020Attention[论文](https://arxiv.org/abs/2001.01568)，ScaleSpaceFlow[论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html)

#### compressai.ops

compressai.ops.**ste_round**(*x: torch.Tensor*)：使用非零梯度进行舍入。 通过用恒等函数代替导数来近似梯度，参见[论文](https://arxiv.org/abs/1703.00395)。

*class* compressai.ops.**LowerBound**(*bound: float*)：下限运算符，使用自定义梯度计算 torch.max(x, bound)。当 x 向边界移动时，导数被恒等函数代替，否则梯度保持为零。

*class* compressai.ops.**NonNegativeParametrizer**()：非负重参数化，用于训练期间的稳定性。

#### compressai.transforms

RGB2YCbCr，YCbCr2RGB，YUV420To444，YUV444To420

这里`Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]`

## MODEL ZOO

#### 图像压缩

模型：{bmshj2018-factorized,bmshj2018-hyperprior,mbt2018-mean,mbt2018,cheng2020-anchor,cheng2020-attn}

| Metric  | Loss function   |
| ------- | --------------- |
| MSE     | $L=λ∗255^2∗D+R$ |
| MS-SSIM | $L=λ∗(1-D)+R$   |

| Quality | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| MSE     | 0.0018 | 0.0035 | 0.0067 | 0.0130 | 0.0250 | 0.0483 | 0.0932 | 0.1800 |
| MS-SSIM | 2.40   | 4.58   | 8.73   | 16.64  | 31.73  | 60.50  | 115.37 | 220.00 |

> 我的理解，表里的数字是$λ$。这个应该就是一个核心的概念——熵模型。

对于不同质量等级，网络结构的通道数也会设计得不一样。

{'bmshj2018-factorized': {1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192), 5: (128, 192), 6: (192, 320), 7: (192, 320), 8: (192, 320)}, 'bmshj2018-hyperprior': {1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192), 5: (128, 192), 6: (192, 320), 7: (192, 320), 8: (192, 320)}, 'mbt2018-mean': {1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192), 5: (192, 320), 6: (192, 320), 7: (192, 320), 8: (192, 320)}, 'mbt2018': {1: (192, 192), 2: (192, 192), 3: (192, 192), 4: (192, 192), 5: (192, 320), 6: (192, 320), 7: (192, 320), 8: (192, 320)}, <font color="purple">'cheng2020-anchor': {1: (128,), 2: (128,), 3: (128,), 4: (192,), 5: (192,), 6: (192,)}</font>, 'cheng2020-attn': {1: (128,), 2: (128,), 3: (128,), 4: (192,), 5: (192,), 6: (192,)}}

> The number of channels for the convolutionnal layers and the entropy bottleneck depends on the architecture and the quality parameter (~targeted bit-rate). For low bit-rates (<0.5 bpp), the literature usually recommends 192 channels for the entropy bottleneck, and 320 channels for higher bitrates. The detailed list of configurations can be found in `compressai.zoo.image.cfgs`.

例如cheng2020-anchor：

 compressai.zoo.**cheng2020_anchor**(*quality*, *metric='mse'*, *pretrained=False*, *progress=True*, ***kwargs*)，参数如下：

- **quality** (*int*) – 质量等级 (最低为1, 最高为6)
- **metric** (*str*) – 优化指标，从 (‘mse’, ‘ms-ssim’) 选择
- **pretrained** (*bool*) – 如果为真，则返回预训练模型
- **progress** (*bool*) – 如果为真，则显示下载到 stderr 的进度条

性能：Cheng2020Anchor > JointAutoregressiveHierarchicalPriors > MeanScaleHyperprior > ScaleHyperprior > FactorizedPrior，

具体参见[率失真曲线图](https://interdigitalinc.github.io/CompressAI/zoo.html):

<img src="./kodak-psnr.png" alt="kodak-psnr" style="zoom:60%;" />

注：VTM是VVC的参考软件。

##### Lossy model 

上面提到CompressAI提供了许多智能编码的模型，全都属于有损压缩+熵编码。

##### Lossless model

有损压缩和熵估计模型部分使用神经网络进行非线性变换，这个我们熟悉。所以感觉重点需要研究的恰恰是无损压缩（熵编码）。

比如，有[仓库](https://github.com/candywr/Lossless-image-compress)将CompressAI的该部分摘出来研究。

<font color="brown">数学基础：</font>

设$X\sim N(\mu, \sigma^{2})$，则

$P(x_{1}<X<x_{2}) = \Phi(\frac{x_{2}-\mu}{\sigma})-\Phi(\frac{x_{1}-\mu}{\sigma})$

而$\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{\frac{-t^{2}}{2}}\mathrm{d}t = \frac{1}{2}\mathrm{erfc}(-\frac{x}{\sqrt{2}})$

<font color="brown">代码实现：</font>

我验证了两种实现方式 ① feature_probs_api，② feature_probs_manual

```python3
#!/usr/bin/env python
import torch

def feature_probs_api(feature, sigma, mu):
    # 零均值
    # mu = torch.zeros_like(sigma) 
    sigma = sigma.clamp(1e-10, 1e10)
    # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    # 算术编码就是一个位置，给一个熵估计，只有教学时才以各位置统一均匀的概率为例。然后bin_width/2 = 0.5
    # total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return probs

def feature_probs_manual(feature, sigma, mu):
    sigma = sigma.clamp(1e-10, 1e10)
    
    values = feature - mu
    upper = (values + .5) / sigma 
    lower = (values - .5) / sigma 
    probs = _standardized_cumulative(upper) - _standardized_cumulative(lower)
    return probs

def _standardized_cumulative(inputs):
        half = torch.tensor(.5)
        const = torch.tensor(-(2 ** -0.5))
        return half * torch.erfc(const * inputs)

feature = torch.randn(1,1,2,2)
sigma = torch.randn(1,1,2,2).abs()
mu = feature + 0.01

probs = feature_probs_api(feature, sigma, mu)
print('torch.distributions probs:', probs)

probs = feature_probs_manual(feature, sigma, mu)
print('torch manual probs:', probs)
```



#### 视频压缩

例如ScaleSpaceFlow([论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html))：

compressai.zoo.**ssf2020**(*quality*, *metric='mse'*, *pretrained=False*, *progress=True*, ***kwargs*)，参数含义与图像压缩一致。

[LIC & DVC](https://github.com/ZhihaoHu/PyTorchDataCompression)

## UTILS



## CLIC

### FAQ

如果我作为公司团队的一员参加，我是否有资格获得奖金？
不会。您有资格获得奖项和获奖证书，但我们会将奖金分配给下一个符合条件的学生团队。我们这样做是为了让更多的学生能够亲自参加（如果会议是亲自举行的）。
