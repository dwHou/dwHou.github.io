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
            GDN(N),
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
bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels) # 我初步理解likelihoods越小，可能意味着熵越小，更有序。相同维度输入经过熵编码后的latent的比特数目越少。

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

## MODEL ZOO

