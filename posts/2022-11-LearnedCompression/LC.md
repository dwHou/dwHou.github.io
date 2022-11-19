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



### 自定义模型

## LIBRARY API

## MODEL ZOO

