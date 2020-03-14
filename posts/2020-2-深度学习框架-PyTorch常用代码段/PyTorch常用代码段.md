# PyTorch常用代码段

这里是PyTorch中常用代码段的一份整理，并且持续更新，初稿在参考资料[1](张皓：PyTorch Cookbook)的基础上进行编辑补充，方便使用时查阅。

```python
#本文需要用到以下包
import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision
```



## 1.基本配置
### 导入包和版本查询
```python
import torch
import torch.nn as nn
import torchvision
print(torch.__version__)              #PyTorch version
print(torch.version.cuda)             #Corresponding CUDA version
print(torch.backends.cudnn.version()) #Corresponding cuDNN version
print(torch.cuda.get_device_name(0))  #GPU type
```
### 可复现性

在硬件设备（CPU、GPU）不同时，完全的可复现性无法保证，即使随机种子相同。但是，在同一个设备上，应该保证可复现性。具体做法是，在程序开始的时候固定torch的随机种子，同时也把numpy的随机种子固定。

```python
torch.manual_seed(0)          #为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(0)     #为当前GPU设置随机种子
torch.cuda.manual_seed_all(0) #如果使用多个GPU，为所有的GPU设置种子
np.random.seed(0)

torch.backends.cudnn.benchmark = True      #cuDNN benchmark模式,会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
torch.backends.cudnn.deterministic = True  #避免这种结果波动
```
### 显卡设置

**指定程序运行在特定GPU卡上。**

在命令行指定环境变量

```shell
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

或在代码中指定

```python
如果只需要一张显卡
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
如果需要指定多张显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

**清除GPU存储**
有时Ctrl+C中止运行吼GPU存储没有及时释放，需要手动清空。在PyTorch内部可以

```python
torch.cuda.empty_cache()
```

或在命令行

```shell
#先使用ps找到程序的PID,再使用kill结束该进程
ps aux | grep python
kill -9 [pid]
#或直接重置没有被清空的GPU
nvidia-smi --gpu-reset -i [gpu_id]
```

## 2.张量(Tensor)处理

### 张量基本信息

```python
tensor.type()  #Data type
tensor.size()  #Shape of tensor. 
tensor.dim()   #Number of dimensions
```

### 数据类型转换

```python
#Set default tensor type. Float in PyTorch is much faster than double.
torch.set_default_tensor_type(torch.FloatTensor) #or torch.cuda.FloatTensor

#Type conversions.
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
```

### torch.Tensor↔np.ndarray

除了CharTensor，其他所有CPU上的张量都支持转换为numpy格式然后转换回来。

```python
#torch.Tensor → np.ndarray.
ndarray = tensor.cpu().numpy()

#np.ndarray → torch.Tensor.
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()  #If ndarray has negative stride.
```

### torch.Tensor↔PIL.Image

PyTorch中的张量默认采用N×C×H×W的顺序，并且数据范围在[0, 1]，需要进行转置和规范化。

```python
# torch.Tensor → PIL.Image.
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0,max=255).byte().permute(1,2,0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor) #Equivalently way

# PIL.Image → torch.Tensor.
path = r'./figure.jpg' #r'string'是raw_string用来防止\ 自动转义的
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float()/255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) #Equivalently way
```

### np.ndarray↔PIL.Image

```python
# np.ndarray → PIL.Image.
image = PIL.Image.fromarray(ndarray.astype(np.unit8))

# PIL.Image → np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
```





https://www.zhangshengrong.com/p/Ap1Zp295N0/

## 3.模型定义和操作







[1]: https://zhuanlan.zhihu.com/p/104019160	"待补充"

