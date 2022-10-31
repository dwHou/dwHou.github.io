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

# benchmark是决定faster，deterministic是决定reproducible
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



### 多gpu并行训练

我一般在使用多GPU的时候, 会喜欢使用`os.environ['CUDA_VISIBLE_DEVICES']`来限制使用的GPU个数, 例如我要使用第0和第3编号的GPU, 那么只需要在程序中设置:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
```

**1.单机多卡**
**DataParallel (DP) :**  Parameter Server模式，一张卡位reducer，实现也超级简单，一行代码。

```python
#模型
if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行
    model = nn.DataParallel(model)
    
#数据    
inputs = inputs.cuda()
labels = labels.cuda()
```



**2.多机多卡**

**DistributeDataParallel (DDP) :**  All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。

```python
import torch.multiprocessing as mp
#import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
```

1.初始化后端

```python
torch.distributed.init_process_group(backend="nccl")
```

2.模型并行化

```python
model=torch.nn.parallel.DistributedDataParallel(model)
```

**最小例程与解释**

☞训练一个MNIST分类的简单卷积网络。

```python
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')

parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
                        
parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

args = parser.parse_args()
train(0, args)
#训练是这么定义的
def train(gpu, args):
#上述代码中肯定有一些我们还不需要的额外的东西(例如gpu和节点的数量)，但是将整个框架放置到位是很有帮助的。
之后在命令行输入 python src/mnist.py -n 1 -g 1 -nr 0
就可以在一个结点上的单个GPU上训练啦~
```

☞加上MultiProcessing

<font size=3>(我们需要一个脚本，用来启动一个进程的每一个GPU。每个进程需要知道使用哪个GPU，以及它在所有正在运行的进程中的阶序（rank）。而且，我们需要**在每个节点上运行脚本**。)</font>

- - args.nodes 是我们使用的结点数
  - args.gpus 是每个结点的GPU数.
  - args.nr 是当前结点的阶序rank，取值范围是 0 到 args.nodes - 1.

```python
		args = parser.parse_args()
    #########################################################
14  args.world_size = args.gpus * args.nodes                #
15  os.environ['MASTER_ADDR'] = '10.57.23.164'              #
16  os.environ['MASTER_PORT'] = '8888'                      #
17  mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################
```

- [比较复杂，还是用到时查看这篇tutorial](https://zhuanlan.zhihu.com/p/105755472)

**3.Apex混合精度训练**

***(在2的基础上)***

```python
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

最近Apex更新了API，只需上述三行代码即可实现**混合精度加速**。

[基于Apex的混合精度加速](https://zhuanlan.zhihu.com/p/79887894)

[混合精度训练与模型并行适配](https://bindog.github.io/blog/2020/04/12/model-parallel-with-apex/)











- **DP**是基于Parameter server的算法，负载不均衡的问题比较严重，有时在模型较大的时候（比如bert-large），reducer的那张卡会多出3-4g的显存占用。

  [优秀的tutorial](https://zhuanlan.zhihu.com/p/105755472)

  <font color='blue'>nn.DataParallel</font>使用起来更加简单（通常只要封装模型然后跑训练代码就ok了）。但是在每个训练批次（batch）中，因为模型的权重都是在 一个进程上先算出来 然后再把他们分发到每个GPU上，所以网络通信就成为了一个瓶颈，而GPU使用率也通常很低。

  

- 官方建议用新的**DDP**，采用all-reduce算法，本来设计主要是为了多机多卡使用，但是单机上也能用。

  <font color='blue'>nn.DistributedDataParallel</font>进行Multiprocessing可以在多个gpu之间复制该模型，每个gpu由一个进程控制。每个进程都执行相同的任务，并且每个进程与所有其他进程通信。只有梯度会在进程/GPU之间传播，这样网络通信就不至于成为一个瓶颈了。

  <font size=1><font color='red'>注：</font>训练过程中，每个进程从磁盘加载自己的小批（minibatch）数据，并将它们传递给自己的GPU。每个GPU都做它自己的前向计算，然后梯度在GPU之间全部约简。每个层的梯度不仅仅依赖于前一层，因此梯度全约简与并行计算反向传播，进一步缓解网络瓶颈。在反向传播结束时，每个节点都有平均的梯度，确保模型权值保持同步（synchronized）。</font>

  

- 混合精度训练，即组合浮点数 (FP32)和半精度浮点数 (FP16)进行训练，允许我们使用更大的batchsize，并利用[NVIDIA张量核](https://link.zhihu.com/?target=https%3A//www.nvidia.com/en-us/data-center/tensorcore/)进行更快的计算。

  我们只需要修改 train 函数即可





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
image = PIL.Image.fromarray(ndarray.astype(np.uint8))

# PIL.Image → np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
```





https://www.zhangshengrong.com/p/Ap1Zp295N0/

## 3.模型定义和操作

### 加载模型
**GPU → CPU**

```python
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
```
或
```python
model.load_state_dict(torch.load('model.pth', map_location=lambda storage, loc: storage))
```

**CPU → GPU**
```python
model.load_state_dict(torch.load('model.pth', map_location=lambda storage, loc: storage.cuda(1)))
# 加载到GPU1中
```

**GPU→GPU**
```python
model.load_state_dict(torch.load('model.pth', map_location={'cuda:1':'cuda:0'}))
```

**多GPU → CPU** 
保存了模型nn.DataParallel，该模型将模型存储在该模型中module，而现在您正试图加载模型DataParallel。您可以nn.DataParallel在网络中暂时添加一个加载目的，也可以加载权重文件，创建一个没有module前缀的新的有序字典，然后加载它。
```python
解决方案有两个：
1：此时的训练加入torch.nn.DataParallel()即可。
2：创建一个没有module.的新字典，即将原来字典中module.删除掉。
解决方案1：

model = torch.nn.DataParallel(model)
cudnn.benchmark = True


解决方案2：
# original saved file with DataParallel
state_dict = torch.load('myfile.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

解决方案3：
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('myfile.pth').items()})
```



[1]: https://zhuanlan.zhihu.com/p/104019160	"PyTorch Cookbook"
[2]: https://zhuanlan.zhihu.com/p/435669796	"flexible learning rate"





