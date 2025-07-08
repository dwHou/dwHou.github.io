https://github.com/rickyang1114/DDP-practice/tree/main

https://github.com/pytorch/examples/tree/main/imagenet

简易DDP模版，只考虑单机多卡，完整程序在[这里](https://github.com/rickyang1114/DDP-practice/blob/main/ddp_main.py)。

### 入口

首先，我们在`if __name__ == '__main__'`中启动 DDP：

```python
if __name__ == '__main__':
    args = prepare()  ###
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())  #import torch.multiprocessing as mp
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

`spawn`函数的主要参数包括以下几个：

1. `fn`，即上面传入的`main`函数。每个线程将执行一次该函数
2. `args`，即`fn`所需的参数。传给`fn`的参数必须写成元组的形式，哪怕像上面一样只有一个
3. `nprocs`启动的进程数，将其设置为`world_size`即可。不传默认为1，与`world_size`不一致会导致进程等待同步而一直停滞。

入口 → spawn nprocs=4 → 启动4个训练进程
       ↘ 每个子进程 → 初始化 DDP（world_size=4, rank=0~3）
                 ↘ 所有进程通过通信组形成 DDP 同步机制

再假设有个复杂场景：**跨两台机器进行 DDP 训练**，但每台机器上使用的 GPU 数不同：

- **机器 A（8卡）**：只使用 GPU 3、4、5（共 3 张）
- **机器 B（4卡）**：使用所有 GPU 0、1、2、3（共 4 张）

总共 **7 个进程 / 7 张卡**，此时：

| 参数         | 值                 | 说明                           |
| ------------ | ------------------ | ------------------------------ |
| `world_size` | `7`                | 全部进程数量 = 3 + 4           |
| `nprocs`     | 每台机器的卡数     | A 机器是 `3`，B 机器是 `4`     |
| `rank`       | 每个进程的全局编号 | A 机器是 `0~2`，B 机器是 `3~6` |

机器A启动：

```shell
# 只在这台机器使用 GPU 3、4、5
CUDA_VISIBLE_DEVICES=3,4,5 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
WORLD_SIZE=7 \
RANK=0 \
python train_ddp.py
```

机器B启动：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
WORLD_SIZE=7 \
RANK=3 \ # 对应全局 rank
python train_ddp.py
```

所以多机多卡对我们训练来说差不多的。类似 `rank = int(os.environ["RANK"])` 获取一下global rank即可。 底层通信机制不一样，但无需感知。

> 本地进程间通信（通过 NCCL 共享内存 + socket）、跨节点网络通信（NCCL over TCP）
>
> 单机多卡时MASTER_ADDR配置为本地回环地址localhost，而多机多卡时会配置为主节点IP。

### 初始化

在`prepare`函数里面，也进行了一些 DDP 的配置：

```python
def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('-e',
                        '--epochs',
                        default=3,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='number of batchsize')
    args = parser.parse_args()
    
    # 下面几行是新加的，用于启动多进程 DDP。使用 torchrun 启动时只需要设置使用的 GPU
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的本地回环地址。
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口，随便选一个没被占用的
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些 GPU
    world_size = torch.cuda.device_count() # 就是上一行使用的 GPU 数量
    os.environ['WORLD_SIZE'] = str(world_size)
    return args
```

### 主函数

再来看看`main`函数里面添加了什么。首先是其添加一个额外的参数`local_rank`（在`mp.spawn`里面不用传，会自动分配，但都是从0开始，所以只能叫local rank）

```python
def main(local_rank, args):
    init_ddp(local_rank)  ### 进程初始化
    model = ConvNet().cuda()  ### 模型的 forward 方法变了
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ### 转换模型的 BN 层
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  ### 套 DDP
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    scaler = GradScaler()  ###  用于混合精度训练
    
    train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)  ### 用于在 DDP 环境下采样
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### shuffle is mutually exclusive with sampler
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        generator=g,
    )  ### 添加额外的 generator，随机种子
    
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset
    )  ### 用于在 DDP 环境下采样
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=test_sampler,
    )
        test_dataset)  
   
    for epoch in range(args.epochs):
        if local_rank == 0:  ### 防止每个进程都输出一次
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_dloader.sampler.set_epoch(epoch)  ### DDP里是需要控制shuffle的种子
        train(model, train_dloader, criterion, optimizer, scaler)
    if local_rank == 0:
        print(f'begin testing')
    test(model, test_dloader)
    if local_rank == 0:  ### 防止每个进程都保存一次
        torch.save({'model': model.state_dict(), 'scaler': scaler.state_dict()}, 'ddp_checkpoint.pt')
    dist.destroy_process_group()  ### 最后摧毁进程，和 init_process_group 相对
```

#### DDP初始化

首先，根据用`init_ddp`函数对模型进行初始化。这里我们使用 nccl 后端，并用 env 作为初始化方法：

```python
def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local_rank)
    torch.cuda.set_device(local_rank)  
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    # 或 dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 但更推荐init_method='env://'
    # 你只需要 设置好环境变量，PyTorch 会自动读取并初始化：
    # shell 中设置：
    # MASTER_ADDR=192.168.1.10
    # MASTER_PORT=12345
    # RANK=3
    # WORLD_SIZE=8
    
```

> 不过我觉得device = torch.device('cuda:{}'.format(rank))，然后统一to(device)更好

在完成了该初始化后，可以很轻松地在需要时获得`local_rank`、`world_size`，而不需要作为额外参数从`main`中一层一层往下传。

```python
import torch.distributed as dist
local_rank = dist.get_rank()
world_size = dist.get_world_size()
```

比如需要`print`, `log`, `save_state_dict`时，由于多个进程拥有相同的副本，故只需要一个进程执行即可，比如：

```python
if local_rank == 0:
    print(f'begin testing')
if local_rank == 0:  ### 防止每个进程都保存一次
    torch.save({'model': model.state_dict(), 'scaler': scaler.state_dict()}, 'ddp_checkpoint.pt')
```

#### 模型

为了加速推理，我们在模型的`forward`方法里套一个`torch.cuda.amp.autocast()`：

使得`forward`函数变为：

```python
def forward(self, x):
    with torch.cuda.amp.autocast():  # 混合精度，加速推理
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
    return out
```

autocast 也可以在推理的时候再套，但是在这里套最方便，而且适用于所有情况。

在模型改变之后，使用`convert_sync_batchnorm`和`DistributedDataParallel`对模型进行包装。

#### scaler

创建 scaler，用于训练时对 loss 进行 scale：

```python
from torch.cuda.amp import GradScaler
scaler = GradScaler()  ###  用于混合精度训练
```

### 训练

训练时，需要使用 DDP 的sampler，并且在`num_workers > 1`时需要传入`generator`，否则对于同一个worker，所有进程的augmentation相同，减弱训练的随机性。详细分析参见[这篇文章](https://zhuanlan.zhihu.com/p/618639620)。

```python
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)  ### 用于在 DDP 环境下采样
g = get_ddp_generator()  ###
train_dloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,  ### shuffle 通过 sampler 完成
                                            num_workers=4,
                                            pin_memory=True,
                                            sampler=train_sampler,
                                            generator=g)  ### 添加额外的 generator
```

并且在多个`epoch`的训练时，需要设置`train_dloader.sampler.set_epoch(epoch)`。

下面来看看`train`函数。

```python
def train(model, train_dloader, criterion, optimizer, scaler):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()  ###
```

最后三行发生了改变。相较于原始的`loss.backward`、`optimizer.step()`，这里通过`scaler`对梯度进行缩放，防止由于使用混合精度导致损失下溢，并且对`scaler`自身的状态进行更新呢。如果有多个`loss`，它们也使用同一个`scaler`。如果需要保存模型的`state_dict`并且在后续继续训练（比如预训练-微调模式），最好连带`scaler`的状态一起保留，并在后续的微调过程中和模型的参数一同加载。

### 测试

测试时，需要将多个进程的数据`reduce`到一张卡上。注意，在`test`函数的外面加上`if local_rank == 0`，否则多个进程会彼此等待而陷入死锁。

```python
def test(model, test_dloader):
    local_rank = dist.get_rank()
    model.eval()
    size = torch.tensor(0.).cuda()
    correct = torch.tensor(0.).cuda()
    for images, labels in test_dloader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
        correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)  ###
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)  ###
    if local_rank == 0:
        acc = correct / size
        print(f'Accuracy is {acc:.2%}')
```

注释的两行即为所需添加的`reduce`操作。

至此，添加的代码讲解完毕。

启动的方式变化不大：

```bash
python ddp_main.py --gpu 0,1
```

相应的结果：

```bash
begin training of epoch 1/3
begin training of epoch 2/3
begin training of epoch 3/3
begin testing
Accuracy is 89.21%

time elapsed: 30.82 seconds
```

## 用torchrun启动

上述是通过`mp.spawn`启动。`mp`模块对`multiprocessing`库进行封装，并没有特定针对`DDP`。我们还可以通过官方推荐的`torchrun`进行启动。完整的程序在[这里](https://github.com/rickyang1114/DDP-practice/blob/main/ddp_main_torchrun.py)。

相比`mp.spawn`启动，`torchrun`自动控制一些环境变量的设置，因而更为方便。我们只需要设置`os.environ['CUDA_VISIBLE_DEVICES']`即可（不设置默认为该机器上的所有GPU），而无需设置`os.environ['MASTER_ADDR']`等。此外，`main`函数不再需要`local_rank`参数。程序入口变为：

```python
if __name__ == '__main__':
    args = prepare()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

运行脚本的命令由`python`变为了`torchrun`，如下：

```bash
torchrun --standalone --nproc_per_node=2 ddp_main_torchrun.py --gpu 0,1
```

其中，`nproc_per_node`表示进程数，将其设置为使用的GPU数量即可。



`torchrun` 是 PyTorch 官方推荐的启动分布式训练（DDP）的工具，是对早期 `python -m torch.distributed.launch` 的升级替代。它非常适合 **单机多卡** 和 **多机多卡** 的训练脚本。

```shell
torchrun [OPTIONS] your_training_script.py [SCRIPT_ARGS...]
```

## 常用启动参数（OPTIONS）

| 参数                 | 作用                                | 示例                          |
| -------------------- | ----------------------------------- | ----------------------------- |
| `--nproc_per_node`   | 当前节点使用的 GPU 数量（即几张卡） | `--nproc_per_node=4`          |
| `--nnodes`           | 总共多少个节点（机器）              | `--nnodes=2`                  |
| `--node_rank`        | 当前节点的编号（0 开始）            | `--node_rank=0`               |
| `--master_addr`      | 主节点 IP（或 hostname）            | `--master_addr=192.168.1.100` |
| `--master_port`      | 主节点端口（通信端口）              | `--master_port=29500`         |
| `--rdzv_backend`     | 通信后端，一般默认不写              | 默认使用 `c10d`               |
| `--max_restarts`     | 进程最大重启次数                    | 可选，调试用                  |
| `--monitor_interval` | 监测失败进程的间隔时间（秒）        | 可选                          |

> 每个进程的 `RANK` 会自动计算为 `node_rank * nproc_per_node + local_rank`
>
> 小知识：不同节点GPU数量不一样，nproc_per_node设置为不同即可。只是很少有这种场景。

```python

# 方案1: mp.spawn()
    mp.spawn(main_fun,
             args=(opt.world_size, opt),
             nprocs=opt.world_size,
             join=True)
# 方案2: Process
    from torch.multiprocessing import Process
    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=main_fun, args=(rank, world_size, opt))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# 方案3: torchrun
```



## Checklist

在写完 DDP 的代码之后，最好检查一遍，否则很容易因为漏了什么而出现莫名奇妙的错误，比如程序卡着不动了，也不报错）

大致需要检查：

1. DDP 初始化有没有完成，包括`if __name__ == '__main__'`里和`main`函数里的。退出`main`函数时摧毁进程。
2. 模型的封装，包括autocast，BN 层的转化和 DDP 封装
3. 指定`train_dloader`的`sampler`、`generator`和`shuffle`，并且在每个`epoch`设置`sampler`，测试集、验证集同理。
4. 训练时使用`scaler`对`loss`进行`scale`
5. 对于`print`、`log`、`save`等操作，仅在一个线程上进行。
6. 测试时进行`reduce`

## PS

多个线程大致相当于增大了相应倍数的`batch_size`，最好相应地调一调`batch_size`和学习率。本文没有进行调节，导致测试获得的准确率有一些差别。

模型较小时速度差别不大，反而DDP与混合精度可能因为一些初始化和精度转换耗费额外时间而更慢。在模型较大时，DDP + 混合精度的速度要明显高于常规，且能降低显存占用。