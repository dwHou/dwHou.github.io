# Neural Engine

苹果的神经引擎The Apple Neural Engine (ANE) 属于一种**NPU**，即专门加速了神经网络的算子比如卷积、矩阵乘法。

由于苹果官方没有向第三方开发人员提供任何指导，主要靠反复试验试错，我摘录并翻译了一份非官方文档，所以其中一些信息可能是错误的。

[TOC]

#### 1.哪些设备有ANE

从iphone8开始搭载的**A11～A15** **Bionic**芯片，以及MacBook(2020)开始搭载的**M1**芯片，但只有搭配Apple Silicon的机器有，而搭配Intel芯片的没有。

M1和 A14 Bionic 中的神经引擎很可能是相同的。



#### 2.为什么要关注ANE

它比 CPU 或 GPU 快得多！ 而且它更节能。

在 ANE 上运行模型将使 GPU 腾出时间来执行图形任务，并腾出 CPU 来运行应用程序的其余部分。

考虑一下：许多现代神经网络架构实际上在 CPU 上比在 GPU 上运行得更快（使用 Core ML 时）。那是因为 iPhone 的 CPU 速度非常快！另外，调度任务在 GPU 上运行总是有一定的开销，这可能会抵消任何速度提升。

鉴于这些事实，您应该在 CPU 上而不是在 GPU 上运行您的神经网络吗？不会——你需要 CPU 来处理 UI 事件、处理网络请求等等……它已经够忙了。最好将这项工作交给 GPU 、ANE这样的并行处理器。

所以**我们应该尽可能调整模型和ANE兼容**。



#### 3.模型运行在ANE上

Core ML 将尽可能尝试在 ANE 上运行您的模型，但您不能强制 Core ML 使用 ANE。

```swift
let config = MLModelConfiguration()
config.computeUnits = .all

let model = try MyModel(configuration: config)
```

使用computeUnits = .all就能够允许模型运行在ANE上。

如果可能，Core ML 将在 ANE 上运行整个模型。 但它会在遇到不受支持的层时切换到另一个处理器。 你不能假设是会切到CPU还是GPU上。 而且Core ML还可以将模型分成多个部分，并使用不同的处理器运行每个部分。 因此它可能在同一个推理过程中同时使用 ANE 和 CPU 或 GPU。

Core ML 为不同的处理器使用以下框架：

- CPU：BNNS，或 Basic Neural Network Subroutines，Accelerate.framework 的一部分
- GPU：Metal Performance Shaders  (MPS)
- ANE：私有frameworks

iPhone、iPad 和 Apple Silicon Mac 具有共享内存，这意味着 CPU、GPU 和 ANE 都使用相同的 RAM。这样的硬件架构能服务于上述运行特性，不过虽然贡享内存并不意味着处理器之间切换没有成本：数据仍然需要转换成合适的格式。



#### 4.模型不运行在ANE

使用computeUnits = .cpuOnly或者.cpuAndGPU的配置来初始化模型，Core ML就用不到ANE。

大多情况使用这个模式，就是因为Core ML的bugs导致有些模型能正常运行在CPU和GPU上，但运行在ANE上会出现奇怪报错。



#### 5.怎么看模型是否在ANE上运行？

调试器时按下暂停按钮。 如果有一个名为 H11ANEServicesThread 的线程，那么 Core ML 至少在模型的某些部分使用了神经引擎。



#### 6.ANE是16-bit的吗？

看起来是这样。

在CPU上，Core ML模型计算和存储中间张量都是用FP32（iOS14和macOS11开始支持在CPU上使用FP16），

在GPU上，Core ML模型存储权重和中间张量用FP16，但计算用FP32，

在ANE上，

