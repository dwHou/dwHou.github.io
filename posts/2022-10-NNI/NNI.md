# AutoML

Katib, Optuna, Ray Tune, Google Vizier, HyperOpt 和 NNI(Neural Network Intelligence)

[AutoML电子书](https://www.automl.org/book/)

### 前言

过去十年见证了机器学习研究和应用的爆炸式增长；特别是，深度学习方法在许多应用领域取得了重大进展，例如计算机视觉、语音处理和游戏。然而，许多机器学习方法的性能对过多的<font color="deep yellow">设计决策非常敏感</font>，这对新用户构成了相当大的障碍。在蓬勃发展的深度学习领域尤其如此，人类工程师需要为所有这些组件选择正确的神经架构、训练程序、正则化方法和超参数，以使他们的网络完成他们应该做的事情足够的性能。必须为每个应用程序重复此过程。即使是专家也经常要经历<font color="deepyellow">繁琐的反复试验</font>，直到他们为特定数据集确定一组好的选择。<font color="deepyellow">自动化机器学习 (AutoML) </font>领域旨在以数据驱动、客观和自动化的方式做出这些决策：用户只需提供数据，AutoML 系统就会自动确定最适合该特定应用程序的方法。因此，AutoML 使对应用机器学习感兴趣但没有资源详细了解其背后技术的其他领域科学家可以使用最先进的机器学习方法。这可以看作是机器学习的民主化：使用 AutoML，定制的最先进的机器学习触手可及。正如我们在本书中所展示的，AutoML 方法已经足够成熟，可以与人类机器学习专家相媲美，有时甚至超越人类机器学习专家。简而言之，AutoML 可以提高性能，同时节省大量时间和金钱，因为机器学习专家既难找又昂贵。因此，近年来对 AutoML 的商业兴趣急剧增长，几家主要的科技公司现在正在开发自己的 AutoML 系统。不过，我们注意到，开源 AutoML 系统比专有的付费黑盒服务更能实现机器学习民主化的目的。本书概述了 AutoML 快速发展的领域。由于社区目前对深度学习的关注，现在一些研究人员错误地将 AutoML 等同于神经架构搜索（NAS）的主题；但是，当然，如果您正在阅读这本书，您就会知道——虽然 NAS 是 AutoML 的一个很好的例子——但 AutoML 比 NAS 有更多的东西。

本书旨在为有兴趣开发自己的 AutoML 方法的研究人员提供一些背景和起点，重点向希望将 AutoML 应用于其问题的从业者介绍可用的系统，并为已经在该领域工作的研究人员提供最新技术的概述。

本书分为三个部分，分别介绍 AutoML 的这些不同方面。

1. 第一部分概述了 AutoML 方法。这部分既为新手提供了坚实的概述，也为经验丰富的 AutoML 研究人员提供了参考。<font color="deepyellow">第 1 章讨论了超参数优化问题</font>，这是 AutoML 考虑的最简单和最常见的问题，并描述了应用的各种不同方法，特别关注当前最有效的方法。<font color="deepyellow">第2章展示了如何学会学习</font>，即如何利用评估机器学习模型的经验来告知如何使用新数据处理新的学习任务。这些技术模仿了人类从机器学习新手到专家的转变过程，并且可以极大地减少在全新的机器学习任务上获得良好性能所需的时间。章。图 3 提供了 NAS 方法的全面概述。这是 AutoML 中最具挑战性的任务之一，因为设计空间非常大，神经网络的单次评估可能需要很长时间。然而，该领域非常活跃，并且经常出现新的解决 NAS 的令人兴奋的方法。







# NNI

NNI (Neural Network Intelligence) 是一个轻量而强大的工具，可以帮助用户 **自动化**：

- [超参调优](https://nni.readthedocs.io/zh/stable/hpo/overview.html)
- [架构搜索](https://nni.readthedocs.io/zh/stable/nas/overview.html)
- [模型压缩](https://nni.readthedocs.io/zh/stable/compression/overview.html)
- [特征工程](https://nni.readthedocs.io/zh/stable/feature_engineering/overview.html)
