# Python并行编程 中文版

[Link](https://python-parallel-programmning-cookbook.readthedocs.io/zh-cn/latest/index.html)

## 第一章 认识并行计算和Python

### 1.介绍

本章将介绍一些并行编程的架构和编程模型。对于初次接触并行编程技术的程序员来说，这些都是非常有用的概念；对于经验丰富的程序员来说，本章可以作为基础参考。 *本章中讲述了并行编程的两种解释，第一种解释是基于系统架构的，第二种解释基于程序示例F。* 并行编程对程序员来说一直是一项挑战。 *本章讨论并行程序的设计方法的时候，深入讲了这种编程方法。* 本章最后简单介绍了Python编程语言。Python的易用和易学、可扩展性和丰富的库以及应用，让它成为了一个全能性的工具，当然，在并行计算方面也得心应手。最后结合在Python中的应用讲了线程和进程。解决一个大问题的一般方法是，将其拆分成若干小的、独立的问题，然后分别解它们。并行的程序也是使用这种方法，用多个处理器同时工作，来完成同一个任务。每一个处理器都做自己的那部分工作（独立的部分）。而且计算过程中处理器之间可能需要交换数据。如果，软件应用要求越来越高的计算能力。提高计算能力有两种思路：提高处理器的时钟速度或增加芯片上的核心数。提高时钟速度就必然会增加散热，然后每瓦特的性能就会降低，甚至可能要求特殊的冷却设备。提高芯片的核心数是更可行的一种方案，因为能源的消耗和散热，第一种方法必然有上限，而且计算能力提高没有特别明显。

为了解决这个问题，计算机硬件供应商的选择是多核心的架构，就是在同一个芯片上放两个或者多个处理器（核心）。GPU制造商也逐渐引进了这种基于多处理器核心的硬件架构。事实上，今天的计算机几乎都是各种多核、异构的计算单元组成的，每一个单元都有多个处理核心。

所以，对我们来说充分利用计算资源就显得至关重要，例如并行计算的程序、技术和工具等。

### 2.并行计算的内存架构

根据指令的同时执行和数据的同时执行，计算机系统可以分成以下四类：

- 单指令，单数据 (SISD)
- 单指令，多数据 (SIMD)
- 多指令，单数据 (MISD)
- 多指令，多数据 (MIMD)

这种分类方法叫做“费林分类”:

![../_images/flynn.png](https://python-parallel-programmning-cookbook.readthedocs.io/zh-cn/latest/_images/flynn.png)
