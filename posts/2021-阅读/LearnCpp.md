[TOC]

## 介绍与起步学习

#### 0.1 关于本教程

**课程组织：**

本章会介绍C++的方方面面，它是如何诞生的。接下来的第一章会广泛但浅显地学习C++最基础的概念。后面的章节再深入这些概念，或者引入新的概念。

**课程的学习目标：**

学C++这门语言的同时，学习编程的思想和方法。

通过大量清晰、简洁的代码例子，实践所学的概念。避免*magic hand wave*，就是说跳跃到让学生一头雾水。也有一些练习用的程序，可以自己尝试，再对比参考答案。

最重要的，编程得开心。这才是来到一个正确位置的心态。



#### 0.2 关于编程语言(PL)

机器码 → 汇编语言 → 高级语言

高级语言翻译成计算机能运行的机器码，有两种主要方式：编译 & 解释

**编译器：**可以读入源码，产生一个可执行程序。早期的时候，编译器很简单，产生的机器码没有优化、很慢。现在，编译器已经能产出优化得非常好的机器码了，甚至比程序员写的汇编更加好。

**解释器：**不需要编译一个可执行程序。更加灵活，但效率较低，因为每次执行都需要解释一次。

传统的C/C++/Pascal都是编译型的，而一些“脚本”语言比如Perl/JS倾向于解释型，而有的语言比如Java，会混合使用二者。



#### 0.3 关于C和C++

C和Unix的命运息息相关。当初创造C语言是为了重写Unix（原本是汇编语言版的），增加可移植性，让它可以编译在各种机器上。

C++也诞生于贝尔实验室。1998年被标准化，03年进行了一次大的更新，之后有四个年份也进行了大的更新 (**C++11**, C++14, C++17, 和C++20) 。尤其是C++11被视为一个新的baseline版本，之后便是三年一更。

C和C++的设计哲学：相信开发者。

**C++擅长什么：**

```
C++ 在需要高性能和精确控制内存和其他资源的情况下表现出色。以下是一些最有可能用 C++ 编写的常见应用程序类型：
Video games
Real-time systems (e.g. for transportation, manufacturing, etc…)
High-performance financial applications (e.g. high frequency trading)
Graphical applications and simulations
Productivity / office applications
Embedded software
Audio and video processing
Artificial intelligence and neural networks
```

有一位德国人说：AI is a ressource eater before the lord, as we germans say.

https://ai.stackexchange.com/questions/6185/why-does-c-seem-less-widely-used-than-python-in-ai

其中C++主要是用在 Keras, Torch, TensorFlow等框架的底层。



#### 0.4 关于C++的开发环境

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20210824162102685.png" alt="image-20210824162102685" style="zoom:50%;" />





































































































































