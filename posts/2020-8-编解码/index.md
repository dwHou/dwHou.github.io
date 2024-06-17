# 视频编解码

视频编码标准，是IT巨头们“跑马圈地”的游戏，想要pick助其出道。 我们需要更高效压缩标准原因可以通过Jevons悖论来解释：业界对节约煤炭研究不会降低煤炭需求，反而会因为提升了煤炭使用效率而加大对煤炭的需求。

## 综述

[使用深度神经网络的视频压缩系统的进展](https://purdueviper.github.io/dnn-coding/)

## 新一代视频编解码标准

AV1，VVC

## 深度学习辅助的混合编码框架

<img src="./DL-based-coding.png" alt="DL-based-coding" style="zoom:50%;" />

Liu, Dong, et al. "Deep learning-based video coding: A review and a case study." *ACM Computing Surveys (CSUR)* 53.1 (2020): 1-35.

- AI辅助块划分
- AI辅助码率控制
- AI辅助帧内预测，生成参考帧
- 基于AI前处理的，端到端编码感知优化

## 端到端视频压缩

## 利用解码信息的视频后处理

## 基于AI模型的视频质量评估

Youtube的实践：[code](https://github.com/google/uvq) | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Rich_Features_for_Perceptual_Quality_Assessment_of_UGC_Videos_CVPR_2021_paper.pdf) | [blog](https://blog.research.google/2022/08/uvq-measuring-youtubes-perceptual-video.html?m=1)

## N卡视频硬编解码

官方文档地址[PDF](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/pdf/Using_FFmpeg_with_NVIDIA_GPU_Hardware_Acceleration.pdf)

https://developer.nvidia.com/ffmpeg

显卡支持矩阵[Reference](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new#Encoder)

https://juejin.cn/post/7034411980316213256

## GPL协议

FFmpeg采用LGPL或GPL （GNU General Public License）许可证。

比如FFmpeg中的某些组件(如libavcodec、libavformat等)是独立发布的, 遵循GPL许可。这里就有我们经常用到的libx264。

> PL要求发布的软件必须附带完整的源代码,或者提供获取源代码的方式。如果您修改了GPL软件并分发修改后的版本,那么新的版本也必须使用GPL许可。这被称为<font color="brown">传染性</font>或<font color="brown">copyleft</font>。

1. GPL许可只污染进程：对于整个项目来说,只有使用了GPL许可软件的那个子进程需要开源。其他没有使用GPL许可软件的部分,并不需要开源。所以我们用子进程调用ffmpeg h.264 encoding没有问题。

2. GPL许可软件在on-premise（本地服务器或计算机硬件上）和非on-premise（云服务提供商的基础设施上）部署的区别：

   1. on-premise部署:

      如果你在自己的服务器上部署和运行GPL软件,那么当你对该软件进行任何修改时,你必须将修改后的源代码以GPL许可证的形式发布。这是GPL的copyleft条款要求的。

   2. 非on-premise部署(如云服务):

      如果你将GPL软件部署在云服务(如AWS)上,向用户提供服务,但用户无法访问你的服务器,那么你不需要发布修改后的源代码。因为你只是在提供一个服务,而不是分发该软件。

   > 云计算与传统软件购买之间的关键区别是：在云计算环境下, 用户不是在为运行GPL软件而付费,而是为使用底层的硬件资源而付费。

3. 相关问答——SaaS 漏洞：https://opensource.stackexchange.com/questions/11467/can-i-use-gpl-software-to-provide-a-commercial-cloud-service

   GPLv2和GPLv3只要求在您发布二进制文件时一并发布源代码。在软件即服务(SaaS)的情况下,您并没有发布二进制文件,因此您也没有义务发布您的源代码。这有时被称为"[SaaS loophole](https://resources.whitesourcesoftware.com/blog-whitesource/the-saas-loophole-in-gpl-open-source-licenses)" 
