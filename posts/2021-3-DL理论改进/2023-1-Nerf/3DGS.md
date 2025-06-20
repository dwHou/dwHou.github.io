## 资源

https://3dgstutorial.github.io/ 

ppt: https://3dgstutorial.github.io/

[3DGS (3D Gaussian Splatting) 原理介绍](https://www.bilibili.com/video/BV1o4UUYbEdW/?share_source=copy_web&vd_source=f26f3b6f5d88e251cea5e8fa6c584e4a)

https://github.com/MrNeRF/awesome-3D-gaussian-splatting

Blog Posts

- [3DGS Introduction](https://huggingface.co/blog/gaussian-splatting) - HuggingFace guide
- [Implementation Details](https://github.com/kwea123/gaussian_splatting_notes) - Technical deep dive
- [Mathematical Foundation](https://github.com/chiehwangs/3d-gaussian-theory) - Theory explanation
- [Capture Guide](https://medium.com/@heyulei/capture-images-for-gaussian-splatting-81d081bbc826) - Image capture tutorial

Talks

- [Gaussian Splats: Ready for Standardization?](https://www.youtube.com/watch?v=0xdPpKSkO3I) - Metaverse Standards Forum 1/28/2025
- [Unity Integration Guide](https://www.youtube.com/watch?v=pM_HV2TU4rU&t=5298s) - Metaverse Standards Forum 5/6/2025

Video Tutorials

- [Getting Started (Windows)](https://youtu.be/UXtuigy_wYc)
- [Gaussian Splats Town Hall - Part 2](https://youtu.be/5_GaPYBHqOo)
- [Two-Minute Explanation](https://youtu.be/HVv_IQKlafQ)
- [Jupyter Tutorial](https://www.youtube.com/watch?v=OcvA7fmiZYM)
- [MIT教程](https://www.scenerepresentations.org/courses/2023/fall/inverse-graphics/)

## 3D GS原理理解

### NeRF到3D GS的演变

#### 简要回顾NeRF原理

- 由相机内参c2w和随机采样的像素pixels确定一条射线$r(t)=o+td$，沿射线采样
- 将采样点$x_i$的位置信息$(x, y, z)$和视角方向信息$(\theta, \varphi)$编码
- 输入神经网络【隐式辐射场】中预测采样点$x_i$的颜色与体密度信息
- 通过体渲染得到沿该射线$r(t)$可观测的最终颜色
- 根据渲染图与真实图的误差，梯度下降更新，优化神经网络参数

#### Point-NeRF：NeRF与<font color="brown">点云</font>的初遇

1. 神经点云：

   为每个点赋予**神经特征**，该特征代表对周围局部3D场景几何和外观的编码，CNN提取。

2. 体渲染过程：

   射线采样点周边搜索半径r内的点的**特征融合**，得到的特征解码得到采样点的颜色和体密度。

3. 自适应点云的生长和剔除：

   在射线具有**最高不透明度的位置**且周围没有点的情况下添加新点；每10k迭代，基于点位于物体表面的概率，设定阈值剔除无用点

#### Plenoxels：无神经网络NeRF+球谐函数

基于稀疏体素网格表示场景，每个体素格点存储体密度和球谐函数系数，

射线采样点的颜色值和体密度，由其所在体素网格的八个格点 三线性插值计算。

**球谐函数**:

简称SH函数，可以看成球坐标系中的傅里叶级数，可用于拟合球面分布，例如光强，记录空间中某个点从不同方向看过去的不同颜色。阶数越高，拟合能力越强。

#### 3D Gaussian Splatting的简要介绍

3D Gaussian Splatting（3D高斯泼溅）

使用大量的3D高斯基元来表示场景，可视作<font color="brown">参数化点云</font>，每个基元有位置，大小，方向，颜色信息（球谐函数系数）和不透明度等信息。

渲染速度有很大优势：

<img src="../../../images/typora-images/image-20250618163027486.png" alt="image-20250618163027486" style="zoom:50%;" />



### 3D GS原理介绍

#### 3D GS框架图介绍

<img src="../../../images/typora-images/image-20250618171234127.png" alt="image-20250618171234127" style="zoom:50%;" />

3D GS高斯椭球形状为什么受协方差矩阵控制？

3D GS高斯椭球如何从3D空间投影到2D空间？

3D GS基于图像块（Tile）的光栅化图像渲染

### NeRF 与 3D GS的对比