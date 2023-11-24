# 从一个项目入门SD+LoRA

秒鸭相机 “Easy Photo”

项目技术背景：

通过使用5到20个相关图像对特定用户ID的数字化替身进行训练，根据训练过的LoRA模型进行微调，可以使用任意模板生成AI照片。

> finetune model according to the trained LoRA model

步骤：

1. 提供图片，训练特定ID的LoRA model。

   > 注：
   >
   > 1. 训练时的prompt, 也就是图中的caption，可以固定如“easyphoto_face, easyphoto, 1person”。实验发现这种简单固定prompt对方式对LoRA训练的影响不大。
   > 2. 一张图训练一个LoRA模型，用于集成。

2. 将训练好的LoRA model融合到base SD model，这个过程可以视为基于lowrank adaptation technology的模型微调。

3. 推理阶段，选择模版，完成脸部区域的repaint任务。

> 后续有望扩展到其他user ID相关的任务，如try-on

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20231109144739893.png" alt="image-20231109144739893" style="zoom:50%;" />

<img src="/Users/DevonnHou/Library/Application Support/typora-user-images/image-20231109151947633.png" alt="image-20231109151947633" style="zoom:50%;" />

> 但感觉上述流程过于复杂了，最终主要是依赖多种prompt的ControlNet和Face Fuse模型。

EasyPhoto有如下创新：

1. 采用了模型集成，来提升生成真实性。
2. 采用了强化学习来优化LoRA模型，给予id相似度奖励。

