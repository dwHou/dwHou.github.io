主页：https://humanaigc.github.io/emote-portrait-alive/

效果非常好，个人认为商业价值可以比拟sora，只是通用（**general video synthesis**）模型和垂直领域（**generation of human-centric videos**）模型的区分。让我们走进它的论文：

#### 目标 Goal

我们的目标是建立一个创新的说话人框架，旨在捕捉广泛的真实面部表情，包括细微的微表情，并促进自然的头部运动，从而使生成的头部视频具有无与伦比的表现力。

为了实现这一目标，我们提出了一种利用<font color="brown">扩散模型</font>的生成能力的方法，能够直接从给定的图像和音频剪辑合成角色头部视频。 这种方法消除了对中间表示或复杂预处理的需要，简化了头部说话视频的创建，这些视频表现出高度的视觉和情感保真度，与音频输入中存在的细微差别紧密结合。

https://andlukyane.com/blog/paper-review-diffusion-dpo