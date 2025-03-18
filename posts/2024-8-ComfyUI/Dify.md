Dify 是一个开源的 LLM 应用开发平台。其直观的界面结合了 AI 工作流、RAG 管道、Agent、模型管理、可观测性功能等，让您可以快速从原型到生产。

# AI+Web3: 从理论到实战的进阶之路

## 大模型炼成

<img src="../../images/typora-images/image-20250317211316601.png" alt="image-20250317211316601" style="zoom:50%;" />

引用自 https://www.youtube.com/watch?v=bZQun8Y4L2A

- 预训练的Base model，只能做单词接龙（GPT）或完形填空（Bert）。

- 监督微调的SFT model，可以称为对话（Chat）模型了。经过少量标注数据进行训练后能适应特定的任务。

- 与人类价值观对齐的RLHF/DPO model，变得更可用、实用、好用。基于人类反馈强化学习 / 直接偏好优化 与人类价值观对齐，借以生成更精确、真实的回答。

  > 作为固定的“判别器”的RM，参考<a href="https://www.53ai.com/news/finetuning/2024091160724.html">link</a>。

局限：幻觉；实时知识；复杂推理；私有数据；交互决策；数学（9.9和9.11哪个大）；复杂文字逻辑/歧义，等等。所以使用LLM仍要谨慎。

## 大模型应用

### 01 提示词工程

情境学习（In Context Learning）：提供少量样本示例，Few-shot

思维链（Chain of Thought / CoT）

提示词工程（Prompt Engineering）：明确问题、提供上下文、明确期望、人类反馈（多轮对话）、英文提示词

提示词注入/泄露/越狱：比如被问出了windows11专业版的序列号

### 02 RAG

（Retrieval-Augmented Generation）

1.什么是Embedding/嵌入向量

Embedding是由AI算法生成的高维度的向量数据，代表着数据的不同特征。

- 在embedding空间中，相似的东西应该“近”，不同的东西应该“远”。embeeding一般是五百至几千道维度，但拿二维来理解的话，远近就类似向量的余弦距离。
- 语义和语法关系也会被编码到embedding空间中。例如`King - Man + Woman = Queen`
- embedding是一种通用的数据表示方式，各种形式、模态、规模大数据都可以转化为embedding。

2.多模态Embedding

CLIP模型 embedding维度：512,768

- 收集4亿图像文本对 进行无监督预训练（对比学习）
- 最大化文本表征和对应图像表征的余弦相似度；最小化文本表征和非对应图像表征的余弦相似度
- 广泛用于图像分类、图像生成、图像检索、视觉问答等任务

3.RAG的核心思路

<img src="../../images/typora-images/image-20250318135212132.png" alt="image-20250318135212132" style="zoom:50%;" />

类似于开卷考试。检索出相关信息，和问题一起交给大模型。

### 03 智能体

智能体Agent的概念，还处于非常早期的阶段。

理想很美好：给出指令，并观察其自动化执行，节约做事的时间成本。

但现实很骨感：生成内容不可靠，过程不稳定，严重依赖人工经验判断。

<img src="../../images/typora-images/image-20250318141751597.png" alt="image-20250318141751597" style="zoom:35%;" />

定义：LLM+工具，外部工具（Function Call）实现LLM能力的显式扩展，生成过程可溯源、可解释。

示例：图中有几个穿着红白条纹毛衣的人？

ChatGPT无法完成处理图片，但是通过外接工具，这项任务可以完成。(1) 调用SAM 找到图中所有的人 (2) 调用CLIP判断这些人中哪些符合条件。 并且这些操作由智能体自主进行。

### 04 大模型的下游任务

例如：文本摘要、文本分类、机器翻译、问答、关系抽取、NL2SQL（自然语言问题转化为SQL查询语句）

对某个下游任务进行针对性微调。

- 全参数微调（Full Fine-tuning）：消耗大量资源，不建议
- 低资源微调（Parameter Efficient Fine Tuning）：有很多方法，其中Lora最为常见。

## Dify

= Define + Modify 定义并持续改进你的AI应用，最终Do it for you.



