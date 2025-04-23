## AI工程师

> [!NOTE]
>
> 我倾向称之为LLM工程师，或者AI API 开发工程师。

AI工程师的崛起  [link](https://www.latent.space/p/ai-engineer)
 新兴能力正在创造一个新兴的职位：要掌握这些能力，我们需要超越提示词工程师，编写 *软件* 以及能够编写软件的人工智能。

![img](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa81555af-0b76-4a61-9b53-595e3d47580a_1005x317.png)

我们正在观察到一次千载难逢的“右移”趋势，应用人工智能的发展受到新兴能力和基础模型的开源/API 可用性的推动。

在2013年，完成的广泛人工智能任务通常需要[五年的时间和一个研究团队](https://xkcd.com/1425/)，而在2023年，现在只需查阅 API 文档和腾出一个下午的时间即可实现。

> “从数字上来看，人工智能工程师的数量可能会显著超过机器学习工程师和大型语言模型工程师。一个人在这个角色中可以取得相当大的成功，而无需训练任何东西。” - 安德烈·卡尔帕西

为什么AI工程师岗位需求越来越大：

1. 基础模型的泛化生成能力是“涌现”的，即使研究者也没法完全明白其中原理。
2. 微软、谷歌、Meta 和大型基础模型的实验室已经聚集了稀缺的研究人才。旨在提供“AI 研究即服务”的 API。你无法雇佣他们，但可以租用他们。
3. GPU囤积 & 全球芯片短缺
4. “开火、准备、瞄准”的LLM原型工作流程，可以让你比传统机器学习的落地速度快10到100倍。验证的成本也会便宜千倍、万倍。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6035ddd2-418d-421d-aebd-6893c32bb6dd_1579x266.png)

5. Python + JavaScript。数据/人工智能传统上极其以 Python 为中心，首批人工智能工程工具如 LangChain、LlamaIndex 和 Guardrails 都源于这个社区。然而，JavaScript 开发者的数量至少与 Python 开发者相当，因此现在工具越来越多地迎合这一广泛扩展的受众，从 LangChain.js 和 Transformers.js 到 Vercel 的新 AI SDK。市场规模的扩展和机会至少增加了 100%。
6. 生成式AI 与 传统机器学习的截然不同。 每当出现一个具有完全不同背景、讲完全不同语言、生产完全不同产品、使用完全不同工具的子群体时，他们最终就会分裂成自己的群体。

> [!NOTE]
>
> [图像](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55d13fad-b282-4d9c-9258-d63a507ee002_2736x1494.jpeg)
>
> [主要的架构分歧](https://twitter.com/swyx/status/1674072122707046402/photo/1)：“智能之上的软件” vs “智能软件”



## AI业务创新

两板斧 —— RAG & SFT

**RAG**（Retrieval-Augmented Generation）是一种将大语言模型与外部知识库结合的技术。它通过检索相关文档或知识来增强模型的回答能力，主要包含以下步骤：  

- 检索（Retrieval）：根据用户输入查询相关的文档或知识 
- 增强（Augmentation）：将检索到的信息与原始查询结合 
- 生成（Generation）：利用大语言模型基于增强后的上下文生成回答

> [!IMPORTANT]
>
> 核心：调 Prompt^*^、知识库预处理、问题集生成。
>
> ^*^：相比于程序严丝合缝的传参， LLM 工程化系统直接以自然语言互相交流。

RAG的主要优势在于： - 可以访问最新信息 - 提供可验证的知识来源 - 减少模型幻觉 - 无需完整重新训练模型

**SFT** (Supervised Fine-Tuning) 是一种通过人工标注数据来微调大语言模型的训练方法。在RAG的上下文中，它指的是对模型进行针对性训练，使其能更好地完成特定任务。

- 通过SFT可以让模型： 
  - 更准确地理解和处理检索到的信息
  - 生成更符合特定场景需求的回答
  - 提高模型在RAG框架中的表现

实践上，SFT 常用来把垂直领域的知识注入 LLM，比如垂直领域知识，低代码平台知识等。分析数据和清洗数据就是 SFT 90% 的工作量。

> SFT出来的领域模型可以不在乎其他通用能力，对垂直领域过拟合即可。

一般来说市面上的公司实践类似 AI agent 业务，简单的就是 RAG ， 有追求的是给一个微调后的模型。

**我们今后要做AI 业务优化，最终都会走到这两条路上来，这也是业界普遍的已经落地的实践。**