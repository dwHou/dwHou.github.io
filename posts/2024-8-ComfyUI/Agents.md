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

# 大模型Agent开发框架

**低代码框架**：<font color="brown">无需代码</font>即可完成Agent开发

Coze（零代码）、Dify（低代码）、LangFlow（低代码，LangChain家族的）

**基础框架**：借助大模型<font color="brown">原生能力</font>进行Agent开发

function calling、tool use

**代码框架**：借助代码完成Agent开发

LangChain、LangGraph（LangChain家族的，更灵活复杂）、LlamaIndex

**Multi-Agent框架/架构**：

CrewAI（基于LangChain构建的一个更加上层的框架）、Swarm（轻量、开源，用于教学和实验）、Assistant API（OpenAI的、闭源）、openai-agents（Swarm 的升级版，可投入生产）

> 热门项目AutoGen、MetaGPT



## Coze

## Dify

>  [!NOTE]
>
> Dify 一词源自 Define + Modify，意指定义并且持续的改进你的 AI 应用，它是为你而做的（Do it for you）。

是一款低代码（low code）生成式AI应用创新引擎。最大的竞品是字节的Coze（扣子）。

### Dify官方文档

**Dify** 是一款开源的低代码LLM应用开发平台。它融合了后端即服务（Backend as Service）和 [LLMOps](https://docs.dify.ai/zh-hans/learn-more/extended-reading/what-is-llmops) 的理念，使开发者可以快速搭建生产级的生成式 AI 应用。

由于 Dify 内置了构建 LLM 应用所需的关键技术栈，包括对数百个模型的支持、直观的 Prompt 编排界面、高质量的 RAG 引擎、稳健的 Agent 框架、灵活的流程编排，并同时提供了一套易用的界面和 API。这为开发者节省了许多重复造轮子的时间，使其可以专注在创新和业务需求上。

#### 为什么使用 Dify？

- LangChain 这类的开发库（Library）想象为有着锤子、钉子的工具箱。与之相比，Dify 好比是一套脚手架，更接近生产需要的完整方案。
- Dify 是**开源**的，由一个专业的全职团队和社区共同打造。在灵活和安全的基础上，同时保持对数据的完全控制。
- 产品简单、克制、迭代迅速。

#### Dify 能做什么？

- **创业**，快速的将你的 AI 应用创意变成现实，无论成功和失败都需要加速。在真实世界，已经有几十个团队通过 Dify 构建 MVP（最小可用产品）获得投资，或通过 POC（概念验证）赢得了客户的订单。
- **将 LLM 集成至已有业务**，通过引入 LLM 增强现有应用的能力，接入 Dify 的 RESTful API 从而实现 Prompt 与业务代码的解耦，在 Dify 的管理界面是跟踪数据、成本和用量，持续改进应用效果。
- **作为企业级 LLM 基础设施**，一些银行和大型互联网公司正在将 Dify 部署为企业内的 LLM 网关，加速 GenAI 技术在企业内的推广，并实现中心化的监管。
- **探索 LLM 的能力边界**，即使你是一个技术爱好者，通过 Dify 也可以轻松的实践 Prompt 工程和 Agent 技术。

#### 下一步行动

- 阅读[**快速开始**](https://docs.dify.ai/zh-hans/guides/application-orchestrate/creating-an-application)，速览 Dify 的应用构建流程
- 了解如何[**自部署 Dify 到服务器**](https://docs.dify.ai/zh-hans/getting-started/install-self-hosted)上，并[**接入开源模型**](https://docs.dify.ai/zh-hans/guides/model-configuration)
- 了解 Dify 的[**特性规格**](https://docs.dify.ai/zh-hans/getting-started/readme/features-and-specifications)和 **Roadmap**
- 在 [**GitHub**](https://github.com/langgenius/dify) 上为我们点亮一颗星，并阅读我们的**贡献指南**

#### Dify实践

你可以通过 3 种方式在 Dify 的工作室内创建应用：

- 基于应用模板创建（新手推荐）

- 创建一个空白应用

- 通过 DSL 文件（本地/在线）创建应用

  > [!NOTE]
  >
  > 1. Dify DSL 是由 Dify.AI 所定义的 AI 应用工程文件标准，文件格式为 YAML。该标准涵盖应用在 Dify 内的基本描述、模型参数、编排配置等信息。
  > 2. 导入 DSL 文件时将校对文件版本号。如果 DSL 版本号差异较大，有可能会出现兼容性问题。

  > YAML 曾被称为“Yet Another Markup Language”（又一个标记语言），但后来为了更好地区分其作为数据导向的目的，而重新解释为YAML Ain't Markup Language（回文缩略词）。意味着 YAML 的设计初衷是处理数据，而不是用于文档标记。

##### 聊天助手

对话型应用采用一问一答模式与用户持续对话。

对话型应用的编排支持：对话前提示词，变量，上下文，开场白和下一步问题建议。

>  [!TIP]
>
> 应用工具箱：
>
> - 对话开场白
> - 下一步问题建议
> - 文字转语音
> - 语音转文字
> - 引用与归属
> - 内容审查
> - 标注回复

##### 多模型调试

你可以同时批量检视不同模型对于相同问题的回答效果。

##### 发布应用

- 发布为公开Web站点

- 嵌入网站

- 基于APIs开发

  > 后端即服务

- 基于前端组件再开发

  > **[WebApp Template](https://docs.dify.ai/zh-hans/guides/application-publishing/based-on-frontend-templates)**，每种类型应用的 WebApp 开发脚手架

##### 智能体

<font color="brown">**Agent** 定义</font>

智能助手（Agent Assistant），利用大语言模型的推理能力，能够自主对复杂的人类任务进行目标规划、任务拆解、工具调用、过程迭代，并在没有人类干预的情况下完成任务。

#### 工作流

工作流通过将复杂的任务分解成较小的步骤（节点）降低系统复杂度，减少了对提示词技术和模型推理能力的依赖，提高了 LLM 应用面向复杂任务的性能，提升了系统的可解释性、稳定性和容错性。

Dify 工作流分为两种类型：

- **Chatflow**：面向对话类情景，包括客户服务、语义搜索、以及其他需要在构建响应时进行多步逻辑的对话式应用程序。
- **Workflow**：面向自动化和批处理情景，适合高质量翻译、数据分析、内容生成、电子邮件自动化等应用程序。

##### 如何开始

- 从一个空白的工作流开始构建或者使用系统模板帮助你开始；
- 熟悉基础操作，包括在画布上创建节点、连接和配置节点、调试工作流、查看运行历史等；
- 保存并发布一个工作流；
- 在已发布应用中运行或者通过 API 调用工作流；

##### 关键概念

- 节点：**节点是工作流的关键构成**，通过连接不同功能的节点，执行工作流的一系列操作。
- 变量：**变量用于串联工作流内前后节点的输入与输出**，实现流程中的复杂处理逻辑，包含系统变量、环境变量和会话变量。[link](https://docs.dify.ai/zh-hans/guides/workflow/variables) 

### Dify实操

Dify 是一个开源的 LLM 应用开发平台。其直观的界面结合了 AI 工作流、RAG 管道、Agent、模型管理、可观测性功能等，让您可以快速从原型到生产。

#### 启动Dify

`cd dify/docker`路径下：

启动服务  `docker compose up -d `

查看容器运行状况  `docker compose ps`

关闭服务  `docker compose down`

更新dify

```shell
cd dify/docker
docker compose down
git pull origin main
docker compose pull
docker compose up -d
```

#### 访问 Dify

你可以先前往管理员初始化页面设置设置管理员账户：

```shell
# 本地环境
http://localhost/install

# 服务器环境
http://your_server_ip/install
```

Dify 主页面：

```shell
# 本地环境
http://localhost

# 服务器环境
http://your_server_ip
```

#### 自定义配置

编辑 `.env` 文件中的环境变量值。然后重新启动 Dify：

```shell
docker compose down
docker compose up -d
```

完整的环境变量集合可以在 `docker/.env.example` 中找到。



## Swarm to OpenAI-Agents

OpenAI Agents SDK 是一个轻量级但功能强大的框架，用于构建多智能体工作流。

> OpenAI Agents SDK 是我们之前针对智能体的实验项目 Swarm 的生产级升级版。

核心概念：

- **智能体**：配置了指令、工具、安全护栏（guardrails）和交接（handoffs）的大语言模型（LLMs）。
- **交接**：Agents SDK 中用于在智能体之间转移控制的专用工具调用。
- **安全护栏**：可配置的安全检查，用于输入和输出的验证。
- **追踪**：内置的智能体运行跟踪，允许您查看、调试和优化工作流。

### 关于

为什么用Agents SDK

该 SDK 具有两个主要设计原则：

- 功能足够丰富，值得使用，但基本构件少，便于快速学习。
- 开箱即用效果很好，但您可以完全自定义具体的操作。

SDK 的主要功能

- **智能体循环**：内置的智能体循环，处理工具调用、将结果发送到语言模型（LLM），并重复执行直到 LLM 完成。
- **以 Python 为中心**：利用内置的语言特性来协调和链接智能体，而不需要学习新的抽象概念。
- **交接**：强大的功能，用于在多个智能体之间协调和委派（delegate）任务。
- **保护措施**：并行运行输入验证和检查，如果检查失败则提前中断。
- **函数工具**：将任何 Python 函数转换为工具，支持自动生成模式和基于 Pydantic 的验证。
- **追踪**：内置的追踪功能，让您可视化、调试和监控工作流，同时使用 OpenAI 的评估、微调和提炼工具套件。

安装

`pip install openai-agents`

Hello world示例

`export OPENAI_API_KEY=sk-...`

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

（*如果运行此程序，请确保设置 `OPENAI_API_KEY` 环境变量，[参考](https://platform.openai.com/api-keys)*）

### 快速开始

```python
from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import asyncio

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# 安全护栏智能体
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

# 数学学科-智能体
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

# 历史学科-智能体
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


# 安全护栏
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )


# 预诊交接
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    # 交接是智能体可以委派的子智能体。
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

# 主函数，先调用预诊交接智能体
async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?")
    print(result.final_output)

    result = await Runner.run(triage_agent, "what is life")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

> [!NOTE]
>
> Agent(name='Math Tutor', instructions='You provide help with math problems. Explain your reasoning at each step and include examples', handoff_description=None, handoffs=[], model=None, model_settings=ModelSettings(temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, tool_choice=None, parallel_tool_calls=None, truncation=None, max_tokens=None, reasoning=None, metadata=None, store=None), tools=[], mcp_servers=[], mcp_config={}, input_guardrails=[], output_guardrails=[], output_type=None, hooks=None, tool_use_behavior='run_llm_again', reset_tool_choice=True)

```python
class Agent(Generic[TContext]):
    """智能体是一个配置了指令、工具、安全护栏、交接等的 AI 模型。

    我们强烈建议传递 `instructions`，它是智能体的“系统提示（system prompt）”。此外，你可以传递 `handoff_description`，这是一个人类可读的智能体描述，在智能体被用于工具/交接时使用。

    智能体是针对上下文类型的泛型。上下文是你创建的一个（可变）对象。这个上下文会被传递给工具函数、交接、安全护栏等。
    """

    name: str
    """智能体的名称。"""
    #  以下是Python的类型注解（type hinting）语法
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None
    """智能体的指令。当调用此智能体时将作为“系统提示”使用。描述智能体应该做什么，以及它如何回应。

    可以是一个字符串，或者是一个动态生成智能体指令的函数。如果提供函数，它将与上下文和智能体实例一起调用。必须返回一个字符串。
    """

    handoff_description: str | None = None
    """智能体的描述。当智能体作为交接使用时使用，以便 LLM 知道它的功能以及何时调用它。
    """

    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)
    """交接是智能体可以委派的子智能体。你可以提供交接的列表，智能体可以选择在相关时委派它们。允许任务拆解和模块化。
    """

    model: str | Model | None = None
    """调用 LLM 时使用的模型实现。

    默认情况下，如果未设置，智能体将使用 `model_settings.DEFAULT_MODEL` 中配置的默认模型。
    """

    model_settings: ModelSettings = field(default_factory=ModelSettings)
    """配置模型特定的调优参数（例如温度、top_p）。
    """

    tools: list[Tool] = field(default_factory=list)
    """智能体可以使用的工具列表。"""

    mcp_servers: list[MCPServer] = field(default_factory=list)
    """智能体可以使用的 [模型上下文协议] 服务器列表。每次智能体运行时，它将从这些服务器中包含可用工具的列表。

    注意：你需要管理这些服务器的生命周期。具体来说，必须在将其传递给智能体之前调用 `server.connect()`，并在服务器不再需要时调用 `server.cleanup()`。
    """

    mcp_config: MCPConfig = field(default_factory=lambda: MCPConfig())
    """MCP 服务器的配置。"""

    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    """在生成响应之前，智能体执行期间并行运行的检查列表。如果智能体是链中的第一个智能体，则运行。
    """

    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    """在生成响应后，对智能体的最终输出进行检查的列表。仅在智能体生成最终输出时运行。
    """

    output_type: type[Any] | None = None
    """输出对象的类型。如果未提供，输出将为 `str`。"""

    hooks: AgentHooks[TContext] | None = None
    """一个接收各种生命周期事件回调的类，用于此智能体。
    """

    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction
    ) = "run_llm_again"
    """这让你配置工具使用的处理方式。
    - "run_llm_again"：默认行为。工具被运行，然后 LLM 收到结果并进行回应。
    - "stop_on_first_tool"：第一个工具调用的输出被用作最终输出。这意味着 LLM 不处理工具调用的结果。
    - 工具名称的列表：如果调用列表中的任何工具，智能体将停止运行。最终输出将是第一个匹配工具调用的输出。LLM 不处理工具调用的结果。
    - 函数：如果传递一个函数，它将与运行上下文和工具结果列表一起调用。它必须返回一个 `ToolToFinalOutputResult`，用于确定工具调用是否产生最终输出。

      注意：此配置特定于 FunctionTools。托管工具，如文件搜索、网络搜索等，总是由 LLM 处理。
    """

    reset_tool_choice: bool = True
    """在调用工具后是否将工具选择重置为默认值。默认为 True。这确保智能体不会进入工具使用的无限循环。"""
```

追踪

要查看智能体运行期间发生的情况，请在 OpenAI 控制面板中导航到[追踪查看器](https://platform.openai.com/traces)，以查看代理运行的追踪记录。

下一步

了解如何构建更复杂的智能体流程：

- 了解如何配置智能体。
- 了解如何运行智能体。
- 了解工具、安全护栏和模型。

### 文档

#### 智能体

智能体是你应用中的核心构件。智能体是一个大型语言模型（LLM），经过配置，包含指令和工具。

##### 1.泛型

 `Agent` 类，使用了泛型。下面是对这段代码的逐部分解读：

**`class Agent(Generic[TContext]):`**

- `Agent` 是类的名称。
- `Generic[TContext]` 表示 `Agent` 是一个泛型类，其中 `TContext` 是一个类型变量。这个类型变量可以在类的实例化时指定具体类型。

泛型的作用

通过使用 `TContext`，`Agent` 类可以与不同类型的上下文（context）配合使用。上下文的类型是泛型的，可以根据需要进行定义。

示例：

```python
@dataclass
class UserContext:
  uid: str
  is_pro_user: bool

  async def fetch_purchases() -> list[Purchase]:
     return ...

agent = Agent[UserContext](
    ...,
)
```

上下文被用作依赖注入工具。这是指上下文可以用于传递所需的依赖项（如配置、服务等）给智能体。

> Agents are generic on their `context` type. Context is a dependency-injection tool: it's an object you create and pass to `Runner.run()`, that is passed to every agent, tool, handoff etc, and it serves as a grab bag of dependencies and state for the agent run. You can provide any Python object as the context.

##### 2.输出类型

默认情况下，智能体生成纯文本（即字符串）输出。如果你希望智能体生成特定类型的输出，可以使用 `output_type` 参数。常见的选择是使用 Pydantic 对象。

> 我们支持任何可以被 Pydantic 的 TypeAdapter 包装的类型，如数据类、列表、TypedDict 等。

```python
from pydantic import BaseModel
from agents import Agent

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
```

> [!Note]
>
> 当你传递 `output_type` 时，这告诉模型使用 [结构化输出](https://platform.openai.com/docs/guides/structured-outputs) 而不是常规的纯文本响应。

##### 3.交接

交接（Handoffs）是智能体可以委托的子智能体。你可以提供一个交接列表，智能体可以在相关时选择委托给它们。这是一种强大的模式，可以协调模块化的、专门化的智能体，使其在单一任务上表现出色。

##### 4.动态指令

 在大多数情况下，你可以在创建智能体时提供指令。不过，你也可以通过函数提供动态指令。该函数将接收智能体和上下文，并必须返回提示。使用常规函数和异步函数都被允许。

```python
def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)
```

##### 5.生命周期事件（钩子）

 有时，你可能想观察智能体的生命周期。例如，你可能希望在某些事件发生时记录事件或预取数据。你可以通过 `hooks` 属性挂钩到智能体生命周期。通过子类化 `AgentHooks` 类，并重写你感兴趣的方法。

##### 6.护栏

 护栏（Guardrails）允许你在智能体运行的同时，对用户输入进行检查和验证。例如，你可以筛选用户的输入以判断其相关性。

##### 7.复制智能体

 通过在智能体上使用 `clone()` 方法，你可以复制一个智能体，并可以选择更改任何属性。

```python
# 海盗
pirate_agent = Agent(
    name="Pirate",
    instructions="Write like a pirate",
    model="o3-mini",
)
# 机器人
robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot",
)
```

##### 8.强制使用工具

 提供工具列表并不总意味着 LLM 会使用某个工具。你可以通过设置 `ModelSettings.tool_choice` 来强制使用工具。有效值包括：

- **auto**：允许 LLM 决定是否使用工具。
- **required**：要求 LLM 使用工具（但可以智能地决定使用哪个工具）。
- **none**：要求 LLM 不使用任何工具。
- 设置特定字符串，例如 `my_tool`，要求 LLM 使用该特定工具。

> [!NOTE]
>
> 为了防止无限循环，框架在每次工具调用后会自动将 `tool_choice` 重置为 "auto"。此行为可以通过 `agent.reset_tool_choice` 进行配置。无限循环的原因是工具结果会被发送给 LLM，随后 LLM 可能会生成另一个工具调用，从而导致无限循环。
>
> 如果你希望智能体在工具调用后完全停止（而不是继续使用自动模式），可以设置 `Agent.tool_use_behavior="stop_on_first_tool"`，这将直接使用工具输出作为最终响应，而不进行进一步的 LLM 处理。



#### 运行智能体

你可以通过 `Runner` 类运行智能体，有三种选项：

1. **`Runner.run()`**：异步运行并返回 `RunResult`。
2. **`Runner.run_sync()`**：同步方法，实质上运行 `run()`。
3. **`Runner.run_streamed()`**：异步运行并返回 `RunResultStreaming`。它以流式模式调用 LLM，并在接收事件时将其流式传输给你。

##### 1.智能体循环

当你在 `Runner` 中使用 `run` 方法时，你需要传入一个起始智能体和输入。输入可以是一个字符串（视为用户消息），也可以是一个输入项的列表。

`Runner` 会运行一个循环：

1. 调用当前智能体的 LLM，使用当前输入。
2. LLM 生成输出。
   1. 如果 LLM 返回 `final_output`，循环结束，我们返回结果。
   2. 如果 LLM 进行移交（handoff），我们更新当前智能体和输入，并重新运行循环。
   3. 如果 LLM 产生工具调用，我们执行这些工具调用，附加结果，并重新运行循环。
3. 如果超过传入的 `max_turns`，则引发 `MaxTurnsExceeded` 异常。

>  [!Note]
>
> 判断 LLM 输出是否被视为`final_output`的规则是：它生成了符合所需类型的文本输出，且没有工具调用。

##### 2.流式

##### 3.运行配置

`run_config` 参数让你可以配置一些全局设置以供智能体运行使用：

- **model**：允许设置全局 LLM 模型，与每个智能体的模型无关。

- **model_provider**：用于查找模型名称的模型提供者，默认为 OpenAI。

- **model_settings**：覆盖特定于智能体的设置。例如，可以设置全局的温度或 `top_p`。

- **input_guardrails, output_guardrails**：在所有运行中包含的输入或输出护栏。

- **handoff_input_filter**：应用于所有移交的全局输入过滤器（如果移交中尚未有过滤器）。输入过滤器允许你编辑发送给新智能体的输入。有关更多详细信息，请参见 `Handoff.input_filter` 的文档。

- **tracing_disabled**：允许禁用整个运行的追踪。

- **trace_include_sensitive_data**：配置追踪是否会包括潜在的敏感数据，例如 LLM 和工具调用的输入/输出。

- **workflow_name, trace_id, group_id**：设置运行的追踪工作流名称、追踪 ID 和追踪组 ID。我们建议至少设置 `workflow_name`。

  组 ID 是一个可选字段，允许你在多个运行之间链接追踪。

- **trace_metadata**：要包含在所有追踪中的元数据。

