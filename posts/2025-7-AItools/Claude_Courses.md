# Anthropic Skilljar 课程笔记

> 课程平台：[Anthropic Skilljar](https://anthropic.skilljar.com/)
> 笔记创建时间：2026-01-21

---

## 目录

- [技术开发类课程](#技术开发类课程)
  - [Claude 101](#claude-101)
  - [Claude with Google Cloud's Vertex AI](#claude-with-google-clouds-vertex-ai)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
- [AI 素养类课程](#ai-素养类课程)
  - [AI Fluency for Students](#ai-fluency-for-students)
  - [AI Fluency for Educators](#ai-fluency-for-educators)
  - [Teaching AI Fluency](#teaching-ai-fluency)
  - [AI Fluency for Nonprofits](#ai-fluency-for-nonprofits)

---

## 技术开发类课程

### Claude 101

**课程简介**
全面覆盖 Claude API 的使用，从基础到高级的模型应用开发。

**学习目标**
-
-

**核心内容**
-

**实践项目**
-

**学习笔记**
-

---

### Claude with Google Cloud's Vertex AI

**课程简介**
通过 Google Cloud 的 Vertex AI 平台使用 Anthropic 模型的完整指南。

**学习目标**
-
-

**核心内容**
-

**集成要点**
-

**学习笔记**
-

---

### Model Context Protocol (MCP)

**课程简介**
使用 Python 从零构建 MCP 服务器和客户端，连接 Claude 与外部服务。

**学习目标**
-
-

**核心内容**
- 工具（Tools）
- 资源（Resources）
- 提示词（Prompts）

**实践练习**
-

**学习笔记**
-

---

## AI 素养类课程

### AI Fluency for Students

**课程简介**
帮助学生培养 AI 素养技能，通过负责任的 AI 协作提升学习、职业规划和学术成功。

**适用对象**
- 在校学生
- 学习者

**核心能力**
-
-

**应用场景**
-

**学习笔记**
-

---

### AI Fluency for Educators

**课程简介**
帮助教师、教学设计师和教育领导者将 AI 素养应用于教学实践和机构战略。

**适用对象**
- 教师
- 教学设计师
- 教育管理者

**核心内容**
-
-

**实践策略**
-

**学习笔记**
-

---

### Teaching AI Fluency

**课程简介**
帮助学术教师和教学设计师在讲师主导的环境中教授和评估 AI 素养。

**适用对象**
- 学术教师
- 教学设计师

**教学方法**
-
-

**评估框架**
-

**课程设计**
-

**学习笔记**
-

---

### AI Fluency for Nonprofits

**课程简介**
帮助非营利组织专业人士培养 AI 素养，提升组织影响力和效率。

**适用对象**
- 非营利组织工作人员
- 社会组织从业者

**核心内容**
-
-

**应用场景**
-

**学习笔记**
-

---

## 学习进度跟踪

| 课程名称 | 开始日期 | 完成日期 | 证书获得 | 状态 |
|---------|---------|---------|---------|------|
| Claude 101 | | | ☐ | 未开始 |
| Claude with Vertex AI | | | ☐ | 未开始 |
| MCP 开发 | | | ☐ | 未开始 |
| AI Fluency for Students | | | ☐ | 未开始 |
| AI Fluency for Educators | | | ☐ | 未开始 |
| Teaching AI Fluency | | | ☐ | 未开始 |
| AI Fluency for Nonprofits | | | ☐ | 未开始 |

---

## 资源链接

- [Anthropic Skilljar 主页](https://anthropic.skilljar.com/)
- [Anthropic 官方学习资源](https://www.anthropic.com/learn)
- [Anthropic Courses GitHub](https://github.com/anthropics/courses)
- [Claude Resources](https://platform.claude.com/docs/en/resources/overview)
- [Recommended Claude Plugin](https://github.com/feiskyer/claude-code-settings/tree/main)
- [Open-source Claude Code](https://github.com/code-yeongyu/oh-my-opencode)

---

## 备注

- 所有课程完成后均可获得证书
- 访问课程只需 Skilljar 账号，无需 Anthropic 账号

## 其他主题

### Claude Code 成本管理

#### 成本驱动因素：哪些操作在消耗费用？
*   **粘贴大文件**：直接向对话框粘贴大量代码。
*   **冗长或多轮提示词**：复杂的指令或频繁的往返对话。
*   **重复运行大型查询**：对相同的大规模问题进行多次询问。
*   **分析的代码库规模**：被扫描或索引的代码总量。
*   **查询的复杂度**：需要深度推理的任务。
*   **涉及的文件数量**：搜索或修改的文件越多，消耗越高。
*   **对话历史长度**：上下文越长，每次请求携带的 Token 越多。
*   **对话压缩频率**：后台处理（如 Haiku 模型生成的摘要、对话压缩）也会产生费用。

---

#### 追踪您的成本
我们已请求 Claude Code 团队在个人用户层面开放分析仪表盘访问权限。在此期间，请通过以下方式追踪使用情况：
*   **实时查看**：在会话中输入 `/cost`。
*   **估算使用量**：运行 `npx ccusage`。
    *   **注意**：`npx ccusage` 存在“低报”现象，因为它无法计算隐藏的上下文、代理步骤、工具调用、重试机制或 Anthropic 侧的 Token 统计。建议在 `ccusage` 的数值基础上**增加 10%-15%** 作为实际成本参考。

---

#### 控制成本的 14 个实用技巧

1.  **精确限定输入范围**：提示词要具体，避免生成冗长或重复的内容。避免可能消耗大量 Token 的模糊或开放式问题。
2.  **复杂任务用 Claude，简单修改手动做**：将 Claude 用于高价值任务（如重构、算法设计、代码解释）。对于格式化或基本语法修复，请使用轻量级编辑器或 Linter。
3.  **限制 Token 使用量**：通过总结上下文而非粘贴整个文件来减少消耗。避免输入不必要的日志、配置文件或冗长的错误信息。
4.  **策略性复用提示词**：保存并复用针对常见任务（如测试生成、API 集成）的有效模板，减少试错成本。
5.  **优先使用文件附件（如支持）**：尽可能通过附件形式提供代码，而非直接粘贴。
6.  **异步协作**：在团队内分享 Claude 的响应（通过文档或聊天工具），避免多人重复运行相同的查询。
7.  **避免长时交互会话**：将复杂请求拆分为较小的、独立的问题，以控制输出长度和精度。
8.  **对话压缩（Compacting）**：
    *   当上下文超过 95% 容量时，Claude 默认开启自动压缩。
    *   运行 `/config` 并导航至 “Auto-compact enabled” 来切换开关。
    *   当上下文过大时，手动使用 `/compact` 命令。
    *   添加自定义压缩指令，例如：`/compact 重点关注代码示例和 API 用法`。
    *   可以通过修改 `CLAUDE.md` 来自定义压缩逻辑。
9.  **编写具体的查询**：避免触发不必要全盘扫描的模糊请求。
10. **拆解复杂任务**：将大任务拆分为多个专注的小互动。
11. **任务间清除历史**：使用 `/clear` 重置上下文，开启新任务。
12. **利用提示词缓存 (Prompt Caching)**：对于重复的提示词或上下文，缓存可降低高达 90% 的输入 Token 成本。
13. **明确读取指令**：明确告知 Claude 应该阅读哪些文件、忽略哪些文件，特别是在只需修改单个文件时。
14. **避免运行输出过长的命令**：在 Claude Code 中执行产生大量输出的终端命令会显著增加 Token 消耗。如可能，请在外部终端运行。

---

#### 开发者进阶技巧
与 Claude 协作就像与另一位开发者合作。对于同一话题，如果对方不了解背景，你会提供相关上下文；如果切换话题，你会给出提示。同理，LLM 也需要相关的上下文，并且需要知道你何时开始了新话题。

###### 1. 利用 CLAUDE.md 进行记忆管理
如果是单仓 (Monorepo) 项目，可以使用嵌套的 `CLAUDE.md` 配置。这些文件是写给 Claude 看的，让它了解当前目录的作用及处理方式。

*   **1.1 设置记忆文件**：
    *   **全局设置**：放在 `~/.claude/CLAUDE.md`（如代码风格、偏好工具、语言偏好）。
    *   **项目设置**：放在项目根目录的 `CLAUDE.md`（如目录结构、常用命令、架构说明）。
*   **1.2 减少重复**：Claude 每轮会话都会读取这些文件，你无需反复解释项目背景。
*   **1.3 快速添加记忆**：在提示词开头使用 `#` 符号快速添加，例如：`# 使用 Prettier 进行格式化并遵循 Airbnb 风格指南。`
*   **1.4 适用场景**：长期维护同一项目、追求一致的输出格式、希望减少往返对话。
*   **1.5 维护建议**：及时更新记忆，删除过时信息；在重大变更时添加检查点记录；保持内容简练。

> **核心要点**：Memory 功能帮助 Claude 在不同会话间保持一致性，减少 Token 消耗，从而节省开支——在大型或长期项目中效果尤为显著。详细文档见：[Anthropic Memory Docs](https://docs.anthropic.com/en/docs/claude-code/memory)

###### 2. 使用 /clear 开启新聊天
对于不相关的查询，务必运行 `/clear`。如果你发现自己在重复解释某些内容，说明你应该考虑使用系统级/项目级代理，或通过 `CLAUDE.md` 和 `/compact` 进行管理。

###### 3. 与 Cursor 配合使用
Cursor 的商业订阅包含免费模型。很多准备工作、记忆管理和代码库探索可以先用 Cursor 的免费模型完成，然后再让付费模型（或 Claude Code）介入处理核心逻辑。

###### 4. 除非必要，避免使用 Opus 模型
Opus 模型的成本是 Sonnet 的 **5 倍**。在 99% 的情况下，Opus 的表现与 Sonnet 相比并无显著优势。如果你不知道如何利用 Opus 的特性，只是像用 Sonnet 一样喂给它相同的内容，你不仅看不到结果的提升，甚至可能因为 API 调用更昂贵而得到性价比极低的结果。
