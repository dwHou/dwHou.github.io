# Claude Code CLI 使用笔记

> 适用场景：在终端中使用 Claude Code 阅读、修改、测试和审查项目代码。

---

## 1. 启动 Claude Code

### 1.1 进入项目并启动交互模式（最常用）

```bash
cd /code/my-project
claude
```

启动后进入交互式会话，可以持续与 Claude 对话、让它读代码、修改文件、运行命令等。

也可以在启动时直接给出第一条任务：

```bash
claude "帮我快速理解这个项目的目录结构和入口文件"
```

---

### 1.2 单次命令模式：执行后退出

适合脚本、CI，或只想快速问一个问题。

```bash
claude -p "帮我解释一下 train.py 里的 forward 函数"
```

`-p` 是 `--print`，表示非交互模式：输出结果后自动退出。

还可以把命令输出通过管道交给 Claude：

```bash
git diff | claude -p "审查这段改动，指出潜在 Bug"
```

> 注意：使用 `claude -p` 创建的会话通常不会显示在交互式会话选择器中；如需恢复，应保存并使用它的 session ID。

---

## 2. 退出、取消与恢复会话

### 2.1 退出 Claude Code

在交互界面中：

```text
Ctrl+D
```

或输入：

```text
exit
```

```text
/exit
```

这些操作会退出当前 Claude Code 会话；会话记录会保存在本地，之后可以恢复。

---

### 2.2 取消当前输入或停止 Claude

```text
Ctrl+C
```

- 如果正在输入：取消当前输入。
- 如果 Claude 正在生成回复或执行操作：中断当前任务。
- 已完成的操作和会话历史通常仍会保留。

---

### 2.3 恢复最近一次会话：`claude -c`

```bash
cd /code/my-project
claude -c
```

`-c` 等同于：

```bash
claude --continue
```

作用：

- 自动恢复**当前目录**最近一次 Claude Code 会话；
- 不会弹出会话列表；
- 适合刚退出终端、想立刻接着上次任务继续时使用。

---

### 2.4 选择并恢复历史会话：`claude -r`

```bash
cd /code/my-project
claude -r
```

`-r` 等同于：

```bash
claude --resume
```

作用：

- 打开**会话选择器**；
- 可以从历史会话中选择一个继续；
- 适合当前项目里有多个任务、多个会话时使用。

会话选择器常用快捷键：

| 快捷键    | 作用                                  |
| --------- | ------------------------------------- |
| `↑` / `↓` | 选择上一个 / 下一个会话               |
| `Enter`   | 恢复当前选中的会话                    |
| `Space`   | 预览会话内容                          |
| `/`       | 搜索会话                              |
| `Ctrl+R`  | 重命名选中的会话                      |
| `Ctrl+A`  | 显示本机所有项目的会话                |
| `Ctrl+W`  | 显示当前 Git 仓库所有 worktree 的会话 |
| `Ctrl+B`  | 按当前 Git 分支筛选会话               |
| `Esc`     | 退出会话选择器                        |

---

### 2.5 按名称或 Session ID 恢复指定会话

如果你已经给会话命名：

```bash
claude -r auth-refactor
```

如果你有 session ID：

```bash
claude -r "abc123"
```

恢复会话后，也可以立刻附带新任务：

```bash
claude -r auth-refactor "继续实现登录接口的单元测试"
```

---

### 2.6 给会话命名（推荐）

给会话取一个容易搜索的名称，方便之后恢复。

启动时命名：

```bash
claude -n auth-refactor
```

在交互会话中重命名：

```text
/rename auth-refactor
```

之后可以直接恢复：

```bash
claude -r auth-refactor
```

> 建议命名格式：`功能-任务`，例如：
>
> - `auth-refactor`
> - `fix-payment-timeout`
> - `review-pr-128`
> - `train-model-debug`

---

## 3. 交互模式中的 3 个常用前缀

在 `Claude>` 提示符下，以下前缀非常常用。

| 前缀 | 用途                                         | 示例                   |
| ---- | -------------------------------------------- | ---------------------- |
| `@`  | 提及项目文件，触发路径补全                   | `请解释 @src/train.py` |
| `/`  | 执行 Claude Code 内置命令或 Skill            | `/compact`             |
| `!`  | 直接运行本地终端命令，并将输出加入对话上下文 | `! git status`         |

---

### 3.1 `@`：提及文件

输入 `@` 可以搜索和补全项目中的文件路径。

示例：

```text
解释一下 @src/train.py 中的 forward 函数
```

```text
比较 @src/config.py 和 @configs/default.yaml 中的配置是否一致
```

适用场景：

- 希望 Claude 聚焦某个文件；
- 同时引用多个文件；
- 避免手动复制大段代码到对话中。

---

### 3.2 `/`：斜杠命令

输入 `/` 可以查看当前可用的内置命令、Skills、插件命令和 MCP 命令。

```text
/
```

也可以直接输入命令：

```text
/compact
```

---

### 3.3 `!`：直接执行终端命令

在输入内容前加 `!`，可以不退出 Claude Code 直接运行 Shell 命令。

```text
! git status
```

```text
! npm test
```

```text
! pytest -q
```

```text
! git diff
```

特点：

- 命令直接在本地终端执行；
- 输出会自动进入当前对话上下文；
- Claude 可以根据测试日志、Git diff、报错信息继续分析；
- 不需要先让 Claude “理解并批准”这条命令。

长时间运行的命令可以按：

```text
Ctrl+B
```

将其转入后台执行。

---

## 4. 最常用斜杠命令

### 4.1 `/compact`：压缩上下文

```text
/compact
```

作用：

- 将长对话压缩为关键信息摘要；
- 释放上下文空间；
- 保留当前任务、关键决策、已修改文件、待办事项等重要内容。

适用场景：

- 会话很长；
- Claude 开始遗忘早期上下文；
- 准备进入任务的下一阶段；
- 希望减少历史对话对后续任务的干扰。

> `/compact` 是“保留任务脉络的压缩”，不是彻底清空会话。

---

### 4.2 `/clear`：开始一个全新对话

```text
/clear
```

作用：

- 创建一个上下文为空的新会话；
- 不再携带当前对话历史；
- 项目级记忆（例如 `CLAUDE.md` 中的规则）仍会保留；
- 旧会话不会消失，之后仍可通过 `/resume` 或 `claude -r` 恢复。

适用场景：

- 当前 Bug 已解决，准备做完全无关的新任务；
- 当前上下文太乱；
- 不希望旧任务、旧假设影响新任务。

也可以在清空时顺便给旧会话命名：

```text
/clear fix-payment-timeout
```

---

### 4.3 `/resume`：在交互界面中切换会话

```text
/resume
```

作用：

- 在**当前 Claude Code 交互界面内**打开会话选择器；
- 可以切换到其他历史会话；
- 不需要先退出，再执行 `claude -r`。

如果会话已命名，也可以直接恢复：

```text
/resume auth-refactor
```

---

### 4.4 `/rename`：重命名当前会话

```text
/rename fix-payment-timeout
```

作用：

- 为当前会话设置一个易读名称；
- 方便后续在 `claude -r` 或 `/resume` 中搜索和恢复。

---

### 4.5 `/diff`：查看代码改动

```text
/diff
```

作用：

- 打开交互式 diff 查看器；
- 查看当前工作区未提交的 Git 改动；
- 也可查看 Claude 在各轮对话中造成的改动。

适用场景：

- 提交代码前检查；
- 让 Claude 修改文件后人工确认；
- 排查“它到底改了什么”。

常见工作流：

```text
帮我修复登录接口的超时问题，并运行相关测试
```

```text
/diff
```

```text
! git diff
```

---

### 4.6 `/code-review`：审查当前改动

```text
/code-review
```

作用：

- 审查当前 Git diff；
- 重点检查正确性问题、潜在 Bug 和可清理项；
- 可用于 Claude 已修改代码后的复核。

示例：

```text
/code-review
```

更深入地审查：

```text
/code-review high
```

让 Claude 尝试直接修复发现的问题：

```text
/code-review --fix
```

> `/code-review` 主要审查当前改动，而不是无差别扫描整个项目。  
> 若要了解项目整体结构、架构问题或潜在风险，可以直接提问，例如：  
> `请全面审查这个项目的架构、模块边界和潜在稳定性问题。`

---

### 4.7 `/context`：查看上下文占用情况

```text
/context
```

作用：

- 查看当前上下文主要被哪些内容占用；
- 判断是否需要执行 `/compact`；
- 适合长时间连续开发时使用。

---

### 4.8 `/rewind`：回退对话或代码

```text
/rewind
```

作用：

- 回到此前某个检查点；
- 可以选择回退对话、代码，或两者都回退；
- 适合 Claude 沿着错误方向改了很多内容时使用。

快捷操作：

```text
Esc Esc
```

---

### 4.9 `/help`：查看帮助

```text
/help
```

作用：

- 查看可用命令和帮助信息；
- 当不确定当前版本是否支持某个命令时，优先使用它。

---

## 5. 推荐日常工作流

### 场景 A：继续昨天没做完的任务

```bash
cd /code/my-project
claude -c
```

如果不是最近的那个任务：

```bash
claude -r
```

在列表中选择对应会话。

---

### 场景 B：开始一个新功能

```bash
cd /code/my-project
claude -n add-user-profile
```

然后告诉 Claude：

```text
先阅读项目结构和现有用户模块，不要修改代码。
完成后给出实现用户资料页的方案、涉及的文件和风险点。
```

确认方案后再执行：

```text
按照方案实现。每完成一个步骤都运行相关测试。
```

完成后检查：

```text
/diff
```

```text
/code-review
```

---

### 场景 C：快速定位测试失败

```bash
cd /code/my-project
claude
```

```text
! pytest -q
```

然后：

```text
根据上面的报错定位根因。先解释问题和修复方案，等我确认后再改代码。
```

---

### 场景 D：修改前先规划，避免直接乱改

```text
先不要修改文件。请分析实现这个需求涉及哪些模块，
给出分步骤计划、风险点和测试方案。
```

确认计划后：

```text
按计划开始实现。修改前说明要改哪些文件，完成后运行测试。
```

---

## 6. 一页速查表

| 目标                 | 命令 / 快捷键               |
| -------------------- | --------------------------- |
| 启动交互会话         | `claude`                    |
| 启动并发送第一条任务 | `claude "任务描述"`         |
| 单次执行后退出       | `claude -p "任务描述"`      |
| 恢复当前目录最近会话 | `claude -c`                 |
| 打开历史会话选择器   | `claude -r`                 |
| 按名称恢复会话       | `claude -r <name>`          |
| 按 session ID 恢复   | `claude -r <session-id>`    |
| 启动时命名会话       | `claude -n <name>`          |
| 在会话中命名         | `/rename <name>`            |
| 在会话中切换历史会话 | `/resume`                   |
| 退出会话             | `Ctrl+D` / `exit` / `/exit` |
| 取消当前生成或输入   | `Ctrl+C`                    |
| 提及文件             | `@文件路径`                 |
| 执行终端命令         | `! 命令`                    |
| 压缩上下文           | `/compact`                  |
| 开始全新对话         | `/clear`                    |
| 查看代码改动         | `/diff`                     |
| 审查当前改动         | `/code-review`              |
| 查看上下文占用       | `/context`                  |
| 回退对话或代码       | `/rewind` 或 `Esc Esc`      |
| 查看命令帮助         | `/help`                     |

---

## 7. 实用习惯

1. **一个大任务对应一个会话，并及时命名。**
   ```text
   /rename fix-payment-timeout
   ```

2. **修改代码前先让 Claude 给计划。**
   ```text
   先不要改代码。请先分析问题、给出修改计划和测试方案。
   ```

3. **修改后固定检查三件事。**
   ```text
   /diff
   ! git status
   ! pytest -q
   ```

4. **长会话定期压缩上下文。**
   ```text
   /compact
   ```

5. **任务完全切换时用 `/clear`，而不是硬接着聊。**
   ```text
   /clear
   ```

6. **优先通过 `@文件`、`!测试命令` 提供精确上下文。**
   ```text
   请检查 @src/api/auth.py 的异常处理是否正确
   ```

   ```text
   ! npm test
   ```

---

## 8. 最小高效模板

```bash
cd /code/my-project
claude -n task-name
```

```text
先阅读相关代码，不要修改文件。
请说明：
1. 当前实现逻辑；
2. 问题或需求涉及的文件；
3. 推荐修改方案；
4. 需要补充或执行的测试。
```

确认后：

```text
按方案实现。修改完成后运行相关测试，并总结改动。
```

最后：

```text
/diff
```

```text
/code-review
```