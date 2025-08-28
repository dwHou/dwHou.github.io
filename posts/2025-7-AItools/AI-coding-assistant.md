# Cursor

## 安全相关

**敏感代码库的定义**
 如果满足以下任一条件，代码库将被归类为敏感代码库：

- 被明确标记为“受限”；或

- 包含已超出服务等级协议（SLA）范围的**最高**或**高严重等级**的安全漏洞。

  > 如果一个非敏感代码库中被新发现了**最高**或**高严重等级**的安全漏洞，且未在服务等级协议（SLA）规定的时间内修复，则该代码库将被重新归类为敏感代码库。
  >  一旦代码库状态变为敏感，开发人员必须**立即更新其环境或配置**，以确保敏感代码库中的相关代码不会被暴露。

使用经过认可的方法（例如 Cursor 使用的 `.cursorignore` 文件），明确防止敏感文件或文件夹被暴露给云端 AI 工具。

使用 `.cursorignore` 的说明：

1. 仓库的所有者给所有分支更新`.cursorignore` 文件。

2. 将 `.cursorignore` 添加到 `CODEOWNERS` 文件中，以确保其变更由安全负责人进行审查。

   > 在 GitLab（以及 GitHub）中，`CODEOWNERS` 是一个特殊文件，用于**指定某些文件或目录的负责人（代码所有者）**。当这些文件发生变更时，指定的人员会自动被要求进行 **代码审核（review）**。

3. 仓库的开发者们及时拉取更新`.cursorignore` 后的代码。

   重启 Cursor IDE，以重新加载上下文并应用 `.cursorignore` 的更改。
    记住清理本地 Cursor 缓存，以防止之前已暴露的文件被重复使用：

   - macOS: `~/Library/Application Support/Cursor`

   - Windows: `%APPDATA%\Cursor`

   - Linux: `~/.config/Cursor`

参考：

- [Cursor Ignore Files](https://docs.cursor.com/context/ignore-files)
- [GitLab CODEOWNERS Reference](https://docs.gitlab.com/user/project/codeowners/reference/)

## 使用之道

1. `/Generate Cursor Rules `  命令 +  一些项目架构的关键词，先生成规则文档。会作为该项目的持久的上下文。
2. 提示词：`请先列出修改计划，等我批准后再写代码。` 避免先斩后奏。
3. 提问技巧：提供项目背景，@项目代码，描述具体需求

## 使用之术

Do’s ✅

- 将AI作为强大的编程助手，而非自己的替代品
- 保持对关键技术细节的理解和掌控
- 利用AI加速学习新技术，但要消化吸收
- 定期整理和总结AI辅助的工作成果

Don’ts ❌

- 不要完全依赖AI自主完成复杂任务
- 不要直接使用未经理解的AI代码
- 不要忽视团队协作中的技术同步
- 不要让AI的复杂实现影响代码可维护性

AI使用陷阱：

- 自动化偏见（Automation bias）：当使用AI取得一定成功后，开始过度信任AI
- 沉没成本（Sunk cost fallacy）：迷恋多行自动补全，文件自动生成，舍不得删除或者修改
- 锚定偏见（Anchoring bias）：一旦看过AI的建议，很难再想到别的解决方案
- 审查疲劳（Security vigilance）：结合第一点，大量生产的代码很难沉下心来review，容易引入一些似是而非的依赖和代码

