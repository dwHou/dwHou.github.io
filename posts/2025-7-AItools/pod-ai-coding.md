1. 安装插件：[Pod Terminal插件使用指南](N/A) （一个内部vscode插件）
   code --install-extension pod-terminal-0.1.6.vsix

2. 容器内安装 npm、Claude

   ```python
   cd /code/npmInstall
   
   # 1. 下载官方 Node.js v22（Claude 官方推荐版本）
   wget https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz
   
   # 2. 解压并重命名为 nodejs 文件夹
   tar -xvf node-v22.11.0-linux-x64.tar.xz
   mv node-v22.11.0-linux-x64 nodejs
   
   # 3. 临时将 node/npm 路径加入当前终端，以便下一步安装使用
   export PATH="/code/npmInstall/nodejs/bin:$PATH"
   
   # 4. 删除无用压缩包
   rm node-v22.11.0-linux-x64.tar.xz
   
   # 5. 指定路径，安装Claude
   npm install -g @anthropic-ai/claude-code --prefix /code/ClaudeInstall
   which claude
   claude --version
   claude doctor
   
   # 6. 安装OpenSpec
   npm install -g @fission-ai/openspec@latest --prefix /code/OpenSpecInstall
   ```

3. 新的K8S容器直接使用：

   ```python
   export PATH="/code/npmInstall/nodejs/bin:/code/ClaudeInstall/bin:/code/OpenSpecInstall/bin:$PATH"
   export DISABLE_AUTOUPDATER=1
   
   node -v
   claude --version
   openspec --version
   # claude doctor
   ```

4. 安装tmux

   配合Pod Terminal，可以很方便的保持Claude对话。

   ```python
   sudo apt update
   sudo apt install -y tmux
   ```

5. Claude CLI

   ```python
   # 启动交互模式（最常用）
   cd /code/my-project
   claude
   
   # 单次命令模式（不进入交互，适合写脚本）
   # 如果你只想问一个问题或者让它改个文件就退出，加上 -p（print）参数：
   claude -p "帮我解释一下 train.py 里的 forward 函数"
   
   # 交互模式常用的3个前缀符号
   # 1. @ 提及文件
   "在对话时，输入 @ 可以快速联想你项目里的文件。"
   # 2. / 斜杠命令
   "用于控制 Claude 自身的系统行为（下方会详解最常用的命令）"
   # 3. ! 执行终端命令
   "如果你想在不退出 Claude 交互界面的情况下运行本地命令，可以直接在前面加 !。"
   
   # 交互模式下最常用的“斜杠命令” (Slash Commands)
   '''
   在 🤖 Claude> 提示符下，输入以下命令能极大提升效率并节省 Token：
   
   1. /compact（上下文压缩）：
   当和 Claude 聊得很久、Token 占用很高（超过 80%）时，它会变慢且变贵。输入 /compact，Claude 会把之前的聊天历史压缩总结，只保留关键信息，瞬间释放 Context 空间。
   
   2. /clear（彻底清空）：
   当你搞定了一个 Bug，准备换个完全不相关的新任务时，输入 /clear。这会彻底清空当前会话历史，避免旧代码干扰新任务，彻底节省 Token。
   
   3. /diff（查看修改）：
   Claude 在帮你改代码时，会直接修改你的本地文件。在你提交代码或运行前，输入 /diff，能以高亮形式清晰看到 Claude 刚刚对你的代码做了哪些改动。
   
   4. /code-review（代码审查）：
   让 Claude 对当前项目的代码质量、潜在 Bug 进行一次全面体检。
   
   5. /resume（恢复会话）：
   如果不小心断网或退出了，在新终端里输入 claude -c 或是交互界面输入 /resume，可以一键回到上一次的对话状态，继续之前的任务。
   '''
   ```

