# 背景

因为Unity不稳定的商业策略，许多游戏转向开源游戏引擎Godot（戈多）。Godot给广大开发者提供一种反垄断的，能够去跟Unity、其他收费游戏引擎议价的选择空间。

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=1556446603&bvid=BV151421t7yw&cid=1648154624&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

# 资料

Godot视频教程（Unity教程大神Brackeys➡️回归但是Godot）

Brackeys主页：https://www.youtube.com/@Brackeys

其他up主：[玩物不丧志的老李](https://space.bilibili.com/8618918)

Godot官网：https://godotengine.org/

Godot官方中文文档：https://docs.godotengine.org/zh-cn/4.x/

Godot官方英文文档：https://docs.godotengine.org/en/stable/

# 特点

对于没有经验的开发者，使用Godot游戏引擎+GD脚本来入门，是现阶段最好的学习途径。

唯一值得考虑的缺点是，作为开源免费的游戏引擎，Godot在商业化方面做的不好，在资源和工具上提供的就比较少。对于很多不会写代码或者没有制作素材能力的开发者，如果使用Unity、Unreal，我直接就做一个插件战士，直接在资产库（asset store）里买买买，攒一个游戏就完事了。但在Godot，甚至在可预见的未来，都达不到Unity那样资产商店的规模。

其他参考资料：

[1] [Godot vs. Unity](https://blog.csdn.net/lengyoumo/article/details/132112450)

# 官方教程

## 关于

### 前言

本文档分为几个部分：

- **关于** 包含了此简介以及有关引擎，历史，许可，作者等的信息。它还包含 [常见问题](https://docs.godotengine.org/zh-cn/4.x/about/faq.html#doc-faq)。
- **入门** 包含了有关使用引擎制作游戏的所有必要信息。它从 [渐进式](https://docs.godotengine.org/zh-cn/4.x/getting_started/step_by_step/index.html#toc-learn-step-by-step) 教程开始，这应该是所有新用户的切入点。**如果你是新手，这是最好的起点！**
- **手册** 可根据需要以任何顺序阅读或参考。它包含特定功能的教程和文档。
- **贡献** 提供了向Godot贡献相关的信息 ，无论是核心引擎、文档、 demo 还是其他部分。 它描述了如何报告 bug ，如何组织贡献者工作流等。 它还包含面向高级用户和贡献者的部分， 提供有关编译引擎的信息，为编辑器做出贡献， 或开发C++模块。
- **社区** 致力于 Godot 社区的生态。它指向各种社区渠道，如 [Godot 贡献者聊天](https://chat.godotengine.org/) 和 [Discord](https://discord.gg/4JBkykG)，并包含本文档之外推荐的第三方教程和资料。
- 最后，**类参考**记录的是完整的 Godot API，另外也可以直接在引擎的脚本编辑器中查看。你可以在这里找到关于所有类、函数、信号等相关的信息。

除本文档外，你可能还会对各种 [Godot 示例项目](https://github.com/godotengine/godot-demo-projects)感兴趣。

### 特性列表

1. 平台：Godot 的目标是尽可能地独立于平台，并且可以相对轻松地 [移植到新平台](https://docs.godotengine.org/zh-cn/4.x/contributing/development/core_and_modules/custom_platform_ports.html#doc-custom-platform-ports) 。

2. 编辑器：

   - 场景树编辑器。
   - 内置脚本编辑器。
   - 支持 Visual Studio Code、VIM 等[外部文本编辑器](https://docs.godotengine.org/zh-cn/4.x/tutorials/editor/external_editor.html#doc-external-editor)。
   - GDScript [调试器](https://docs.godotengine.org/zh-cn/4.x/tutorials/scripting/debug/debugger_panel.html#doc-debugger-panel)。
   - 可视化（性能）分析器能指出在渲染管线中 CPU 与 GPU 在每个步骤花费的时间。
   - 性能监视工具，包括[自定义性能监视器](https://docs.godotengine.org/zh-cn/4.x/tutorials/scripting/debug/custom_performance_monitors.html#doc-custom-performance-monitors)。

   等等

3. 渲染：

   桌面平台默认Forward+，移动平台默认Forward Mobile，Web平台默认Compatibility。

4. 2D 图形

5. 2D 工具

6. 2D 物理

7. 3D 图形

8. 3D 工具

9. 3D 物理

10. 着色器（Shaders）

11. 编写脚本（Scripting）

    - General
    - GDScript
    - C#
    - GDExtension（C、C++、Rust、D……）

12. 音频

13. 导入（import）

14. 输入（input）

15. 导航

16. 网络

17. 国际化

18. 窗口 和 操作系统整合

19. 移动端

20. XR支持（AR和VR）

21. GUI系统

22. 动画

23. 文件格式

24. 杂项

### 系统需求

Godot编辑器的推荐配置，最低配置。导出Godot项目的推荐配置，最低配置。

**编辑器：**

- Windows、macOS、Linux、*BSD、Android（实验性）、[网页版](https://editor.godotengine.org/)（实验性）

**导出游戏：**

- Windows、macOS、Linux、*BSD、Android、iOS、Web

### 常见问题

<font color="blue">Q：</font>Godot 支持哪些编程语言？

<font color="Red">A：</font>Godot 官方支持的语言是 GDScript、C# 和 C++。如果你刚开始接触 Godot 或一般的游戏开发，推荐学习并使用 <font color="brown">GDScript</font> 语言，它是 Godot 的原生语言。虽从长远来看，脚本语言的性能往往不如低级语言，但对于原型设计、开发最小可行产品（Minimum Viable Products）以及关注上市时间（Time-To-Market）而言，GDScript 可提供一种快速、友好、能力强的游戏开发方式。

对于新语言，可以通过第三方使用 GDExtension 获得支持。

<font color="blue">Q：</font>GDScript是什么？ 为什么要使用这门语言？

<font color="Red">A：</font>GDScript 是 Godot 所集成的一门是从零开始构建的脚本语言，其目的就是用最少的代码将 Godot 的潜力最大化地发挥出来，让新手和专业开发人员都能尽可能快地利用 Godot 的优势进行开发。<font color="brown">如果你曾经用类似 Python 这样的语言写过任何东西，那么你就会对 GDScript 感到得心应手。</font>想了解关于 GDScript 的示例以及完整的功能介绍，请参阅 [GDScript 脚本指南](https://docs.godotengine.org/zh-cn/4.x/tutorials/scripting/gdscript/gdscript_basics.html#doc-gdscript)。

使用 GDScript 有不少原因，特别是你在进行原型设计时、在项目的 alpha/beta 阶段、或者项目不是 3A 大作时会用到它，但 GDScript 最突出的优势就是整体**复杂度得到降低**。

<font color="blue">Q：</font>为什么 Godot 使用 Vulkan/OpenGL 而不是 Direct3D ？

<font color="Red">A：</font>Godot 致力于实现跨平台兼容性和开放式标准。OpenGL 和 Vulkan 是几乎在所有平台上都开放且可用的技术。得益于这一设计，在 Windows 上使用 Godot 开发的项目也能在 Linux、macOS 等平台上开箱即用。

虽然我们主要专注于 Vulkan 和 OpenGL，因为它们具有开放标准和跨平台的优势，但 Godot 4.3 引入了对 Direct3D 12 的实验性支持。此举旨在提升在 Direct3D 12 广泛使用的平台（如 Windows 和 Xbox）上的性能和兼容性。然而，Vulkan 和 OpenGL 将继续作为所有平台（包括 Windows）上的默认渲染后端。

<font color="blue">Q：</font>是否能用 Godot 创建非游戏应用？

<font color="Red">A：</font>是的！Godot 具有广泛的内置 UI 系统，其较小的软件包可以使它成为 Electron 或 Qt 等框架的合适替代品。

当创建一个非游戏的应用程序时，确保在项目设置中启用 [低处理器模式](https://docs.godotengine.org/zh-cn/4.x/classes/class_projectsettings.html#class-projectsettings-property-application-run-low-processor-mode) 以减少CPU和GPU占用。

### 遵守许可证

对于MIT许可证，唯一的要求是将许可证文本包含在你的游戏或衍生项目中。

### 发布策略

Godot 的发布政策是在不断改进的。以下内容提供了大致的预期结果，但实际会发生什么取决于核心贡献者的选择，以及社区在特定时期的需求。

Godot版本：Godot 松散地遵循了[语义化版本](https://semver.org/)，采用了 `major.minor.patch` 的版本系统

> 如果你必须学习为 Godot 3.x 设计的教程，我们建议在单独的选项卡中保持 [从 Godot 3 升级到 Godot 4](https://docs.godotengine.org/zh-cn/4.x/tutorials/migrating/upgrading_to_godot_4.html#doc-upgrading-to-godot-4) 打开，以检查哪些方法已被重命名（如果你在尝试使用特定节点或在 Godot 4.x 中被重命名的方法时遇到了脚本错误的话）。

## 入门

### 前言

#### Godot简介



## 手册

## 贡献

## 社区

## 类参考

# 视频教程

