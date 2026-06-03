# World Action Model

世界动作模型 / 具身世界模型



年度进展APR

LLM（文本）      

↓ 添加视觉能力 

VLM（视觉+语言理解）      

↓ 添加动作控制 

VLA（视觉+语言+动作执行）      

↓ + 世界模型融合 

WAM / 具身世界模型（端到端感知-预测-决策-动作闭环）

- **2017-2021**：LLM为主（Transformer、GPT-3、CLIP）

- **2022-2023**：VLM兴起（PaLM-E、RT-1）

- **2023-2025**：VLA爆发（RT-2、RT-X、OpenVLA）

- **2025-2026**：VLA+世界模型融合（WAM`世界动作模型`、DyWA、GR00T）

  > World Model 作为 RL 训练环境为 VLA 后训练，潜在空间 CoT 替代文本 CoT，VLA 策略与 WM 迭代协同改进。

| **维度** | **LLM**      | **VLM**       | **VLA**        | **WAM**      |
| -------- | ------------ | ------------- | -------------- | ------------ |
| **模态** | 语言         | 视觉+语言     | 视觉+语言+动作 | 多模态+动态  |
| **能力** | 推理理解     | 感知+理解     | 感知+理解+执行 | 预测+因果    |
| **视角** | 数字世界     | 数字+物理感知 | 物理交互       | 物理规律     |
| **定位** | 基础语言模态 | 扩展感知能力  | 扩展执行能力   | 增强预测能力 |



## 1. 什么是 World Action Model？

World Action Model（WAM，世界动作模型）不是传统意义上的“只预测未来视频”的 world model，也不是“当前观测 → 动作”的 VLA policy。它的关键是把 **world prediction** 与 **action prediction** 合并为一个闭环学习目标。



**==工作定义：==**给定历史观测、语言目标和机器人状态，WAM 联合预测未来世界状态与动作序列。未来视觉状态不是附属可视化，而是动作策略学习物理因果、接触动态和跨具身迁移的密集监督信号。

$p(o_{t+1:t+H}, a_{t:t+H} | o \le t, language, robot_{state})$

$P(视频_{t+1},动作_{t}∣视频_{t},指令)$ [^1]



> [!TIP]
>
> 有人提出可以这么快速理解 WM 和 VLA：
>
> WM是客观模型、物理模型，从第三方视角描述世界，
>
> VLA是主观模型，从第一视角或者说机器人视角去定义。
>
> WAM 是 VLA和WM结合协同。
>
> VLA 学的是 “what action token should I output now”；WAM 学的是 “if I act this way, what will happen, and which action sequence makes that future likely”。





[^1]: [WAM vs. VLA](https://www.cnblogs.com/sasasatori/p/19759573) 