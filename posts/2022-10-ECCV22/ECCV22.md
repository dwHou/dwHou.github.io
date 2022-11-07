1756 MuLUT: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution（LUT）

2021 Adaptive Patch Exiting for Scalable Single Image Super-Resolution（已知的策略，有的patch提取退出。但方法是把LR图像split，再把SR图像merge。这个有很多overhead。）

4417 Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks（思路类似，先过Base-Net有的patch继续走Refine-Net）

6048 Image Super-Resolution with Deep Dictionary（结合稀疏编码，主要在OOD数据上效果有优势）

6420 A Codec Information Assisted Framework for Efficient Compressed Video Super-Resolution（帧间对齐利用编码mv信息，利用编码residuals信息来选择性跳过“不重要”区域）

6698 Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution（LUT）

7199 Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images（渲染管线里的超分辨率）

Low-level vision Transformer: Restormer, Swin-IR, [HAT](https://arxiv.org/pdf/2205.04437.pdf)

AIM333 Efficient Image Super-Resolution Using Vast-Receptive-Field Attention（梳理了一遍注意力机制，并且提出了PN层，比较有意思。对于[B,C,HW]，BN是[B,HW]维度，LN是[C,HW]维度，IN是[HW]维度，PN是[C]维度。）









