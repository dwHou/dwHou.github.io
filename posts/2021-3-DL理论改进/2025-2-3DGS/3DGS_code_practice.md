https://github.com/graphdeco-inria/gaussian-splatting

官方推荐：https://www.youtube.com/watch?v=UXtuigy_wYc



#### 官方代码Quick-Start

##### 步骤1 - 安装环境

我是基于 pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel 的镜像

```shell
# 1. 检查已有包（推荐）
pip list | grep -E "(torch|opencv|plyfile|tqdm)"

# 2. 安装基础依赖
pip install plyfile opencv-python-headless joblib

# 3. 确保子模块已初始化
git submodule update --init --recursive

# 4. 安装 CUDA 扩展（需要编译）
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

验证：

```shell
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    import diff_gaussian_rasterization
    print('✓ diff-gaussian-rasterization')
except Exception as e:
    print(f'✗ diff-gaussian-rasterization: {e}')

try:
    import simple_knn
    print('✓ simple-knn')
except Exception as e:
    print(f'✗ simple-knn: {e}')

try:
    import fused_ssim
    print('✓ fused-ssim')
except Exception as e:
    print(f'✗ fused-ssim: {e}')

try:
    import plyfile
    print('✓ plyfile')
except Exception as e:
    print(f'✗ plyfile: {e}')

try:
    import cv2
    print(f'✓ opencv-python ({cv2.__version__})')
except Exception as e:
    print(f'✗ opencv-python: {e}')
"
```



##### 步骤2 - 处理数据

拍摄的几十张王老吉照片

修改`convert.py`:

- 设置 QT_QPA_PLATFORM=offscreen（Qt 无头模式）

- 设置 LIBGL_ALWAYS_SOFTWARE=1（强制软件渲染）

- 设置 MESA_GL_VERSION_OVERRIDE=3.3（Mesa OpenGL 版本）

- 设置 XDG_RUNTIME_DIR（解决运行时目录问题）

- 自动检测无头环境并禁用 GPU（避免 OpenGL 上下文错误）

```python
# 在第 28 行后添加以下代码：
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Additional environment variables for headless OpenGL support
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'
    os.makedirs(os.environ['XDG_RUNTIME_DIR'], exist_ok=True)

# Auto-disable GPU in headless environments if not explicitly requested
if not args.no_gpu and 'DISPLAY' not in os.environ:
    logging.warning("No DISPLAY detected in headless environment. Disabling GPU to avoid OpenGL context errors.")
    use_gpu = 0
else:
    use_gpu = 1 if not args.no_gpu else 0
```

运行 COLMAP 重建（数据预处理）

执行过程：

1. 特征提取（Feature Extraction）- 从图像提取 SIFT 特征
2. 特征匹配（Feature Matching）- 匹配图像间的特征点
3. 稀疏重建（Bundle Adjustment）- 生成相机参数和 3D 点云
4. 图像去畸变（Image Undistortion）- 生成理想针孔相机图像

```shell
python convert.py -s /containers-shared/3DGS/gaussian-splatting/dataset/wanglaoji
```

```
dataset/wanglaoji/
├── input/                    # 原始输入图像（31张）
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
├── images/                    # 去畸变后的图像（24张，7张被过滤）
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── ...
│
├── sparse/0/                  # COLMAP 稀疏重建结果（训练必需）
│   ├── cameras.bin           # 相机内参（二进制格式，64B）
│   ├── images.bin            # 相机外参和图像信息（二进制格式，3.2MB）
│   ├── points3D.bin          # 3D点云（二进制格式，388KB）
│   └── points3D.ply          # 3D点云（PLY格式，137KB）- 训练时自动生成
│
├── stereo/                    # COLMAP密集重建相关（训练不需要）
│   ├── depth_maps/           # 深度图
│   ├── normal_maps/          # 法线图
│   ├── consistency_graphs/    # 一致性图
│   ├── fusion.cfg            # 融合配置
│   └── patch-match.cfg       # 补丁匹配配置
│
└── distorted/                 # 中间文件（可删除）
    ├── database.db           # COLMAP数据库
    └── sparse/0/             # 畸变空间的重建结果
```

##### 步骤3 - 训练

```shell
cd /data/devonn/containers-shared/3DGS/gaussian-splatting

python train.py \
  -s /containers-shared/3DGS/gaussian-splatting/dataset/wanglaoji \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer
```

- -s: 数据集路径

- --iterations: 总迭代次数（默认 30,000）

- --test_iterations: 测试评估的迭代点

- --save_iterations: 保存模型的迭代点

- --disable_viewer: 禁用 GUI 查看器（无头环境必须）

训练过程：

- 前 500 次迭代：初始化阶段

- 500-15,000 次迭代：密集化阶段（逐步增加高斯点）

- 15,000-30,000 次迭代：优化阶段（微调参数）

- 在 7,000 和 30,000 次迭代时自动保存模型

步骤4 - 查看训练结果

```
output/f1298753-a/
├── cfg_args              # 训练配置参数
├── cameras.json          # 24个相机的参数
├── exposure.json         # 曝光补偿参数
├── input.ply             # 初始点云（136KB）
└── point_cloud/
    ├── iteration_7000/
    │   └── point_cloud.ply  # 中间模型（60MB）
    └── iteration_30000/
        └── point_cloud.ply  # 最终模型（70MB，约29.4万个高斯点）
```

可以用 [3DGS Viewers](https://www.3dgsviewers.com/) 来查看。



