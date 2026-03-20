# WP5 3D语义分割 — 模型训练与压缩加速

本项目是 [3D-IntelliScan](https://github.com/wangyisong-njust/intelliscan) 半导体检测流水线中 **3D语义分割模块** 的训练与优化工程。

实现了完整的优化流程：**基线训练** → **结构化剪枝** → **微调恢复** → **TensorRT量化** → **性能基准测试**，最终在 **精度无损**（Dice 0.9080 vs 基线 0.9038）的前提下，实现 **单patch推理11.7倍加速**、**参数量减少75%**。

---

## 项目结构

```
wp5-seg/
├── train.py                        # 基线模型训练
├── eval.py                         # 模型评估（Dice/IoU/HD/ASD）
├── run.sh                          # 训练启动脚本
├── run_eval.sh                     # 评估启动脚本
├── 3ddl-dataset/                   # 数据集加载子模块
├── pruning/
│   ├── prune_basicunet.py          # L2-norm结构化剪枝
│   ├── finetune_pruned.py          # 剪枝后微调训练
│   ├── export_onnx.py              # PyTorch → ONNX导出
│   ├── build_trt_engine.py         # ONNX → TensorRT引擎构建
│   ├── benchmark.py                # PyTorch FP32/AMP性能测试
│   ├── benchmark_trt.py            # TensorRT vs PyTorch性能对比
│   └── run_pruning_pipeline.sh     # 一键运行完整剪枝流程
└── pyproject.toml                  # 依赖配置（Python 3.12+）
```

---

## 模型架构

| 项目 | 规格 |
|------|------|
| 模型 | MONAI BasicUNet (3D) |
| 输入 | 单通道3D体积 `(1, 1, X, Y, Z)` |
| 输出 | 5类语义分割 |
| Features | `(32, 32, 64, 128, 256, 32)` |
| 参数量 | 5.75M |
| 归一化层 | InstanceNorm3d |
| 跳跃连接 | 拼接（Concatenation） |

**分割类别：**
- Class 0：背景（Background）
- Class 1：铜柱（Copper Pillar）
- Class 2：焊料（Solder）
- Class 3：空洞/缺陷（Void）
- Class 4：铜垫（Copper Pad）

### MONAI BasicUNet 与原始 UNet 的区别

原始UNet（2015年）是一个2D架构，用于医学图像分割，采用经典的编码器-解码器 + 跳跃连接设计。MONAI BasicUNet是其现代化的3D扩展版本，主要区别如下：

| 特性 | 原始UNet | MONAI BasicUNet |
|------|---------|-----------------|
| **维度** | 2D（Conv2d） | **3D**（Conv3d）— 处理体积数据 |
| **卷积块** | 2个3x3 Conv + BN + ReLU | TwoConv: 2个3x3x3 Conv + **InstanceNorm3d** + Dropout + **LeakyReLU** |
| **通道配置** | 固定（64, 128, 256, 512, 1024） | **可配置** `features=(f0, f1, f2, f3, f4, f5)` |
| **上采样** | 转置卷积或双线性插值 | **ConvTranspose3d**（可学习参数） |
| **归一化** | BatchNorm | **InstanceNorm3d**（适合小batch） |
| **激活函数** | ReLU | **LeakyReLU**（避免神经元死亡） |
| **跳跃连接** | 拼接（Concatenation） | 拼接（相同） |

简单说：BasicUNet = **3D版UNet + InstanceNorm + LeakyReLU + 可配置通道数**，是UNet针对3D医学图像的现代化改进版。

### 为什么使用 InstanceNorm3d 而不是 BatchNorm？

归一化层通过标准化中间激活值来稳定训练。不同归一化方式的核心区别在于**在哪个维度上计算均值和方差**：

假设输入张量形状为 `[B, C, D, H, W]`（批次、通道、深度、高、宽）：

| 归一化方式 | 计算范围 | 适用场景 |
|-----------|---------|---------|
| **BatchNorm** | 整个batch内，**同一通道**的所有体素 | batch较大时（如分类任务） |
| **InstanceNorm** | **单个样本、单个通道**的所有体素 | batch较小时（如分割、风格迁移） |
| LayerNorm | 单个样本的所有通道和体素 | NLP、Transformer |
| GroupNorm | 单个样本、分组通道的体素 | 介于BN和IN之间 |

```
输入: [B=4, C=32, D, H, W]

BatchNorm:    把4个样本的同一个通道的所有体素放一起，计算均值/方差
InstanceNorm: 只看1个样本的1个通道的所有体素，独立计算均值/方差
```

**这里为什么重要？** 3D医学图像分割的batch size通常很小（1~4），因为3D体积数据占用大量显存。BatchNorm在小batch下统计量不稳定（方差估计噪声大），导致训练不稳定。而InstanceNorm每个样本独立计算，不依赖batch大小，因此更加稳定可靠。

---

## 环境配置

```bash
# 基础依赖（Python 3.12+）
pip install monai>=1.5.1 nibabel>=5.3.3 torch

# 剪枝与TensorRT优化额外依赖
pip install onnx tensorrt pycuda
```

### 数据格式

```
data/
├── images/              # NIfTI .nii.gz 图像文件
├── labels/              # NIfTI .nii.gz 标签文件
├── metadata.jsonl       # 样本元数据
└── dataset_config.json  # 训练/测试划分（test_serial_numbers）
```

---

## 第一阶段：基线训练

### 训练命令

```bash
python train.py \
  --data_dir /path/to/data \
  --output_dir runs/wp5_baseline \
  --epochs 30 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

### 训练配置

| 项目 | 配置 |
|------|------|
| 损失函数 | 0.5 x CrossEntropy + 0.5 x Dice（忽略class 6） |
| 优化器 | Novograd（不可用时回退到Adam） |
| 学习率调度 | MultiStepLR（在60%和85%处衰减） |
| 数据增强 | ClipZScoreNormalize + 三轴随机翻转 + 随机裁剪 |
| ROI尺寸 | 112 x 112 x 80 |
| 推理方式 | 滑动窗口（overlap=0.5，高斯加权） |

**数据预处理 — ClipZScoreNormalize：**

这个名字拆开是三步操作：**Clip（截断）** + **Z-Score（标准化）** + **Normalize（归一化）**。

**什么是Z-Score？** Z-Score（标准分数）是统计学中最基本的标准化方法：

```
z = (x - 均值) / 标准差
```

将数据变换为均值=0、标准差=1的分布。变换后的值表示"距离均值多少个标准差"。

**什么是Clip？** Clip就是**截断**，把超出范围的值强制限制在范围内：

```
原始数据     = [0.1, 0.5, 0.8, 1.2, 100.0, -50.0]
                                      ↑        ↑ 异常值

clip到[0, 2] = [0.1, 0.5, 0.8, 1.2,   2.0,   0.0]
                                        ↑       ↑ 被截断
```

**ClipZScoreNormalize的完整流程：**

```python
# 假设一个3D体积的体素强度值
data = [0, 1, 2, 3, 5, 8, 10, 12, 15, 500]
#                                       ↑ 异常高值（比如金属伪影）

# 第1步：计算百分位数
p1  = 第1百分位  ≈ 0.09    # 最低1%的分界值
p99 = 第99百分位 ≈ 66.5    # 最高1%的分界值

# 第2步：Clip — 截断到[p1, p99]范围
clipped = [0.09, 1, 2, 3, 5, 8, 10, 12, 15, 66.5]
#          ↑ 被拉到p1                        ↑ 500被截断到66.5

# 第3步：Z-Score标准化
mean = clipped的均值 ≈ 12.26
std  = clipped的标准差 ≈ 19.5
result = (clipped - mean) / std
# 最终所有值大致在[-2, +3]范围内，均值0，标准差1
```

**为什么不直接用min-max归一化？**

```
# min-max归一化：
normalized = (x - min) / (max - min)

# 如果数据是 [0, 1, 2, 3, ..., 15, 500]
# min=0, max=500
# 那么 15/(500-0) = 0.03
# 正常的数据全部被压到[0, 0.03]这个极小范围里
# 99%的动态范围被一个异常值浪费了
```

ClipZScore先把异常值截断掉，再标准化，正常数据能保留完整的动态范围。这在半导体CT数据中尤为重要，因为金属材料（铜、焊料）容易产生强度极端值。

### 评估命令

```bash
python eval.py \
  --ckpt runs/wp5_baseline/best.ckpt \
  --data_dir /path/to/data \
  --output_dir runs/wp5_baseline/eval \
  --save_preds --heavy --hd_percentile 95
```

支持的评估指标：Dice系数、IoU（Jaccard）、Hausdorff距离（HD95）、平均表面距离（ASD）。

### 基线精度（174个测试样本）

| 类别 | Dice | IoU |
|------|------|-----|
| Class 0（背景） | 0.9919 | — |
| Class 1（铜柱） | 0.9409 | — |
| Class 2（焊料） | 0.9056 | — |
| Class 3（空洞） | 0.7995 | — |
| Class 4（铜垫） | 0.8818 | — |
| **平均** | **0.9038** | **0.8516** |

---

## 第二阶段：模型压缩

### 总体流程

```
训练好的模型（5.75M参数，22MB）
    ↓
[1] 结构化剪枝（50%通道）  →  1.44M参数，5.6MB
    ↓
[2] 微调恢复（50 epochs）   →  Dice: 0.3341 → 0.9080
    ↓
[3] TensorRT量化（FP16）    →  4.5MB引擎，1.72ms延迟
```

---

### 步骤1：结构化剪枝

**方法：** 基于L2-norm的通道重要性评估，对BasicUNet进行对称编码器-解码器结构化剪枝。

```bash
python pruning/prune_basicunet.py \
  --model_path runs/wp5_baseline/best.ckpt \
  --pruning_ratio 0.5 \
  --output_path output/pruned_model.ckpt
```

#### 技术细节

**1）通道重要性计算**

对于每个卷积层，输出通道 `i` 的重要性通过以下公式计算：

```
importance_i = ||W_i||_2 × |γ_i|
```

其中 `W_i` 是通道 `i` 对应的卷积核权重（展平后的L2范数），`γ_i` 是对应InstanceNorm3d层的仿射缩放参数。这个公式结合了滤波器本身的权重大小和归一化层学到的缩放因子，能更准确地衡量每个通道对输出的贡献。

- 如果一个通道的卷积权重很小（L2范数低），说明该通道提取的特征幅度小
- 如果归一化层的γ也很小，说明网络在训练过程中学会了抑制该通道
- 两者相乘后排序，移除重要性最低的通道

**2）对称剪枝约束**

BasicUNet采用 **拼接（Concatenation）** 跳跃连接，解码器中每个UpCat块将编码器的skip特征与上采样的解码器特征拼接：`cat[skip(f_enc), upsample(f_dec)]`。这产生了硬约束：编码器和对应的解码器层必须保持相同的通道数，否则拼接维度不匹配。

因此需要**对称剪枝**：

| 对称对 | 编码器 | 解码器 |
|--------|--------|--------|
| f1层 | `down_1` | `upcat_2` |
| f2层 | `down_2` | `upcat_3` |
| f3层 | `down_3` | `upcat_4` |

对于每个对称对，将编码器和解码器的通道重要性取平均，避免剪枝偏向某一侧。另有三个层独立剪枝：`conv_0`（输入层）、`down_4`（瓶颈层）、`upcat_1`（最终输出层）。

> 注意：这与VNet的剪枝不同。VNet使用元素级加法（element-wise addition）跳跃连接，而BasicUNet使用拼接。拼接方式在剪枝时需要特殊处理拼接维度的索引映射。

**3）架构感知的权重复制**

我们没有采用mask-based稀疏剪枝（只是将权重置零，不减少实际计算量），而是利用MONAI BasicUNet的 `features` 参数直接构建一个物理上更小的新网络。将原模型中被选中保留的通道权重精确复制到新模型中。

对于解码器的权重复制需要特别注意：拼接层的输入通道索引分为两部分——前半部分来自skip连接（编码器），后半部分来自上采样（解码器），需要分别映射正确的通道索引。

**剪枝结果（50%剪枝率）：**

| 项目 | 原始模型 | 剪枝后 | 压缩比 |
|------|----------|--------|--------|
| Features | (32, 32, 64, 128, 256, 32) | (16, 16, 32, 64, 128, 16) | — |
| 参数量 | 5.75M | 1.44M | **减少75%** |
| 模型大小 | 22MB | 5.6MB | **减少75%** |

> 为什么50%通道剪枝能减少75%参数？因为卷积层参数量与通道数呈二次关系（每层参数 = C_out × C_in × k³），当输入和输出通道都减半时，参数量变为原来的1/4。

---

### 步骤2：微调恢复精度

剪枝后模型精度从0.9038骤降至0.3341，需要通过微调（fine-tuning）恢复。为确保公平对比，微调使用与基线完全相同的训练配置。

```bash
python pruning/finetune_pruned.py \
  --pruned_model_path output/pruned_model.ckpt \
  --data_dir /path/to/data \
  --output_dir runs/finetune_pruned \
  --epochs 50 \
  --lr 1e-4
```

| 项目 | 配置 |
|------|------|
| 损失函数 | 0.5 x CE + 0.5 x Dice（与基线完全一致） |
| 优化器 | Adam，lr=1e-4（低于从头训练的学习率） |
| 学习率调度 | ReduceLROnPlateau（patience=5, factor=0.5） |
| 训练轮数 | 50 epochs |
| 数据增强 | 与基线训练完全一致 |

**微调结果：**

| 阶段 | 平均Dice |
|------|----------|
| 剪枝后（未微调） | 0.3341 |
| 微调后（50 epochs） | **0.9080** |
| 原始基线 | 0.9038 |

微调后的剪枝模型精度反而略高于原始基线（+0.0042 Dice），说明剪枝起到了**正则化**效果——去除冗余通道后，模型的泛化能力反而有所提升。

---

### 步骤3：TensorRT量化

将剪枝后的PyTorch模型转换为TensorRT优化推理引擎，进一步压缩延迟。

**转换流程：** `PyTorch (.ckpt) → ONNX (opset 18) → TensorRT Engine`

```bash
# 步骤3a：导出ONNX
python pruning/export_onnx.py \
  --model_path output/pruned_finetuned.ckpt \
  --model_format pruned \
  --output output/pruned.onnx

# 步骤3b：构建TensorRT引擎（FP16精度）
python pruning/build_trt_engine.py \
  --onnx_path output/pruned.onnx \
  --engine_path output/pruned_fp16.engine \
  --precision fp16

# 也支持FP32和INT8精度：
python pruning/build_trt_engine.py \
  --onnx_path output/pruned.onnx \
  --engine_path output/pruned_int8.engine \
  --precision int8
```

**ONNX导出细节：**
- 使用静态输入形状 `(1, 1, 112, 112, 80)`，不设置动态轴，以获得TensorRT最佳优化效果
- 权重内联存储在单个ONNX文件中（TensorRT要求）
- 使用opset version 18

**TensorRT构建细节：**
- 支持FP32、FP16、INT8三种精度级别
- INT8使用熵校准（`IInt8EntropyCalibrator2`），支持自定义校准数据
- INT8模式下同时启用FP16回退（用于不支持INT8的层）
- 4GB工作空间用于层优化策略搜索

---

### 性能基准测试

```bash
# PyTorch性能测试（原始 vs 剪枝）
python pruning/benchmark.py \
  --model_path best.ckpt --model_format state_dict \
  --compare_path pruned.ckpt --compare_format pruned \
  --num_runs 100 --amp

# TensorRT性能测试
python pruning/benchmark_trt.py \
  --pytorch_model pruned.ckpt --model_format pruned \
  --trt_engines pruned_fp32.engine pruned_fp16.engine pruned_int8.engine \
  --trt_labels FP32 FP16 INT8 \
  --num_runs 200
```

**单patch推理延迟对比（输入：112x112x80）：**

| 配置 | 延迟 | 加速比 | 参数量 | 模型大小 |
|------|------|--------|--------|----------|
| 原始 PyTorch FP32 | 20.14ms | 1.00x | 5.75M | 22.0MB |
| 原始 TRT FP32 | 14.54ms | 1.39x | 5.75M | 26.3MB |
| 原始 TRT FP16 | 5.20ms | 3.87x | 5.75M | 13.5MB |
| 原始 TRT INT8 | 3.39ms | 5.94x | 5.75M | 10.5MB |
| 剪枝 PyTorch FP32 | 8.87ms | 2.27x | 1.44M | 5.6MB |
| 剪枝 TRT FP32 | 4.49ms | 4.49x | 1.44M | 7.5MB |
| **剪枝 TRT FP16** | **1.72ms** | **11.71x** | **1.44M** | **4.5MB** |
| 剪枝 TRT INT8 | 1.72ms | 11.71x | 1.44M | 4.7MB |

**最优配置：剪枝 + TRT FP16**。INT8在该模型规模下无额外加速收益，而FP16无需校准数据，部署更简单。

### 为什么选择 TensorRT 而不是 torch.compile()？

`torch.compile()`（PyTorch 2.0引入）是另一种模型推理加速方案。我们对两者进行了评估，最终选择了TensorRT。以下是分析：

**Pipeline瓶颈背景：**

通过Profiling发现，模型推理仅占每个bbox处理时间的 **5%**，真正的瓶颈是文件I/O（77%）：

```
优化前每个bbox耗时分解：
  文件I/O (nib.save)    39.39ms   77.1%  ← torch.compile帮不了
  CPU归一化              8.27ms   16.2%  ← torch.compile帮不了（numpy操作）
  模型推理               2.60ms    5.1%  ← torch.compile只能优化这部分
  其他                   0.84ms    1.6%
```

即使推理加速到0ms，整个pipeline也只快5%。

**torch.compile() vs TensorRT 对比：**

| 维度 | torch.compile() | TensorRT（我们的选择） |
|------|-----------------|----------------------|
| 易用性 | 一行代码 `model = torch.compile(model)` | 需要ONNX导出 + engine构建 |
| 推理加速 | 约1.5-3x | **约5-12x**（FP16/INT8） |
| FP16支持 | 需配合 `torch.amp` | 原生支持，kernel级融合 |
| INT8支持 | 有限 | 成熟，有校准框架 |
| 首次调用 | 慢（30秒到几分钟编译） | 构建engine时慢，运行时快速加载 |
| 动态shape | 原生支持 | 需要profile或固定shape |
| 3D卷积优化 | 一般（Inductor后端） | 很好（专门优化的kernel） |

**延迟估算对比：**

```
原始 PyTorch FP32:          20.14ms
torch.compile（估计）:      ~8-12ms   （约2x加速）
TensorRT FP16（实测）:        1.72ms   （11.7x加速）
```

**什么情况下 torch.compile() 更合适？**

1. **快速原型阶段** — 不想折腾ONNX导出和TensorRT构建时，一行代码获得初步加速
2. **动态输入尺寸** — 如果输入shape经常变化，TensorRT需要重建engine，torch.compile更灵活
3. **不支持的算子** — 某些自定义算子TensorRT不支持，torch.compile覆盖范围更广
4. **训练加速** — torch.compile可以加速训练过程，TensorRT只能用于推理
5. **无TensorRT环境** — 当没有NVIDIA GPU或TensorRT不可用时，torch.compile可作为轻量替代（约2x加速）

**本项目的结论：** 在固定输入shape（112x112x80）、纯推理部署、TensorRT可用的条件下，TensorRT FP16严格优于torch.compile()。相比torch.compile()额外的5.8倍加速（1.72ms vs ~10ms），在每个样本处理数百个bbox时差异显著。而且，真正的瓶颈（文件I/O）是通过pipeline工程优化解决的，而非推理加速。

---

### 一键运行完整流程

```bash
MODEL_PATH=runs/wp5_baseline/best.ckpt \
DATA_DIR=/path/to/data \
PRUNING_RATIO=0.5 \
FINETUNE_EPOCHS=50 \
bash pruning/run_pruning_pipeline.sh
```

---

## 最终结果总结

| 指标 | 基线 | 优化后 | 变化 |
|------|------|--------|------|
| 单patch推理延迟 | 20.14ms | 1.72ms | **11.7x加速** |
| 参数量 | 5.75M | 1.44M | **减少75%** |
| 模型大小 | 22MB | 4.5MB (TRT FP16) | **减少80%** |
| 平均Dice | 0.9038 | 0.9080 | **+0.0042（无损）** |

**逐类Dice对比（174个测试样本）：**

| 类别 | 基线 | 优化后 |
|------|------|--------|
| Class 0（背景） | 0.9919 | 0.9918 |
| Class 1（铜柱） | 0.9409 | 0.9389 |
| Class 2（焊料） | 0.9056 | 0.9062 |
| Class 3（空洞） | 0.7995 | 0.8039 |
| Class 4（铜垫） | 0.8818 | 0.8934 |
| **平均Dice** | **0.9038** | **0.9069** |
| **平均IoU** | **0.8516** | **0.8535** |

---

## 与IntelliScan流水线集成

优化后的模型部署在 [IntelliScan](https://github.com/wangyisong-njust/intelliscan) 半导体检测流水线中。除了模型压缩之外，我们还实施了 **4大类Pipeline级工程优化**，以解决真正的瓶颈：文件I/O占处理时间的77%，而模型推理仅占5%。

### 瓶颈分析

将TRT FP16引擎集成到pipeline后，分割阶段仅从14.14s降到13.42s，加速效果远低于预期。逐步骤profiling揭示了原因：

| 步骤 | 耗时 | 占比 |
|------|------|------|
| `nib.save()` x2（保存.nii.gz文件） | 39.39ms | **77.1%** |
| normalize（CPU numpy） | 8.27ms | 16.2% |
| sliding_window_inference | 2.60ms | 5.1% |
| 其他（expand_bbox, to_tensor, argmax） | 0.84ms | 1.6% |
| **每个bbox合计** | **51.10ms** | **100%** |

**关键发现：** 模型推理仅占每个bbox处理时间的5%，77%的时间花在gzip压缩写中间NIfTI文件上。

### 优化1：去除中间文件I/O（最大贡献，63.7%）

**问题：** 每个bbox保存2个`.nii.gz`文件（crop + prediction），写入磁盘后，metrology阶段再从磁盘读回。

**方案：** 将分割和计量合并为单个阶段（`--combined-seg-metrology`），预测结果直接在内存中传递给metrology计算，完全消除 `nib.save()` + `nib.load()`。

此外，YOLO检测阶段也改为内存模式（`--inmemory`）：检测结果以Python字典形式返回，而非写入成千上万个文本文件到磁盘。

**效果：** 分割阶段 14.14s → 10.60s（**节省3.54s**）

### 优化2：跳过小crop的sliding_window_inference

**问题：** Pipeline中大多数bbox crop尺寸（约91x75x75）小于ROI尺寸（112x112x80）。MONAI的 `sliding_window_inference` 对这些crop只会产生单个窗口，但框架开销（高斯权重计算、padding策略、窗口管理）仍然存在。

**方案：** 对小于ROI尺寸的crop，直接使用**对称padding**（两侧均匀填充）到ROI尺寸后做单次forward，绕过sliding window框架。大于ROI的crop仍走sliding window。

**关键细节——对称padding：** padding必须是对称的（两侧等量填充），以匹配MONAI内部的padding策略。非对称padding（仅填充一侧）会导致Dice下降1.35%：

| Padding方式 | Dice | vs 基线 |
|------------|------|---------|
| sliding_window（基线） | 0.9069 | — |
| 对称padding（正确） | 0.9069 | **0.0000** |
| 非对称padding（错误） | 0.8945 | -0.0124 |

### 优化3：GPU归一化

**问题：** 每个crop的ClipZScore归一化在CPU上用numpy执行（`np.percentile` + `np.clip` + z-score），约8ms/crop。

**方案：** 将归一化搬到GPU，使用 `torch.quantile` + `torch.clamp` 替代numpy操作，数据全程不离开GPU，消除CPU-GPU传输开销。

**精度说明：** `torch.quantile` 与 `np.percentile` 存在微小数值差异，导致Dice从0.9080降至0.9069（-0.0011），可忽略不计。

### 优化4：批量推理

**问题：** 95个bbox逐个推理，每次都有独立的CUDA kernel launch、内存分配等开销。

**方案：** 将多个小crop对称padding到ROI尺寸后stack成batch，一次model forward处理多个crop。默认batch_size=8，将95次独立推理调用减少为约12次batch调用。

### 各优化贡献分解

| 优化 | 节省时间 | 贡献占比 |
|------|---------|---------|
| 去除.nii.gz文件I/O | 3.54s | **63.7%** |
| GPU归一化 | 0.58s | 10.4% |
| 批量推理 + TRT FP16 | 1.33s | 23.9% |
| 跳过sliding_window | 0.11s | 2.0% |
| **合计** | **5.56s** | **100%** |

### 端到端效果

| 指标 | 基线 | 优化后 | 变化 |
|------|------|--------|------|
| 分割阶段耗时 | 14.14s | 8.60s | **1.64x加速** |
| 流水线总耗时 | 59.59s | 31.71s | **1.88x加速** |
| 平均Dice | 0.9038 | 0.9069 | **+0.0031（无损）** |

### 两阶段优化总结

1. **模型压缩（剪枝 + TensorRT）** 将单次推理加速11.7倍，但受限于pipeline中非推理开销（文件I/O、框架开销等），实际pipeline加速有限。
2. **Pipeline工程优化** 通过profiling定位真正瓶颈（77%时间在文件I/O），针对性地去除中间文件保存、GPU化预处理、批量推理，实现1.65倍pipeline级加速。
3. 两阶段优化互补：模型压缩降低了推理计算量，工程优化消除了非计算瓶颈，最终在精度无损的前提下实现整体pipeline **1.88倍加速**（59.59s → 31.71s）。

---

## 生成的模型文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `segmentation_model_pruned.ckpt` | 剪枝+微调后的PyTorch模型 | 5.6MB |
| `segmentation_pruned_fp16.engine` | TensorRT FP16推理引擎（生产部署用） | 4.5MB |

## 依赖

- Python >= 3.12
- MONAI >= 1.5.1
- nibabel >= 5.3.3
- PyTorch（需CUDA支持）
- ONNX、TensorRT、PyCUDA（用于优化流程）
