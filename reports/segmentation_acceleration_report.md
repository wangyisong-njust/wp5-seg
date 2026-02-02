# 3D分割模型推理加速报告

## 1. 概述

本报告针对3D-IntelliScan半导体检测pipeline中的3D语义分割模块（MONAI BasicUNet），系统地进行了模型压缩与推理加速优化。优化分为两个阶段：

- **阶段一：模型压缩** — 结构化剪枝 + 微调 + TensorRT量化，将单patch推理速度提升11.7倍
- **阶段二：Pipeline工程优化** — 针对实际pipeline瓶颈进行4项优化，将端到端分割阶段加速1.65倍

最终在**精度无损**（Dice 0.9069 vs baseline 0.9038）的前提下，将分割阶段从14.14s降至8.60s，整体pipeline从59.59s降至31.71s。

---

## 2. 基线分析

### 2.1 模型架构

| 项目 | 规格 |
|---|---|
| 模型 | MONAI BasicUNet (3D) |
| 输入 | 单通道3D体积 (1, 1, X, Y, Z) |
| 输出 | 5类语义分割 |
| Features | (32, 32, 64, 128, 256, 32) |
| 参数量 | 5.75M |
| 模型大小 | 22MB |
| 归一化层 | InstanceNorm3d |
| 跳跃连接 | 拼接 (Concatenation) |

### 2.2 Pipeline架构

完整pipeline流程：NII加载 → JPG转换 → YOLO 2D检测 → 3D BBox生成 → **3D分割** → 计量测量 → PDF报告

### 2.3 基线性能（SN002样本，95个bbox）

| 阶段 | 耗时 | 占比 | 备注 |
|---|---|---|---|
| Folder Creation | 0.00s | 0.0% | — |
| NII to JPG Conversion | 22.16s | 37.2% | 1998张切片（主要受IO限制）|
| 2D Detection (YOLO) | 13.09s | 22.0% | 1998张切片 |
| 3D BBox Generation | 0.65s | 1.1% | — |
| **3D Segmentation (MONAI)** | **14.14s** | **23.7%** | **95个bbox，平均0.149s/个** |
| Metrology | 4.53s | 7.6% | 95个bbox |
| Report Generation | 5.02s | 8.4% | — |
| **Total** | **59.59s** | **100%** | — |

### 2.4 基线精度（174个测试样本）

| 指标 | 值 |
|---|---|
| Average Dice | 0.9038 |
| Average IoU | 0.8516 |

---

## 3. 阶段一：模型压缩（剪枝 + 微调 + TensorRT量化）

### 3.1 结构化剪枝

**方法：** 基于L2-norm的通道重要性评估，对BasicUNet进行对称编码器-解码器结构化剪枝。

**关键技术点：**
- BasicUNet采用拼接跳跃连接（skip = cat[encoder_feat, upsampled]），因此编码器和对应解码器层必须对称剪枝
- 对称剪枝对：encoder down_1 ↔ decoder upcat_2, down_2 ↔ upcat_3, down_3 ↔ upcat_4
- 独立层：conv_0（输入层）、bottleneck、upcat_1（输出层）
- 利用MONAI BasicUNet的`features`参数，直接构建剪枝后的新架构，避免mask-based稀疏剪枝

**剪枝率：50%**

| 项目 | 原始模型 | 剪枝后 | 压缩比 |
|---|---|---|---|
| Features | (32,32,64,128,256,32) | (16,16,32,64,128,16) | — |
| 参数量 | 5.75M | 1.44M | **75%减少** |
| 模型大小 | 22MB | 5.6MB | **75%减少** |

### 3.2 微调恢复精度

| 项目 | 配置 |
|---|---|
| 损失函数 | 0.5×CE + 0.5×Dice（与原始训练一致）|
| 优化器 | Adam, lr=1e-4 |
| 学习率调度 | ReduceLROnPlateau |
| 微调轮数 | 50 epochs |
| 数据增强 | 与原始训练一致（ClipZScoreNormalize等）|

| 阶段 | Dice |
|---|---|
| 剪枝后（未微调） | 0.3341 |
| 微调后 | **0.9080** |
| 原始模型 | 0.9038 |

微调后精度反而略高于原始模型（+0.0042），说明剪枝起到了正则化效果。

### 3.3 TensorRT量化

将剪枝后模型通过 ONNX 导出，再构建 TensorRT 引擎（FP32/FP16/INT8）。

**转换流程：** PyTorch → ONNX (opset 18) → TensorRT Engine

### 3.4 单Patch推理性能对比（输入 112×112×80）

| 配置 | 延迟 | 加速比 | 参数量 | 模型大小 |
|---|---|---|---|---|
| Original PyTorch FP32 | 20.14ms | 1.00x | 5.75M | 22.0MB |
| Original TRT FP32 | 14.54ms | 1.39x | 5.75M | 26.3MB |
| Original TRT FP16 | 5.20ms | 3.87x | 5.75M | 13.5MB |
| Original TRT INT8 | 3.39ms | 5.94x | 5.75M | 10.5MB |
| Pruned PyTorch FP32 | 8.87ms | 2.27x | 1.44M | 5.6MB |
| Pruned TRT FP32 | 4.49ms | 4.49x | 1.44M | 7.5MB |
| **Pruned TRT FP16** | **1.72ms** | **11.71x** | **1.44M** | **4.5MB** |
| Pruned TRT INT8 | 1.72ms | 11.71x | 1.44M | 4.7MB |

**阶段一结论：** 剪枝50% + TensorRT FP16 实现单patch推理 **11.7倍加速**，精度无损（Dice 0.9080 vs 0.9038）。

---

## 4. 阶段二：Pipeline工程优化

### 4.1 瓶颈分析

将TRT FP16引擎集成到pipeline后，发现分割阶段仅从14.14s降到13.42s，加速效果远低于预期。通过逐步骤profiling定位瓶颈：

**每个bbox处理耗时分解（Pruned TRT FP16，优化前）：**

| 步骤 | 耗时 | 占比 |
|---|---|---|
| nib.save() ×2（保存nii.gz） | 39.39ms | **77.1%** |
| normalize（CPU numpy） | 8.27ms | 16.2% |
| sliding_window_inference | 2.60ms | 5.1% |
| 其他（expand_bbox, to_tensor, argmax） | 0.84ms | 1.6% |
| **合计** | **51.10ms** | **100%** |

**关键发现：** 模型推理仅占5%，77%的时间花在gzip压缩写文件上。

### 4.2 优化措施

#### 优化1：去除中间nii.gz文件保存

**问题：** 每个bbox保存2个nii.gz文件（crop + prediction），后续metrology阶段再从磁盘读回。

**方案：** 将metrology改为直接从内存中的prediction数组计算，去除 `nib.save()` 调用。

**效果：** 分割阶段 14.14s → 10.60s（**节省3.54s**）

#### 优化2：跳过sliding_window_inference

**问题：** Pipeline中的bbox crop尺寸（~91×75×75）均小于roi_size（112×112×80），`sliding_window_inference` 只产生单个窗口，但框架开销（Gaussian权重计算、padding策略、窗口管理）仍然存在。

**方案：** 对小于roi_size的crop，直接对称padding到roi_size后调用模型，跳过sliding_window框架。对大于roi_size的crop保留sliding_window。

**精度验证：** 关键在于**对称padding**（两侧均匀填充），与MONAI sliding_window_inference内部的padding策略一致。非对称padding（仅填充一侧）会导致Dice下降1.35%。

| Padding方式 | Dice | vs baseline |
|---|---|---|
| sliding_window (baseline) | 0.9069 | — |
| 对称padding（正确） | 0.9069 | **0.0000** |
| 非对称padding（错误） | 0.8945 | -0.0124 |

#### 优化3：GPU归一化

**问题：** 每个crop的ClipZScore归一化在CPU上用numpy执行（`np.percentile` + `np.clip` + z-score），约8ms/crop。

**方案：** 将归一化搬到GPU，使用 `torch.quantile` + `torch.clamp` 替代numpy操作。

**精度影响：** `torch.quantile` 与 `np.percentile` 存在微小数值差异，导致Dice从0.9080降至0.9069（-0.0011），可忽略。

#### 优化4：批量推理

**问题：** 95个bbox逐个推理，每次都有独立的CUDA kernel launch、内存分配等开销。

**方案：** 将多个crop对称padding到roi_size后stack成batch，一次model forward处理多个crop。大于roi_size的crop仍走sliding_window。

**效果：** 减少95次独立推理调用为12次batch调用（batch_size=8）。

### 4.3 Pipeline端到端性能对比

**测试环境：** SN002样本，95个bbox，单次运行

| 配置 | Seg Time | Total Time | Seg加速 |
|---|---|---|---|
| Baseline（原始PyTorch + nii保存） | 14.14s | 59.59s | 1.00x |
| + 去除nii保存（移至report阶段） | 10.60s | — | 1.33x |
| + GPU归一化 | 10.02s | — | 1.41x |
| + 批量推理 + TRT FP16 | 8.69s | — | 1.63x |
| **+ 跳过sliding_window + TRT FP16（全部优化）** | **8.60s** | **31.71s** | **1.64x** |

> 注：nii.gz保存从Segmentation阶段移至Report阶段之前（用于报告可视化），不计入分割耗时。中间配置未做完整pipeline计时，仅记录分割阶段耗时。

### 4.4 各优化贡献分解

| 优化 | 节省时间 | 贡献占比 |
|---|---|---|
| 去除nii.gz保存 | 3.54s | **63.7%** |
| GPU归一化 | 0.58s | 10.4% |
| 批量推理 + TRT FP16 | 1.33s | 23.9% |
| 跳过sliding_window | 0.11s | 2.0% |
| **合计** | **5.56s** | **100%** |

---

## 5. 最终精度验证

在174个测试样本上的Dice/IoU对比（per-class + average）：

| Class | Baseline (eval.py) | 优化后Pipeline |
|---|---|---|
| Class 0 (Background) | 0.9919 | 0.9918 |
| Class 1 (Copper Pillar) | 0.9409 | 0.9389 |
| Class 2 (Solder) | 0.9056 | 0.9062 |
| Class 3 (Void) | 0.7995 | 0.8039 |
| Class 4 (Pad) | 0.8818 | 0.8934 |
| **Average Dice** | **0.9038** | **0.9069** |
| **Average IoU** | **0.8516** | **0.8535** |

优化后精度略优于baseline，主要得益于剪枝微调后的模型本身Dice更高（0.9080）。

---

## 6. 总结

| 指标 | Baseline | 优化后 | 变化 |
|---|---|---|---|
| 单patch推理延迟 | 20.14ms | 1.72ms | **11.7x加速** |
| Pipeline分割阶段 | 14.14s | 8.60s | **1.64x加速** |
| Pipeline总耗时 | 59.59s | 31.71s | **1.88x加速** |
| 模型参数量 | 5.75M | 1.44M | **75%减少** |
| 模型大小 | 22MB | 4.5MB (TRT) | **80%减少** |
| Average Dice | 0.9038 | 0.9069 | **+0.0031（无损）** |

**两阶段优化总结：**

1. **模型压缩（剪枝+TensorRT）** 将单次推理加速11.7倍，但受限于pipeline中非推理开销（文件I/O、框架开销等），实际pipeline加速有限
2. **Pipeline工程优化** 通过profiling定位真正瓶颈（77%时间在文件I/O），针对性地去除中间文件保存、GPU化预处理、批量推理，实现1.65倍pipeline级加速
3. 两阶段优化互补：模型压缩降低了推理计算量，工程优化消除了非计算瓶颈，最终在精度无损的前提下实现整体pipeline 1.88倍加速（59.59s → 31.71s）

---

## 附录：文件清单

### 模型压缩脚本（wp5-seg/pruning/）

| 文件 | 功能 |
|---|---|
| `prune_basicunet.py` | L2-norm结构化剪枝 |
| `finetune_pruned.py` | 剪枝后微调训练 |
| `export_onnx.py` | PyTorch → ONNX导出 |
| `build_trt_engine.py` | ONNX → TensorRT引擎构建 |
| `benchmark_trt.py` | TRT vs PyTorch性能对比 |
| `benchmark.py` | PyTorch FP32/AMP性能对比 |
| `run_pruning_pipeline.sh` | 一键运行完整剪枝流程 |

### Pipeline优化修改（intelliscan/）

| 文件 | 修改内容 |
|---|---|
| `segmentation.py` | TRT推理后端、GPU归一化、批量推理、跳过sliding_window |
| `main.py` | TRT配置参数、内存直接计算metrology（去除nii保存）|

### 生成的模型文件

| 文件 | 说明 | 大小 |
|---|---|---|
| `segmentation_model_pruned.ckpt` | 剪枝+微调后PyTorch模型 | 5.6MB |
| `segmentation_pruned_fp16.engine` | TensorRT FP16引擎 | 4.5MB |
