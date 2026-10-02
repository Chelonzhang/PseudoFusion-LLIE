# 零样本低照度图像增强 (Zero-Shot Low-Light Image Enhancement)

本仓库为论文方法的开源实现。方法在**零样本（zero-shot）**设定下工作：对每张输入图像独立进行测试时优化，**无需任何配对训练数据**。

## 方法概述

```
输入低照度图
  └─ 输入预处理 (YUV 色度双边滤波)
      └─ SRIE Retinex 分解 (MATLAB) → 反射分量 R / 光照分量 L
          └─ 伪多帧生成 + RAFT 光流对齐 + 分层保边融合 → R_fusion
              └─ 视界-LSTM (ViL) 双头精修: 反射残差 ΔR + 光照曲线参数 α
                  └─ 合成 (熵权锐化 + 可学习融合层) → 输出后处理 (gamma + 锐化)
                      └─ 最终增强图
```

核心模块：
- **SRIE 分解**：`model_retinex.py`（调用 MATLAB `srie.m` / `srie_wrapper.m`）
- **伪多帧与光流对齐**：`zero_shot_trainer.py`（`DeepViLSequenceModel` 前的序列构造）+ `step3_optical_flow.py`（RAFT）
- **ViL 双头序列精修**：`step4_vlstm_fusion.py`（`DeepViLSequenceModel`）+ `vision_lstm2.py`（官方 Vision-LSTM）
- **主训练/推理流程**：`zero_shot_trainer.py`
- **管线入口**：`pipeline.py`

## 环境依赖

- Python 3.8+
- PyTorch + torchvision
- numpy, opencv-python, einops
- **MATLAB**（R2018b 及以上，需安装适用于 Python 的 MATLAB Engine）
  - SRIE 分解由 MATLAB 实现（`srie.m`, `processing.p`），通过 `matlab.engine` 调用。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

安装 MATLAB Engine（在 MATLAB 安装目录下执行，路径按版本调整）：

```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

> 首次运行时，`model_retinex.py` 会自动启动 MATLAB 引擎并 `addpath` 到本目录，
> 以便调用 `srie_wrapper.m`。

## 运行（单张图）

```bash
python demo.py --input path/to/low_light.jpg --output output
```

或直接修改 `demo.py` 顶部的 `INPUT_PATH` / `OUTPUT_DIR`，然后：

```bash
python demo.py
```

输出：`output/<图像名>_enhanced.png`（最终增强结果）。

> 说明：本方法对每张图进行约 500 步测试时优化，单张处理需要一定时间（依赖硬件与分辨率）。

## 目录结构

```
重构开源/
├── demo.py                  # 单图增强入口
├── pipeline.py              # 增强管线 (预处理 + 优化 + 后处理)
├── zero_shot_trainer.py     # 零样本优化器与主前向
├── model_retinex.py         # SRIE Retinex 分解与去噪
├── step3_optical_flow.py    # RAFT 光流对齐
├── step4_vlstm_fusion.py    # ViL 双头序列精修 + 损失
├── vision_lstm2.py          # Vision-LSTM (官方实现, NXAI)
├── vision_lstm_util.py      # Vision-LSTM 工具
├── config.py                # 超参数配置
├── core_untils.py           # 张量/numpy 与保存工具
├── srie.m                   # SRIE 分解 (MATLAB)
├── srie_wrapper.m           # SRIE 封装 (MATLAB)
├── processing.p             # SRIE 依赖 (MATLAB)
├── requirements.txt
└── README.md
```

## 引用

若使用本代码，请引用相应论文。
（Vision-LSTM 来自 NXAI，见 `vision_lstm2.py` 头部版权声明。）
