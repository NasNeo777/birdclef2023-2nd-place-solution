# BirdCLEF 2023 第二名解决方案 (Kaggle)

## 项目概述
本项目是针对 [Kaggle BirdCLEF 2023 竞赛](https://www.kaggle.com/competitions/birdclef-2023) 的深度学习解决方案。该竞赛的核心挑战是识别长音频录音中的鸟类物种。本项目凭借结合了 CNN 和声音事件检测（SED）的集成架构，最终在比赛中获得了 **第二名**。

## 核心技术栈
- **编程语言**: Python
- **深度学习框架**: PyTorch, PyTorch Lightning
- **模型库**: `timm` (PyTorch Image Models)
- **音频处理**: `librosa`, `audiomentations`
- **实验管理**: Weights & Biases (W&B)
- **推理优化**: ONNX, OpenVINO

## 主要架构与方法
- **模型架构**:
    - **CNN 分类器**: 基于 EfficientNet-V2, ResNet, SeresNext 等强大的骨干网络。
    - **SED (Sound Event Detection)**: 专门的声音事件检测框架，能够更好地处理音频中声音出现的起止时间和重叠情况。
- **训练策略**:
    - **多阶段训练**: 预训练 (Pretraining) -> 正式训练 (Training) -> 微调 (Finetuning)。
    - **损失函数**: 结合使用交叉熵 (CE) 和二元交叉熵 (BCE) 损失来优化模型表现。
- **数据增强**: 针对生物声学数据进行了重度增强，包括音高平移（Pitch Shifting）、噪声注入（Noise Injection）和 Mixup 策略。

## 目录结构
- `train.py`: 训练模型的主入口点。
- `convert.py`: 将 PyTorch 模型转换为推理专用的 ONNX 和 OpenVINO 格式。
- `configs/`: 配置文件目录。
    - `common.py`: 全局通用配置（采样率、标签映射等）。
    - `*.py`: 特定模型的超参数配置。
- `modules/`: 核心逻辑组件。
    - `model.py`: 模型定义及 PyTorch Lightning 训练逻辑。
    - `dataset.py`: 数据加载器实现。
    - `preprocess.py`: 数据预处理与标签生成。
- `inputs/`: 存放原始音频、元数据及伪标签数据。
- `outputs/`: 存放训练好的模型权重和转换后的模型文件。

## 快速开始

### 1. 环境准备
安装必要的 Python 依赖包：
```bash
pip install -r requirements.txt
```

### 2. 配置实验追踪
在 `configs/common.py` 中设置你的 W&B API Key：
```python
cfg.WANDB_API_KEY = 'your_key_here'
```

### 3. 执行训练
通过指定阶段（Stage）和模型名称来启动训练：
```bash
# 示例：训练一个 EfficientNet-B0 CNN 模型
python3 train.py --stage train_ce --model_name cnn_b0ns
```
- **可选阶段**: `pretrain_ce`, `pretrain_bce`, `train_ce`, `train_bce`, `finetune`
- **可选模型**: `sed_v2s`, `sed_b3ns`, `sed_seresnext26t`, `cnn_v2s`, `cnn_resnet34d`, `cnn_b3ns`, `cnn_b0ns`

### 4. 模型转换
模型训练完成后，可转换为推理格式：
```bash
python3 convert.py --model_name cnn_b0ns
```

## 开发与运行建议
- **音频标准**: 本项目统一采用 **32kHz** 采样率。
- **路径调整**: 脚本中可能包含部分 Kaggle 平台或特定环境的硬编码路径（如 `/content/...`），在本地运行前请检查并更新。
- **模块化设计**: 超参数通过配置文件继承，修改配置时应优先修改 `configs/` 下的文件，而非直接改动核心逻辑代码。
