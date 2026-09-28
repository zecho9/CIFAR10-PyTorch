# Experiment Log

本文件用于记录 CIFAR-10 项目的实验配置、训练结果和实验结论。

---

## Exp-001: ResNet-18 CIFAR-10 Baseline

### Date

2026-09

### Goal

建立 CIFAR-10 ResNet-18 baseline，跑通完整的：

- Training
- Validation
- Checkpoint
- Early Stopping
- Learning Rate Scheduler
- Test
- Confusion Matrix
- TensorBoard

### Model

ResNet-18

针对 CIFAR-10 的 32×32 图像进行了以下修改：

- 第一层卷积：7×7, stride=2 → 3×3, stride=1
- 移除初始 MaxPool
- 最终全连接层输出：1000 → 10
- ImageNet pretrained weights：False

### Dataset

CIFAR-10

| Split | Samples |
|---|---:|
| Train | 45000 |
| Validation | 5000 |
| Test | 10000 |

### Config

```text
Model: ResNet-18
Batch Size: 128
Optimizer: Adam
Initial LR: 0.001
Max Epochs: 30

Scheduler:
    ReduceLROnPlateau
    Factor: 0.5
    Patience: 2
    Min Delta: 0.001

Early Stopping:
    Patience: 5
    Min Delta: 0.001


## Exp-002: ResNet-18 Reproducible Baseline

### Date

2026-09-28

### Goal

在 Exp-001 基础上加入完整随机种子控制，建立可复现的 CIFAR-10 ResNet-18 baseline。

### Changes

固定全局随机种子：

```text
SEED = 42
