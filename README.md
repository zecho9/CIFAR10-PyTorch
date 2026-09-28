# CIFAR-10 Image Classification with PyTorch

基于 PyTorch 实现的 CIFAR-10 图像分类入门项目。

项目完整实现从数据加载、CNN / ResNet-18 模型构建，到训练、验证、Checkpoint、Early Stopping、学习率调度、测试评估、混淆矩阵和 TensorBoard 可视化的完整深度学习流程。

---

## 1. 项目功能

- CIFAR-10 Dataset / DataLoader
- 数据增强与标准化
- Train / Validation / Test 数据划分
- SimpleCNN
- ResNet-18
- CrossEntropyLoss
- Adam Optimizer
- ReduceLROnPlateau 学习率调度
- Checkpoint 最佳模型保存
- Early Stopping
- Test Accuracy
- Confusion Matrix
- TensorBoard 训练可视化
- 固定随机种子
- 实验超参数记录
- 实验日志管理

---

## 2. 项目结构

```text
CIFAR10-PyTorch/
│
├── models/
│   ├── cnn.py
│   └── resnet.py
│
├── utils/
│   ├── __init__.py
│   └── dataset.py
│
├── experiments/
│   └── experiment_log.md
│
├── train.py
├── test.py
├── test_model.py
│
├── confusion_matrix_resnet18.png
├── requirements.txt
├── .gitignore
└── README.md
```