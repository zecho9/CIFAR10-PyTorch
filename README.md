# CIFAR-10 Image Classification with PyTorch

基于 PyTorch 实现的 CIFAR-10 图像分类入门项目，完整实现从数据加载、CNN/ResNet-18 模型构建，到训练、验证、模型保存、Early Stopping、学习率调度、测试评估和 TensorBoard 可视化的完整深度学习流程。

## 1. 项目功能

- CIFAR-10 Dataset / DataLoader
- 数据增强与标准化
- SimpleCNN
- ResNet-18
- Train / Validation / Test 数据划分
- CrossEntropyLoss
- Adam Optimizer
- ReduceLROnPlateau 学习率调度
- Checkpoint 最佳模型保存
- Early Stopping
- Test Accuracy
- Confusion Matrix
- TensorBoard 训练可视化

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
├── train.py
├── test.py
├── test_model.py
├── confusion_matrix_resnet18.png
├── requirements.txt
├── .gitignore
└── README.md

