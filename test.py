import os

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.resnet import get_resnet18
from utils.dataset import get_dataloaders


# =========================
# 1. 设备
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# =========================
# 2. 测试数据
# =========================

_, _, test_loader = get_dataloaders(
    batch_size=128,
    num_workers=0
)


# CIFAR-10类别名称
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# =========================
# 3. 创建ResNet-18
# =========================

model = get_resnet18(
    num_classes=10
)

model = model.to(device)


# =========================
# 4. 加载最佳Checkpoint
# =========================

checkpoint_path = os.path.join(
    "checkpoints",
    "best_resnet18.pth"
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print(
    f"加载最佳模型：Epoch "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint Val Loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Checkpoint Val Acc: "
    f"{checkpoint['val_acc']:.2f}%"
)


# =========================
# 5. 切换到评估模式
# =========================

model.eval()


# =========================
# 6. 测试统计变量
# =========================

correct = 0

total = 0


# 保存所有真实标签
all_labels = []


# 保存所有预测标签
all_predictions = []


# =========================
# 7. Test Loop
# =========================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        # -----------------
        # 前向传播
        # -----------------

        outputs = model(images)


        # -----------------
        # 得到预测类别
        # -----------------

        _, predicted = outputs.max(
            dim=1
        )


        # -----------------
        # Accuracy统计
        # -----------------

        total += labels.size(0)

        correct += (
            predicted.eq(labels)
            .sum()
            .item()
        )


        # -----------------
        # 保存标签
        # -----------------

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predicted.cpu().tolist()
        )


# =========================
# 8. Test Accuracy
# =========================

test_acc = (
    100.0 *
    correct /
    total
)


print(
    f"\nTest Accuracy: "
    f"{test_acc:.2f}%"
)


# =========================
# 9. Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\nConfusion Matrix:")

print(cm)


# =========================
# 10. 可视化混淆矩阵
# =========================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


fig, ax = plt.subplots(
    figsize=(10, 10)
)


display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)


plt.title(
    f"ResNet-18 CIFAR-10 Confusion Matrix\n"
    f"Test Accuracy: {test_acc:.2f}%"
)


plt.tight_layout()


# 保存图片
output_path = "confusion_matrix_resnet18.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print(
    f"\n混淆矩阵已保存："
    f"{output_path}"
)


plt.show()