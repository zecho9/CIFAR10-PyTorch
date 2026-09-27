import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from utils.dataset import get_dataloaders


# =========================
# 1. 选择运行设备
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# =========================
# 2. 加载数据
# =========================

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=128,
    num_workers=0
)


# =========================
# 3. 创建模型
# =========================

model = SimpleCNN().to(device)


# =========================
# 4. 损失函数
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# 5. 优化器
# =========================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 6. Learning Rate Scheduler
# =========================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,

    # 监控的指标越小越好
    mode="min",

    # 每次降低为原来的 0.5
    factor=0.5,

    # 验证指标若连续若干轮没有明显改善，
    # 就降低学习率
    patience=2,

    # 至少改善 0.001 才算明显改善
    threshold=0.001,

    threshold_mode="abs",

    # 学习率最低不低于这个值
    min_lr=1e-6
)


# =========================
# 7. 训练配置
# =========================

num_epochs = 30


# =========================
# 8. Checkpoint 配置
# =========================

best_val_loss = float("inf")


checkpoint_dir = "checkpoints"

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)


checkpoint_path = os.path.join(
    checkpoint_dir,
    "best_model.pth"
)


# =========================
# 9. Early Stopping 配置
# =========================

patience = 5

early_stop_counter = 0

min_delta = 0.001


# =========================
# 10. Training Loop
# =========================

for epoch in range(num_epochs):

    print(
        f"\nEpoch {epoch + 1}/{num_epochs}"
    )


    # ==================================================
    # Train
    # ==================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        # --------------------------
        # 数据搬到 GPU
        # --------------------------

        images = images.to(device)
        labels = labels.to(device)


        # --------------------------
        # 1. 清空上一批数据的梯度
        # --------------------------

        optimizer.zero_grad()


        # --------------------------
        # 2. 前向传播
        # --------------------------

        outputs = model(images)


        # --------------------------
        # 3. 计算 Loss
        # --------------------------

        loss = criterion(
            outputs,
            labels
        )


        # --------------------------
        # 4. 反向传播
        # --------------------------

        loss.backward()


        # --------------------------
        # 5. 更新模型参数
        # --------------------------

        optimizer.step()


        # --------------------------
        # 统计 Loss
        # --------------------------

        running_loss += loss.item()


        # --------------------------
        # 获取预测类别
        # --------------------------

        _, predicted = outputs.max(
            dim=1
        )


        # 当前 batch 样本数量
        total += labels.size(0)


        # 当前 batch 正确数量
        correct += (
            predicted.eq(labels)
            .sum()
            .item()
        )


    # ==================================================
    # Train 指标
    # ==================================================

    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_acc = (
        100.0 *
        correct /
        total
    )


    # ==================================================
    # Validation
    # ==================================================

    model.eval()


    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            # 前向传播
            outputs = model(images)


            # 计算验证 Loss
            loss = criterion(
                outputs,
                labels
            )


            # 累加验证 Loss
            val_running_loss += loss.item()


            # 获取预测类别
            _, predicted = outputs.max(
                dim=1
            )


            # 验证样本总数
            val_total += labels.size(0)


            # 验证正确数量
            val_correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )


    # ==================================================
    # Validation 指标
    # ==================================================

    val_loss = (
        val_running_loss /
        len(val_loader)
    )


    val_acc = (
        100.0 *
        val_correct /
        val_total
    )


    # ==================================================
    # 当前 Learning Rate
    # ==================================================

    current_lr = optimizer.param_groups[0]["lr"]


    # ==================================================
    # 输出结果
    # ==================================================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    # ==================================================
    # 11. Learning Rate Scheduler
    # ==================================================

    old_lr = optimizer.param_groups[0]["lr"]


    # ReduceLROnPlateau需要把监控指标传进去
    scheduler.step(val_loss)


    new_lr = optimizer.param_groups[0]["lr"]


    # 如果学习率发生变化
    if new_lr < old_lr:

        print(
            f"学习率降低："
            f"{old_lr:.6f} -> {new_lr:.6f}"
        )


    # ==================================================
    # 12. Checkpoint + Early Stopping
    # ==================================================

    if val_loss < best_val_loss - min_delta:

        # --------------------------
        # 验证集出现明显改善
        # --------------------------

        best_val_loss = val_loss


        # Early Stopping重新计数
        early_stop_counter = 0


        # --------------------------
        # 保存最佳模型
        # --------------------------

        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                # scheduler状态也一起保存
                "scheduler_state_dict":
                    scheduler.state_dict(),

                "train_loss":
                    train_loss,

                "train_acc":
                    train_acc,

                "val_loss":
                    val_loss,

                "val_acc":
                    val_acc,

                "best_val_loss":
                    best_val_loss
            },

            checkpoint_path
        )


        print(
            f"保存最佳模型："
            f"Val Loss = {best_val_loss:.4f}"
        )


    else:

        # --------------------------
        # 验证集没有明显改善
        # --------------------------

        early_stop_counter += 1


        print(
            f"验证集未改善："
            f"{early_stop_counter}/{patience}"
        )


        # --------------------------
        # Early Stopping
        # --------------------------

        if early_stop_counter >= patience:

            print(
                "连续多轮验证集未改善，"
                "触发 Early Stopping。"
            )

            break


# =========================
# 13. 训练结束
# =========================

print("\n训练结束")


print(
    f"最佳 Val Loss："
    f"{best_val_loss:.4f}"
)


print(
    f"最佳模型保存在："
    f"{checkpoint_path}"
)