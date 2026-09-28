import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models.cnn import SimpleCNN
from models.resnet import get_resnet18
from utils.dataset import get_dataloaders


# ==================================================
# 0. 实验配置
# ==================================================

# --------------------------
# 随机种子
# --------------------------

SEED = 42


# --------------------------
# 模型
# --------------------------

MODEL_NAME = "resnet18"

NUM_CLASSES = 10


# --------------------------
# 数据
# --------------------------

DATA_DIR = "./data"

BATCH_SIZE = 128

NUM_WORKERS = 0


# --------------------------
# Optimizer
# --------------------------

INITIAL_LR = 0.001


# --------------------------
# Training
# --------------------------

NUM_EPOCHS = 30


# --------------------------
# Early Stopping
# --------------------------

EARLY_STOP_PATIENCE = 5

MIN_DELTA = 0.001


# --------------------------
# Learning Rate Scheduler
# --------------------------

LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-6


# ==================================================
# 1. 固定随机种子
# ==================================================

def set_seed(seed):
    """
    尽可能固定实验中的随机性，
    提高实验结果的可复现性。
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)

    # 多 GPU
    torch.cuda.manual_seed_all(seed)

    # cuDNN 使用确定性实现
    torch.backends.cudnn.deterministic = True

    # 不自动搜索最快卷积算法，
    # 避免不同运行之间选择不同算法
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ==================================================
# 2. 选择运行设备
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("使用设备：", device)

print(
    f"随机种子：{SEED}"
)


# ==================================================
# 3. 加载数据
# ==================================================

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=SEED
)


# ==================================================
# 4. 创建本次实验名称
# ==================================================

run_name = (
    f"{MODEL_NAME}_"
    f"seed{SEED}_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


# ==================================================
# 5. 创建 TensorBoard Writer
# ==================================================

writer = SummaryWriter(
    log_dir=os.path.join(
        "runs",
        run_name
    )
)


print(
    f"TensorBoard日志："
    f"runs/{run_name}"
)


# ==================================================
# 6. 实验超参数
# ==================================================

hparams = {
    "model":
        MODEL_NAME,

    "seed":
        SEED,

    "num_classes":
        NUM_CLASSES,

    "batch_size":
        BATCH_SIZE,

    "num_workers":
        NUM_WORKERS,

    "optimizer":
        "Adam",

    "initial_lr":
        INITIAL_LR,

    "max_epochs":
        NUM_EPOCHS,

    "early_stop_patience":
        EARLY_STOP_PATIENCE,

    "min_delta":
        MIN_DELTA,

    "lr_scheduler":
        "ReduceLROnPlateau",

    "lr_factor":
        LR_FACTOR,

    "lr_patience":
        LR_PATIENCE,

    "min_lr":
        MIN_LR
}


# 把超参数转换成 Markdown 文本
config_text = "\n".join(
    [
        f"- {key}: {value}"
        for key, value in hparams.items()
    ]
)


# 写入 TensorBoard
writer.add_text(
    "Config/Hyperparameters",
    config_text,
    0
)


# ==================================================
# 7. 创建模型
# ==================================================

if MODEL_NAME == "simple_cnn":

    model = SimpleCNN()


elif MODEL_NAME == "resnet18":

    model = get_resnet18(
        num_classes=NUM_CLASSES
    )


else:

    raise ValueError(
        f"未知模型：{MODEL_NAME}"
    )


model = model.to(device)


print(
    f"当前模型：{MODEL_NAME}"
)


# ==================================================
# 8. Loss
# ==================================================

criterion = nn.CrossEntropyLoss()


# ==================================================
# 9. Optimizer
# ==================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)


# ==================================================
# 10. Learning Rate Scheduler
# ==================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,

    # Validation Loss 越小越好
    mode="min",

    # 每次衰减为原来的 0.5
    factor=LR_FACTOR,

    # 连续若干轮没有改善才降低 LR
    patience=LR_PATIENCE,

    # 至少改善 MIN_DELTA 才算有效改善
    threshold=MIN_DELTA,

    threshold_mode="abs",

    # 最低学习率
    min_lr=MIN_LR
)


# ==================================================
# 11. Checkpoint 配置
# ==================================================

best_val_loss = float("inf")


# 最佳 Val Loss 对应模型的 Val Accuracy
best_checkpoint_val_acc = 0.0


# 最佳 checkpoint 所在 epoch
best_epoch = 0


checkpoint_dir = "checkpoints"


os.makedirs(
    checkpoint_dir,
    exist_ok=True
)


checkpoint_path = os.path.join(
    checkpoint_dir,
    f"best_{MODEL_NAME}.pth"
)


# ==================================================
# 12. Early Stopping 配置
# ==================================================

early_stop_counter = 0


# ==================================================
# 13. Training Loop
# ==================================================

for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
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
        # 1. 清空旧梯度
        # --------------------------

        optimizer.zero_grad()


        # --------------------------
        # 2. Forward
        # --------------------------

        outputs = model(images)


        # --------------------------
        # 3. Loss
        # --------------------------

        loss = criterion(
            outputs,
            labels
        )


        # --------------------------
        # 4. Backward
        # --------------------------

        loss.backward()


        # --------------------------
        # 5. 更新参数
        # --------------------------

        optimizer.step()


        # --------------------------
        # 统计训练 Loss
        # --------------------------

        running_loss += loss.item()


        # --------------------------
        # 获取预测结果
        # --------------------------

        _, predicted = outputs.max(
            dim=1
        )


        # 当前 batch 样本总数
        total += labels.size(0)


        # 当前 batch 正确数量
        correct += (
            predicted
            .eq(labels)
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


            # --------------------------
            # Forward
            # --------------------------

            outputs = model(images)


            # --------------------------
            # Validation Loss
            # --------------------------

            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item()
            )


            # --------------------------
            # Prediction
            # --------------------------

            _, predicted = outputs.max(
                dim=1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted
                .eq(labels)
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

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ==================================================
    # 14. TensorBoard Scalars
    # ==================================================

    writer.add_scalar(
        "Loss/Train",
        train_loss,
        epoch + 1
    )


    writer.add_scalar(
        "Loss/Validation",
        val_loss,
        epoch + 1
    )


    writer.add_scalar(
        "Accuracy/Train",
        train_acc,
        epoch + 1
    )


    writer.add_scalar(
        "Accuracy/Validation",
        val_acc,
        epoch + 1
    )


    writer.add_scalar(
        "Learning_Rate",
        current_lr,
        epoch + 1
    )


    # ==================================================
    # 15. 输出结果
    # ==================================================

    print(
        f"Epoch "
        f"[{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: "
        f"{train_loss:.4f} "
        f"Train Acc: "
        f"{train_acc:.2f}% "
        f"Val Loss: "
        f"{val_loss:.4f} "
        f"Val Acc: "
        f"{val_acc:.2f}% "
        f"LR: "
        f"{current_lr:.6f}"
    )


    # ==================================================
    # 16. Learning Rate Scheduler
    # ==================================================

    old_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ReduceLROnPlateau
    # 根据 Validation Loss 调整学习率
    scheduler.step(
        val_loss
    )


    new_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # 如果学习率发生变化
    if new_lr < old_lr:

        print(
            f"学习率降低："
            f"{old_lr:.6f} "
            f"-> "
            f"{new_lr:.6f}"
        )


    # ==================================================
    # 17. Checkpoint + Early Stopping
    # ==================================================

    if (
        val_loss
        <
        best_val_loss - MIN_DELTA
    ):

        # --------------------------
        # Validation 出现改善
        # --------------------------

        best_val_loss = val_loss


        # 记录这个最佳模型对应的 Val Acc
        best_checkpoint_val_acc = (
            val_acc
        )


        # 最佳模型所在 epoch
        best_epoch = (
            epoch + 1
        )


        # Early Stopping 重新计数
        early_stop_counter = 0


        # --------------------------
        # 保存最佳 Checkpoint
        # --------------------------

        torch.save(
            {
                # ------------------
                # 当前 Epoch
                # ------------------

                "epoch":
                    epoch + 1,


                # ------------------
                # 模型参数
                # ------------------

                "model_state_dict":
                    model.state_dict(),


                # ------------------
                # Optimizer 状态
                # ------------------

                "optimizer_state_dict":
                    optimizer.state_dict(),


                # ------------------
                # Scheduler 状态
                # ------------------

                "scheduler_state_dict":
                    scheduler.state_dict(),


                # ------------------
                # 本轮指标
                # ------------------

                "train_loss":
                    train_loss,

                "train_acc":
                    train_acc,

                "val_loss":
                    val_loss,

                "val_acc":
                    val_acc,


                # ------------------
                # 当前最佳指标
                # ------------------

                "best_val_loss":
                    best_val_loss,


                # ------------------
                # 实验配置
                # ------------------

                "hparams":
                    hparams
            },

            checkpoint_path
        )


        print(
            f"保存最佳模型："
            f"Val Loss = "
            f"{best_val_loss:.4f}"
        )


    else:

        # --------------------------
        # Validation 没有明显改善
        # --------------------------

        early_stop_counter += 1


        print(
            f"验证集未改善："
            f"{early_stop_counter}/"
            f"{EARLY_STOP_PATIENCE}"
        )


        # --------------------------
        # Early Stopping
        # --------------------------

        if (
            early_stop_counter
            >=
            EARLY_STOP_PATIENCE
        ):

            print(
                "连续多轮验证集未改善，"
                "触发 Early Stopping。"
            )

            break


# ==================================================
# 18. TensorBoard HParams
# ==================================================

writer.add_hparams(
    hparam_dict=hparams,

    metric_dict={
        "hparam/best_val_loss":
            best_val_loss,

        "hparam/best_checkpoint_val_acc":
            best_checkpoint_val_acc
    }
)


# 关闭 TensorBoard Writer
writer.close()


# ==================================================
# 19. 训练结束
# ==================================================

print(
    "\n训练结束"
)


print(
    f"最佳 Epoch："
    f"{best_epoch}"
)


print(
    f"最佳 Val Loss："
    f"{best_val_loss:.4f}"
)


print(
    f"最佳模型对应 Val Acc："
    f"{best_checkpoint_val_acc:.2f}%"
)


print(
    f"最佳模型保存在："
    f"{checkpoint_path}"
)