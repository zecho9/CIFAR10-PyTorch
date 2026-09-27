import torch.nn as nn

from torchvision.models import resnet18


def get_resnet18(num_classes=10):

    # =========================
    # 1. 创建 ResNet-18
    # =========================

    # weights=None：
    # 不加载 ImageNet 预训练参数，
    # 从随机初始化开始训练
    model = resnet18(
        weights=None
    )


    # =========================
    # 2. 修改第一层卷积
    # =========================

    # torchvision 原始 ResNet：
    #
    # 7×7 Conv
    # stride=2
    #
    # 适合 224×224 ImageNet 图像
    #
    # CIFAR-10 只有 32×32，
    # 所以改成 3×3、stride=1

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )


    # =========================
    # 3. 删除原来的 MaxPool
    # =========================

    model.maxpool = nn.Identity()


    # =========================
    # 4. 修改分类层
    # =========================

    # ResNet 原版：
    # 512 -> 1000
    #
    # CIFAR-10：
    # 512 -> 10

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )


    return model