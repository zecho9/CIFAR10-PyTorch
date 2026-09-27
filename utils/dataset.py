import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=0
):
    """
    创建 CIFAR-10 的训练、验证、测试 DataLoader

    参数:
        data_dir:
            数据集所在目录

        batch_size:
            每个 batch 的图片数量

        num_workers:
            DataLoader 加载数据使用的子进程数量

    返回:
        train_loader
        val_loader
        test_loader
    """

    # =========================
    # 1. 数据预处理
    # =========================

    # 训练集：
    # 使用数据增强，提高模型泛化能力
    train_transform = transforms.Compose([

        # 随机水平翻转
        transforms.RandomHorizontalFlip(),

        # 先padding，再随机裁剪为32×32
        transforms.RandomCrop(
            32,
            padding=4
        ),

        # PIL Image -> Tensor
        transforms.ToTensor(),

        # CIFAR-10标准化
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])


    # 验证集和测试集：
    # 不进行随机数据增强
    eval_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])


    # =========================
    # 2. 创建 Dataset
    # =========================

    # 完整训练集：
    # 使用训练数据增强
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )


    # 验证集使用同样的原始50000张训练图片，
    # 但是不使用随机数据增强
    full_val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform
    )


    # 官方测试集10000张
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=eval_transform
    )
    # =========================
    # 3. 划分 Train / Validation
    # =========================

    num_samples = len(full_train_dataset)

    num_train = 45000
    num_val = 5000


    generator = torch.Generator().manual_seed(42)


    indices = torch.randperm(
        num_samples,
        generator=generator
    ).tolist()


    train_indices = indices[:num_train]

    val_indices = indices[
        num_train:num_train + num_val
    ]


    # 这里非常关键
    train_dataset = Subset(
        full_train_dataset,
        train_indices
    )

    val_dataset = Subset(
        full_val_dataset,
        val_indices
    )


    # =========================
    # 4. DataLoader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    return train_loader, val_loader, test_loader