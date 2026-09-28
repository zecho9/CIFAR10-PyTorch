import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=0,
        seed=42
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

        seed:
            随机种子，用于固定 Train / Validation 划分
            以及训练集 DataLoader 的 shuffle 顺序

    返回:
        train_loader
        val_loader
        test_loader
    """

    # =========================
    # 1. 数据预处理
    # =========================

    # 训练集：
    # 使用随机数据增强
    train_transform = transforms.Compose([

        # 随机水平翻转
        transforms.RandomHorizontalFlip(),

        # 先 padding，再随机裁剪为 32×32
        transforms.RandomCrop(
            32,
            padding=4
        ),

        # PIL Image -> Tensor
        transforms.ToTensor(),

        # CIFAR-10 标准化
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


    # 验证集与训练集来自相同的原始 50000 张图片，
    # 但验证集不使用随机数据增强
    full_val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform
    )


    # CIFAR-10 官方测试集
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

    # CIFAR-10 原训练集共 50000 张
    num_val = 5000
    num_train = num_samples - num_val


    # 专门用于数据集划分的随机数生成器
    #
    # 这样相同 seed 下：
    # 每次运行得到完全相同的
    # Train / Validation 索引
    split_generator = torch.Generator().manual_seed(
        seed
    )


    # 随机打乱 0 ~ 49999
    indices = torch.randperm(
        num_samples,
        generator=split_generator
    ).tolist()


    # 前 45000 个作为训练集
    train_indices = indices[:num_train]


    # 后 5000 个作为验证集
    val_indices = indices[num_train:]


    # =========================
    # 4. 创建 Subset
    # =========================

    train_dataset = Subset(
        full_train_dataset,
        train_indices
    )


    val_dataset = Subset(
        full_val_dataset,
        val_indices
    )


    # =========================
    # 5. 固定 DataLoader Shuffle
    # =========================

    # 专门用于训练 DataLoader shuffle
    #
    # 与 split_generator 分开，
    # 避免数据划分过程改变 shuffle 的随机状态
    train_generator = torch.Generator().manual_seed(
        seed
    )


    # =========================
    # 6. 创建 DataLoader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,

        # 训练时打乱样本
        shuffle=True,

        num_workers=num_workers,

        # 固定 shuffle 随机性
        generator=train_generator
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,

        # 验证时不打乱
        shuffle=False,

        num_workers=num_workers
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,

        # 测试时不打乱
        shuffle=False,

        num_workers=num_workers
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )