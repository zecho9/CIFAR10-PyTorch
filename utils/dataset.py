import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=0
):
    """
    创建 CIFAR-10 DataLoader

    参数:
        data_dir:
            数据集路径

        batch_size:
            每个batch图片数量

        num_workers:
            数据加载进程数量

    返回:
        train_loader
        test_loader
    """


    # 1. 数据增强 + 预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])



 
    # 2. Dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )


    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )



    # 3. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        # 测试不需要打乱
        shuffle=False,
        num_workers=num_workers
    )


    return train_loader, test_loader