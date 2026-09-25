import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from utils.dataset import get_dataloaders

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("使用设备：",device)

train_loader,test_loader = get_dataloaders(
    batch_size=128,
    num_workers=0
)
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 5
for epoch in range(num_epochs):
    print(
        f"Epoch{epoch+1}/{num_epochs}"
    )
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images , labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(
            outputs,
            labels
        )
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
        _, predicted = outputs.max(dim=1)
        total +=labels.size(0)
        correct += (
            predicted.eq(labels)
            .sum()
            .item()
        )
    epoch_loss = (
        running_loss /
        len(train_loader)
    )
    epoch_acc =(
        100.0 *
        correct / 
        total
    )
    print(
        f"Epoch[{epoch + 1}/{num_epochs}]"
        f"Loss:{epoch_loss:.4f}"
        f"Accuracy:{epoch_acc :.2f}%"
    )
print("训练完成")

