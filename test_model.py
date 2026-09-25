import torch
from models.cnn import SimpleCNN

model = SimpleCNN()

x = torch.randn(
    128,
    3,
    32,
    32
)
output = model(x)

print("输入:",x.shape)

print("输出:",output.shape)
total_params = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "参数数量:",
)