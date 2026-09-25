import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(
                in_channels = 3,
                out_channels= 32,
                kernel_size = 3,
                padding=1),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding=1
                ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
                )
            )


        self.classifier = nn.Sequential(

            nn.Linear(
                64*8*8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self,x):
            x = self.feature(x)
            x = torch.flatten(
                x,
                start_dim = 1
            )
            x = self.classifier(x)

            return x