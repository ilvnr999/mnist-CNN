import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.cf1 = nn.Linear(32*4*4, 10)

    def forward(self, X):
        a = self.cnn1(X)
        a = F.relu(a)
        a = F.max_pool2d(a, kernel_size=2)

        a = self.cnn2(a)
        a = F.relu(a)
        a = F.max_pool2d(a, 2)

        a = torch.flatten(a, start_dim=1) # 保留batch那個維度 從1維度開始攤平

        out = self.cf1(a)
        return out