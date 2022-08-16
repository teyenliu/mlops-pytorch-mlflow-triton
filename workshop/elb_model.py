import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ A simple MLP network for classification on CIFAR-10.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
