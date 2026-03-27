import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """1D CNN for signal classification with adaptive pooling."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        self.fc1 = nn.Linear(32 * 64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyConv(nn.Module):
    """Minimal 1D CNN with adaptive pooling."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(8 * 32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, input_size=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(name, num_classes=2, input_size=256):
    """Return a PyTorch model by name."""
    if name == 'SimpleCNN':
        return SimpleCNN(num_classes=num_classes)
    elif name == 'TinyConv':
        return TinyConv(num_classes=num_classes)
    elif name == 'MLP':
        return MLP(input_size=input_size, num_classes=num_classes)
    else:
        return SimpleCNN(num_classes=num_classes)

