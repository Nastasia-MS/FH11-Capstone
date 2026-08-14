import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """1D CNN for signal classification with adaptive pooling."""
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=9, padding=4)
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
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=5, padding=2)
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
    """Multi-layer perceptron.

    ``forward`` flattens every channel into one vector, so the first layer has
    to be sized ``in_channels * input_size``.  It used to take ``input_size``
    alone, which meant any 2-channel I/Q or multi-antenna dataset failed with
    "mat1 and mat2 shapes cannot be multiplied" on the first batch.
    """
    def __init__(self, input_size=256, num_classes=2, in_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    """1D residual block for ResNet."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection adjustment
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class ResNet1DOptimized(nn.Module):
    """ResNet1D optimized for IQ signal classification (2-channel input)."""
    def __init__(self, num_classes=2, in_channels=2, base_filters=64, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        
        # Initial convolution
        self.conv_initial = nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_initial = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool_initial = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with increasing depth
        self.layer1 = self._make_layer(base_filters, base_filters, num_blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, num_blocks=2, stride=2, dropout=dropout)
        
        # Global average pooling and classification head
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_initial(self.conv_initial(x)))
        x = self.pool_initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(name, num_classes=2, input_size=256, in_channels=1, **kwargs):
    """Return a PyTorch model by name.

    Args:
        name: Model architecture name ('SimpleCNN', 'TinyConv', 'MLP', 'ResNet1DOptimized').
        num_classes: Number of output classes.
        input_size: Input length (for MLP).
        in_channels: Number of input channels for Conv1d models.
            Defaults to 1 for single-channel; set to num_rx_ant for multi-channel or 2 for IQ.
        **kwargs: Model-specific hyperparameters:
            - base_filters: For ResNet1DOptimized (default 64)
            - dropout: For ResNet1DOptimized (default 0.2)
    """
    if name == 'SimpleCNN':
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    elif name == 'TinyConv':
        return TinyConv(num_classes=num_classes, in_channels=in_channels)
    elif name == 'MLP':
        return MLP(input_size=input_size, num_classes=num_classes,
                   in_channels=in_channels)
    elif name == 'ResNet1DOptimized':
        base_filters = int(kwargs.get('base_filters', 64))
        dropout = float(kwargs.get('dropout', 0.2))
        return ResNet1DOptimized(num_classes=num_classes, in_channels=in_channels,
                                 base_filters=base_filters, dropout=dropout)
    else:
        # Default fallback
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

