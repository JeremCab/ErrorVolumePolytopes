"""Simple neural network models (PyTorch).

Contains a small MLP and a small CNN for quick experiments on MNIST.
"""
import torch.nn as nn
import torch.nn.functional as F

class SmallMLP(nn.Module):
    """3-layer MLP for flattened MNIST (784 → 64 → 32 → 10)."""
    def __init__(self, input_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MLP(nn.Module):
    """3-layer MLP for flattened MNIST (784 → 2000 → 1000 → 10)."""
    def __init__(self, input_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SmallCNN(nn.Module):
    """Tiny CNN for MNIST-style images."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



# ==================== #
# Fashion-MNIST models #
# ==================== #


class FashionMLP_Large(nn.Module):
    """5-layer MLP for Fashion-MNIST (784 → 1024 → 512 → 256 → 128 → 10)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)
    
    
# class FashionCNN_Small(nn.Module):
#     """Small CNN for Fashion-MNIST: two conv blocks followed by two FC layers."""
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

#         self.pool = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)

#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)

#         x = F.relu(self.conv2(x))
#         x = self.pool(x)

#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x
    

class FashionCNN_Small(nn.Module):
    """CNN with ~15K neurons for Fashion-MNIST."""
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(20 * 7 * 7, 96)
        self.fc2 = nn.Linear(96, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# class FashionCNN_Small(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(10)

#         self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(20)

#         self.pool = nn.MaxPool2d(2)

#         self.fc1 = nn.Linear(20 * 7 * 7, 64)
#         self.fc2 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)

#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)