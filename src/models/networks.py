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


class FashionCNN_NoPool(nn.Module):
    """FashionCNN_Small with the max-pools replaced by stride-2 convolutions.

    Same tensor shapes and the same two conv / two FC structure, but no pooling.
    Two reasons, both measured on `FashionCNN_Small` (see
    results/audit/FINDINGS_FOR_JIRI.md):

    1. Max-pool creates EXACT ties between equal post-ReLU values, so x0 lands on
       several hundred genuine faces of its own linearity region. The cell that
       PyTorch's argmax tie-break then selects is arbitrary, which makes Xi_x --
       and hence gamma -- convention-dependent. Striding removes ties entirely:
       the region is cut by ReLU hyperplanes only.
    2. Constraint count. The pooled net has 11 856 ReLU neurons (7 840 + 3 920 +
       96); this one has 3 036 (1 960 + 980 + 96), against the MLP's 1 920. The
       Chebyshev LP on the pooled net does not converge in 1800 s with the dual
       simplex, where the MLP solves in 92 s.

    Downsampling is done by the feature convolutions themselves rather than by
    extra layers, so no ReLU is added (all-convolutional net, Springenberg et al.
    2015). `_collect_conv2d` propagates shortcut weights through
    `conv._conv_forward`, which honours stride, so the polytope builder needs no
    change.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, stride=2, padding=1)    # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(10, 20, 3, stride=2, padding=1)   # 14x14 ->  7x7
        self.fc1 = nn.Linear(20 * 7 * 7, 96)
        self.fc2 = nn.Linear(96, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
