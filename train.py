import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class DeepFakeDetector(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()

        self.cnn = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        for name, param in self.cnn.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch, seq, c, h, w = x.shape

        x = x.view(batch * seq, c, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq, -1)

        _, (hidden, _) = self.lstm(features)

        out = self.dropout(hidden[-1])
        out = self.fc(out)

        return out