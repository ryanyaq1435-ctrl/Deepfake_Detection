import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, num_classes=2):
        super(DeepFakeDetector, self).__init__()

        # CNN Backbone
        self.cnn = models.resnext50_32x4d(weights="DEFAULT")
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        # Freeze most layers
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Unfreeze last few layers
        for param in list(self.cnn.parameters())[-4:]:
            param.requires_grad = True

        # LSTM
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Classifier
        self.dropout = nn.Dropout(0.5)   # FIXED
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch, seq, c, h, w = x.shape

        features = []
        for t in range(seq):
            f = self.cnn(x[:, t])
            f = f.view(batch, -1)
            features.append(f)

        x = torch.stack(features, dim=1)

        _, (hidden, _) = self.lstm(x)

        out = self.dropout(hidden[-1])
        out = self.fc(out)

        return out