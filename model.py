# =========================
# model.py
# =========================

import torch
import torch.nn as nn
import torchvision.models as models


class DeepFakeDetector(nn.Module):

    def __init__(self,
                 hidden_size=64,
                 num_layers=1,
                 num_classes=2):

        super(DeepFakeDetector, self).__init__()

        # ====================================
        # CNN BACKBONE
        # ====================================

        backbone = models.resnext50_32x4d(weights="DEFAULT")

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Freeze CNN initially
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        for param in list(self.cnn.parameters())[-10:]:
            param.requires_grad = True

        # ====================================
        # LSTM
        # ====================================

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # ====================================
        # CLASSIFIER
        # ====================================

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_size, num_classes)

    # ====================================
    # FEATURE EXTRACTOR FOR GRAD-CAM
    # ====================================

    def forward_features(self, x):

        x = self.cnn(x)

        x = x.view(x.size(0), -1)

        return x

    # ====================================
    # MAIN FORWARD
    # ====================================

    def forward(self, x):

        # x shape:
        # (batch, seq, c, h, w)

        batch_size, seq_len, c, h, w = x.shape

        features = []

        for t in range(seq_len):

            frame = x[:, t, :, :, :]

            feat = self.forward_features(frame)

            features.append(feat)

        features = torch.stack(features, dim=1)

        lstm_out, (hidden, cell) = self.lstm(features)

        out = self.dropout(hidden[-1])

        out = self.fc(out)

        return out