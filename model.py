import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        
        # CNN Backbone
        self.cnn = models.resnext50_32x4d(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze CNN layers
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Unfreeze last layers
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
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        cnn_features = []
        for t in range(seq_len):
            frame_features = self.cnn(x[:, t])
            frame_features = frame_features.view(batch_size, -1)
            cnn_features.append(frame_features)
        
        cnn_seq = torch.stack(cnn_features, dim=1)
        
        lstm_out, (hidden, _) = self.lstm(cnn_seq)
        
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        
        return out