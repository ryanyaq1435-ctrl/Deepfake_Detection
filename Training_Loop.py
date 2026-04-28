from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# DATASET CLASS
# =========================
class VideoDataset(Dataset):
    def __init__(self, data_path, sequence_length=20):
        # Load videos from all subfolders
        self.video_paths = list(Path(data_path).rglob("*.mp4"))
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = str(self.video_paths[idx])

        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (112, 112))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        # Padding if video is short
        while len(frames) < self.sequence_length:
            frames.append(np.zeros((112, 112, 3)))

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (seq, C, H, W)

        # Label from folder name
        if "fake" in video_path.lower():
            label = 1
        else:
            label = 0

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_loader, val_loader, config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    best_auc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)

                outputs = model(frames)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)

        # AUC needs probabilities
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0

        print(f"Epoch {epoch+1}: Val Acc={acc:.4f}, AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"models/best_model_epoch_{epoch+1}.pth")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    from model import DeepFakeDetector

    config = {
        'epochs': 3,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }

    train_dataset = VideoDataset("data/processed/")
    val_dataset = VideoDataset("data/processed/")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = DeepFakeDetector()

    train_model(model, train_loader, val_loader, config)