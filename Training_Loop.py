import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from model import DeepFakeDetector

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 10   # reduced for speed
IMAGE_SIZE = 112

config = {
    'batch_size': 4,
    'epochs': 10,       # you can increase later
    'learning_rate': 1e-4,
    'weight_decay': 1e-4
}

# =========================
# DATASET
# =========================
class CombinedDataset(Dataset):
    def __init__(self, real_folder, fake_folder):
        self.paths = list(Path(real_folder).glob("*.mp4")) + \
                     list(Path(fake_folder).glob("*.mp4"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        label = 0 if "real" in path.lower() else 1

        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        # Padding if video is short
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label)

# =========================
# TRAIN FUNCTION
# =========================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeepFakeDetector().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # =========================
    # LOAD DATA
    # =========================
    train_dataset = CombinedDataset(
        "data/processed/train_real_faces",
        "data/processed/train_fake_faces"
    )

    val_dataset = CombinedDataset(
        "data/processed/test_real_faces",
        "data/processed/test_fake_faces"
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    best_auc = 0.0

    # =========================
    # TRAIN LOOP
    # =========================
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

        # ✅ FIXED LOSS (AVERAGE)
        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)

                outputs = model(frames)
                probs = torch.softmax(outputs, dim=1)

                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)

        # ✅ SAFE AUC (no crash)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        print(f"Val Acc={acc:.4f}, AUC={auc:.4f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"models/best_model_epoch_{epoch+1}.pth")
            print("✅ Best model saved!")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()