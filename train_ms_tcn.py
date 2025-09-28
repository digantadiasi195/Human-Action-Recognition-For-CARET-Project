import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 52  # 51 actions + background (class 0)
IN_FEATURES = 150  # 2 persons × 25 joints × 3 coords
MODEL_SAVE_PATH = "models/ms_tcn_pku.pth"
DATA_ROOT = r"E:/CARET_Project/CARET/PKUMMDv2"
SPLIT_TYPE = "cross-subject"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# === Data Loading ===

def load_skeleton(path):
    """Load skeleton data (150D per frame)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skeleton file not found: {path}")
    data = []
    with open(path, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            if len(coords) != 150:
                coords = (coords + [0.0] * 150)[:150]
            data.append(coords)
    return np.array(data, dtype=np.float32)


def create_frame_labels(label_path, total_frames):
    """Create per-frame labels from PKU-MMD annotation."""
    labels = np.zeros(total_frames, dtype=np.int64)
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                action_id = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                if 1 <= action_id <= 51 and 0 <= start <= end < total_frames:
                    labels[start:end+1] = action_id
            except ValueError:
                continue
    return labels


def normalize_skeleton(skel, mean=None, std=None):
    """Normalize per joint across frames."""
    T = skel.shape[0]
    data = skel.reshape(T, 2, 25, 3)
    if mean is None:
        mean = data.mean(axis=(0, 1, 3), keepdims=True)
        std = data.std(axis=(0, 1, 3), keepdims=True) + 1e-8
    data = (data - mean) / std
    return data.reshape(T, 150), mean, std


def parse_split_file(path):
    """Parse train/val video IDs from split file."""
    with open(path, 'r') as f:
        content = f.read()
    train_match = re.search(r'Training videos:\s*\n(.*?)(?:\n\n|\nValidataion)', content, re.DOTALL)
    val_match = re.search(r'Validataion videos:\s*\n(.*?)$', content, re.DOTALL)
    train_ids = [v.strip() for v in train_match.group(1).split(',')] if train_match else []
    val_ids = [v.strip() for v in val_match.group(1).split(',')] if val_match else []
    return train_ids, val_ids


# === Dataset ===

class PKUMMDDataset(Dataset):
    def __init__(self, video_ids, root, mean=None, std=None):
        self.root = root
        self.mean, self.std = mean, std
        self.skeletons, self.labels = [], []
        for vid in video_ids:
            try:
                skel = load_skeleton(os.path.join(root, "Data", "Skeleton", f"{vid}.txt"))
                skel, self.mean, self.std = normalize_skeleton(skel, self.mean, self.std)
                lbl = create_frame_labels(os.path.join(root, "Label", f"{vid}.txt"), len(skel))
                self.skeletons.append(torch.tensor(skel, dtype=torch.float32))
                self.labels.append(torch.tensor(lbl, dtype=torch.long))
            except Exception as e:
                print(f"Skipped {vid}: {e}")

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        return self.skeletons[idx], self.labels[idx]


# === Model ===

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_ch, out_ch):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(dim, num_f_maps, 1),
            nn.ReLU(),
            *[DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)]
        )
        self.stages = nn.ModuleList([
            nn.Sequential(*[DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)])
            for _ in range(num_stages - 1)
        ])
        self.classifier = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
        out = self.stage1(x)
        for stage in self.stages:
            out = stage(out) + out
        out = self.classifier(out)
        return out.permute(0, 2, 1)  # (B, T, C)


# === Metrics ===

def compute_metrics(y_true, y_pred, num_classes):
    mask = y_true != -100
    y_true, y_pred = y_true[mask], y_pred[mask]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, f1, cm


def plot_cm(cm, epoch):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"logs/confusion_matrix_epoch_{epoch}.png")
    plt.close()


# === Training ===

def train():
    train_ids, val_ids = parse_split_file(os.path.join(DATA_ROOT, "Split", f"{SPLIT_TYPE}.txt"))
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_ds = PKUMMDDataset(train_ids, DATA_ROOT)
    val_ds = PKUMMDDataset(val_ids, DATA_ROOT, train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MS_TCN(4, 10, 64, IN_FEATURES, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for skel, lbl in train_loader:
            skel, lbl = skel.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            out = model(skel)
            loss = criterion(out.view(-1, NUM_CLASSES), lbl.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f}")

        # Validate
        model.eval()
        val_loss, all_true, all_pred = 0, [], []
        with torch.no_grad():
            for skel, lbl in val_loader:
                skel, lbl = skel.to(DEVICE), lbl.to(DEVICE)
                out = model(skel)
                val_loss += criterion(out.view(-1, NUM_CLASSES), lbl.view(-1)).item()
                pred = out.argmax(dim=2)
                all_true.append(lbl.cpu().numpy().flatten())
                all_pred.append(pred.cpu().numpy().flatten())

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        acc, f1, cm = compute_metrics(all_true, all_pred, NUM_CLASSES)
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        scheduler.step(val_loss)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': train_ds.mean,
                'std': train_ds.std
            }, MODEL_SAVE_PATH)
            print("→ Best model saved")

        if (epoch + 1) % 5 == 0:
            plot_cm(cm, epoch + 1)

    np.save("models/joint_mean.npy", train_ds.mean)
    np.save("models/joint_std.npy", train_ds.std)
    print(f"Training finished. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
