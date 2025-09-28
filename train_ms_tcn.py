#train_ms_tcn.py
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

# CONFIGURATION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1          # Videos are long → batch_size=1
NUM_EPOCHS = 20         # Increased for better training
LEARNING_RATE = 0.001
NUM_CLASSES = 52        # 51 actions + 1 background (class 0)
IN_FEATURES = 150       # 2 persons × 25 joints × 3 coordinates
MODEL_SAVE_PATH = "models/ms_tcn_pku.pth"
DATA_ROOT = r"E:/CARET_Project/CARET/PKUMMDv2"
SPLIT_TYPE = "cross-subject"  # or "cross-view"
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# DATA LOADING UTILITIES

def load_skeleton(file_path):
    """Load skeleton data from .txt file (150 floats per line)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Skeleton file not found: {file_path}")
    
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            coords = list(map(float, line.strip().split()))
            if len(coords) != 150:
                coords = (coords + [0.0] * 150)[:150]
            data.append(coords)
    return np.array(data, dtype=np.float32)

def create_frame_labels(label_path, total_frames):
    """
    Load PKU-MMD label file (.txt) with format: action_id,start_frame,end_frame,confidence
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    frame_labels = np.zeros(total_frames, dtype=np.int64)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
                
            try:
                action_id = int(parts[0].strip())
                start_frame = int(parts[1].strip())
                end_frame = int(parts[2].strip())
                
                # Validate action_id and frame range
                if 1 <= action_id <= 51 and 0 <= start_frame <= end_frame < total_frames:
                    frame_labels[start_frame : end_frame + 1] = action_id  # +1 because end is inclusive
            except ValueError:
                continue
    
    return frame_labels

def normalize_skeleton_per_joint(skeleton_data, joint_mean=None, joint_std=None):
    """Normalize skeleton data per joint across all frames"""
    T = skeleton_data.shape[0]
    data_4d = skeleton_data.reshape(T, 2, 25, 3)  # (T, 2, 25, 3)
    
    if joint_mean is None:
        joint_mean = np.mean(data_4d, axis=(0,1,3), keepdims=True)  # Shape: (1,1,25,1)
    if joint_std is None:
        joint_std = np.std(data_4d, axis=(0,1,3), keepdims=True) + 1e-8  # Avoid div by zero
    
    data_4d = (data_4d - joint_mean) / joint_std
    return data_4d.reshape(T, 150), joint_mean, joint_std

def parse_split_file(split_file_path):
    """Parse a split file to extract training and validation video IDs"""
    with open(split_file_path, 'r') as f:
        content = f.read()
    
    # Extract training videos
    train_match = re.search(r'Training videos:\s*\n(.*?)(?:\n\n|\nValidataion videos:|$)', content, re.DOTALL)
    if train_match:
        train_str = train_match.group(1).strip()
        train_videos = [v.strip() for v in train_str.split(',') if v.strip()]
    else:
        train_videos = []
    
    # Extract validation videos
    val_match = re.search(r'Validataion videos:\s*\n(.*?)(?:\n\n|$)', content, re.DOTALL)
    if val_match:
        val_str = val_match.group(1).strip()
        val_videos = [v.strip() for v in val_str.split(',') if v.strip()]
    else:
        val_videos = []
    
    return train_videos, val_videos

# ========================
# DATASET CLASS
# ========================
class PKUMMDDataset(Dataset):
    def __init__(self, video_ids, data_root, joint_mean=None, joint_std=None):
        self.video_ids = video_ids
        self.data_root = data_root
        self.skeletons = []
        self.labels = []
        self.lengths = []
        self.joint_mean = joint_mean
        self.joint_std = joint_std
        self._load_data()
    
    def _load_data(self):
        print(f"Loading {len(self.video_ids)} videos...")
        loaded_count = 0
        
        for vid in self.video_ids:
            try:
                # Load skeleton
                skel_path = os.path.join(self.data_root, "Data", "Skeleton", f"{vid}.txt")
                skeleton = load_skeleton(skel_path)
                
                # Normalize
                if self.joint_mean is not None and self.joint_std is not None:
                    skeleton, _, _ = normalize_skeleton_per_joint(skeleton, self.joint_mean, self.joint_std)
                else:
                    skeleton, self.joint_mean, self.joint_std = normalize_skeleton_per_joint(skeleton)
                
                # Load labels
                label_path = os.path.join(self.data_root, "Label", f"{vid}.txt")
                frame_labels = create_frame_labels(label_path, len(skeleton))
                
                # Store
                self.skeletons.append(torch.tensor(skeleton, dtype=torch.float32))
                self.labels.append(torch.tensor(frame_labels, dtype=torch.long))
                self.lengths.append(len(skeleton))
                
                loaded_count += 1
                if loaded_count % 50 == 0 or loaded_count == len(self.video_ids):
                    print(f"  → Loaded {loaded_count}/{len(self.video_ids)} videos")
                    
            except Exception as e:
                print(f"  ✗ Failed to load {vid}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.skeletons)} videos")
    
    def __len__(self):
        return len(self.skeletons)
    
    def __getitem__(self, idx):
        return self.skeletons[idx], self.labels[idx], self.lengths[idx]

# MS-TCN MODEL
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # Residual connection

class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, features_dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(features_dim, num_f_maps, 1),
            nn.ReLU(),
            *[DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)]
        )
        self.stages = nn.ModuleList([
            nn.Sequential(
                *[DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)]
            ) for _ in range(num_stages - 1)
        ])
        self.classifier = nn.Conv1d(num_f_maps, num_classes, 1)
    
    def forward(self, x):  # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        out = self.stage1(x)
        for stage in self.stages:
            out = stage(out) + out  # Residual
        out = self.classifier(out)
        return out.permute(0, 2, 1)  # (B, T, C)


# EVALUATION METRICS
def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate accuracy, F1-score, and confusion matrix"""
    # Filter out padding
    mask = y_true != -100  # Assuming -100 is padding value
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(range(num_classes)))
    
    return accuracy, f1, cm

def plot_confusion_matrix(cm, classes, epoch, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# TRAINING FUNCTION
def train_model():
    split_file = os.path.join(DATA_ROOT, "Split", f"{SPLIT_TYPE}.txt")
    train_ids, val_ids = parse_split_file(split_file)
    
    print(f"Using {SPLIT_TYPE} split:")
    print(f"Training videos: {len(train_ids)}")
    print(f"Validation videos: {len(val_ids)}")
    
    if len(train_ids) == 0:
        raise ValueError("No training videos found!")
    if len(val_ids) == 0:
        raise ValueError("No validation videos found!")
    
    # Create datasets
    train_dataset = PKUMMDDataset(train_ids, DATA_ROOT)
    val_dataset = PKUMMDDataset(val_ids, DATA_ROOT, train_dataset.joint_mean, train_dataset.joint_std)
    
    if len(train_dataset) == 0:
        raise ValueError("No training data loaded!")
    if len(val_dataset) == 0:
        raise ValueError("No validation data loaded!")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MS_TCN(
        num_stages=4,
        num_layers=10,
        num_f_maps=64,
        features_dim=IN_FEATURES,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    print(f"Model initialized on {DEVICE}")
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (skeletons, labels, lengths) in enumerate(train_loader):
            skeletons = skeletons.to(DEVICE)  # (B, T, 150)
            labels = labels.to(DEVICE)       # (B, T)
            
            optimizer.zero_grad()
            outputs = model(skeletons)       # (B, T, 52)
            
            # Flatten for loss calculation
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        avg_train_loss = total_loss / batch_count
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for skeletons, labels, lengths in val_loader:
                skeletons = skeletons.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(skeletons)
                loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
                val_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs, 2)
                
                # Collect for metrics
                all_y_true.extend(labels.cpu().numpy().flatten())
                all_y_pred.extend(predicted.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy, f1, cm = calculate_metrics(
            np.array(all_y_true), 
            np.array(all_y_pred), 
            NUM_CLASSES
        )
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Learning rate scheduling with manual printing
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"  → Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'f1': f1,
                'joint_mean': train_dataset.joint_mean,
                'joint_std': train_dataset.joint_std
            }, MODEL_SAVE_PATH)
            print(f"  → New best model saved with F1: {f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Plot confusion matrix
        if (epoch + 1) % 5 == 0:
            plot_confusion_matrix(
                cm, 
                list(range(NUM_CLASSES)), 
                epoch+1, 
                f"logs/confusion_matrix_epoch_{epoch+1}.png"
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save normalization parameters
    np.save("models/joint_mean.npy", train_dataset.joint_mean)
    np.save("models/joint_std.npy", train_dataset.joint_std)
    print("Normalization parameters saved.")
    
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()