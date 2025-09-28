import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd

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
        x = x.permute(0, 2, 1)
        out = self.stage1(x)
        for stage in self.stages:
            out = stage(out) + out
        out = self.classifier(out)
        return out.permute(0, 2, 1)


# === Config ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/ms_tcn_pku.pth"
DATA_ROOT = r"E:/CARET_Project/CARET/PKUMMDv2"
NUM_CLASSES = 52
IN_FEATURES = 150
WINDOW_SIZE = 15
CONF_THRES = 0.15
MIN_DURATION = 3

SKELETON_CONN = [
    (0,1),(1,20),(20,2),(2,3),
    (20,4),(4,5),(5,6),(6,7),(7,21),(21,22),
    (20,8),(8,9),(9,10),(10,11),(11,23),(23,24),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19)
]


# === Utils ===

def load_skeleton(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            if len(coords) != 150:
                coords = (coords + [0.0]*150)[:150]
            data.append(coords)
    return np.array(data, dtype=np.float32)


def normalize_skel(skel, mean, std):
    T = skel.shape[0]
    data = skel.reshape(T, 2, 25, 3)
    data = (data - mean) / std
    return data.reshape(T, 150)


def draw_skeleton(frame, skel, idx, mean, std):
    if idx >= len(skel):
        return frame
    h, w = frame.shape[:2]
    data = skel[idx:idx+1].reshape(1, 2, 25, 3)
    data = data * std + mean
    joints = data.reshape(2, 25, 3)
    
    colors = [(0,0,255), (255,0,0)]
    for p in range(2):
        pts = []
        for j in range(25):
            x, y = joints[p, j, :2]
            x = int((x + 1) * w / 2)
            y = int((1 - y) * h / 2)
            pts.append((x, y))
            cv2.circle(frame, (x, y), 3, colors[p], -1)
        for (i, j) in SKELETON_CONN:
            if i < 25 and j < 25:
                cv2.line(frame, pts[i], pts[j], colors[p], 2)
    return frame


def create_action_panel(current_id, frame_cnt, action_names, action_colors):
    panel = np.ones((900, 900, 3), dtype=np.uint8) * 30
    cv2.putText(panel, "Action Recognition", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    
    # Color ID grid
    box_size, margin = 30, 4
    for aid in range(1, 52):
        row, col = (aid-1)//13, (aid-1)%13
        x = 50 + col * (box_size + margin)
        y = 100 + row * (box_size + margin)
        color = action_colors[aid]
        if aid == current_id:
            color = tuple(min(255, int(c*1.5)) for c in color)
            cv2.rectangle(panel, (x-2, y-2), (x+box_size+2, y+box_size+2), color, 2)
        cv2.rectangle(panel, (x, y), (x+box_size, y+box_size), color, -1)
        cv2.putText(panel, f"{aid}", (x+5, y+box_size-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    # Current action info
    if current_id > 0:
        name = action_names[current_id][:25]
        cv2.putText(panel, f"ID: {current_id}", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, action_colors[current_id], 2)
        cv2.putText(panel, f"Action: {name}", (50, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, action_colors[current_id], 2)
    else:
        cv2.putText(panel, "Background", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,128,128), 2)
    
    cv2.putText(panel, f"Frame: {frame_cnt}", (50, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
    return panel


# === Main ===

def main():
    # Load model
    model = MS_TCN(4, 10, 64, IN_FEATURES, NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) and 'model_state_dict' not in ckpt else ckpt.get('model_state_dict', ckpt))
    model.eval()
    
    joint_mean = np.load("models/joint_mean.npy")
    joint_std = np.load("models/joint_std.npy")
    
    # Load action names
    df = pd.read_excel(os.path.join(DATA_ROOT, "Actions.xlsx"))
    action_names = {int(r['Label']): r['Action'] for _, r in df.iterrows()}
    action_names[0] = "Background"
    action_colors = {i: tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[i*137.5 % 360, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    ) for i in range(52)}
    action_colors[0] = (128, 128, 128)
    
    # Select video
    skel_files = [f for f in os.listdir(os.path.join(DATA_ROOT, "Data", "Skeleton")) if f.endswith('.txt')]
    video_ids = [f.replace('.txt', '') for f in skel_files]
    print("Available videos:", video_ids[:5])
    vid = input("Enter video ID (e.g., '0002-M'): ").strip()
    if vid not in video_ids:
        print("Video not found")
        return
    
    # Process video
    skel = load_skeleton(os.path.join(DATA_ROOT, "Data", "Skeleton", f"{vid}.txt"))
    skel = normalize_skel(skel, joint_mean, joint_std)
    
    cap = cv2.VideoCapture(os.path.join(DATA_ROOT, "Data", "RGB_VIDEO", f"{vid}.avi"))
    frame_buffer = []
    last_action, duration = 0, 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read() if cap.isOpened() else (True, np.zeros((480, 640, 3), dtype=np.uint8))
        if not ret:
            break
        
        # Resize frame
        h, w = frame.shape[:2]
        scale = 700 / w
        frame = cv2.resize(frame, (700, int(h*scale)))
        
        # Draw skeleton
        frame = draw_skeleton(frame, skel, frame_idx, joint_mean, joint_std)
        
        # Inference
        if frame_idx < len(skel):
            frame_buffer.append(skel[frame_idx])
            if len(frame_buffer) > WINDOW_SIZE:
                frame_buffer.pop(0)
            
            if len(frame_buffer) == WINDOW_SIZE:
                inp = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(inp)
                    probs = F.softmax(out[0, -1], dim=-1).cpu().numpy()
                    pred = int(np.argmax(probs))
                    conf = probs[pred]
                    
                    if conf >= CONF_THRES:
                        if pred != last_action and duration >= MIN_DURATION:
                            last_action = pred
                            duration = 0
                        else:
                            duration += 1
                    # else: keep last_action
        
        # Create panel
        panel = create_action_panel(last_action, frame_idx, action_names, action_colors)
        
        # Combine and display
        canvas = np.zeros((900, 700, 3), dtype=np.uint8)
        y_off = (900 - frame.shape[0]) // 2
        canvas[y_off:y_off+frame.shape[0]] = frame
        combined = np.hstack((canvas, panel))
        cv2.imshow("PKU-MMD Action Recognition", combined)
        
        if cv2.waitKey(33) & 0xFF in (ord('q'), 27):
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
