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
WINDOW_SIZE = 128
CONF_THRES = 0.15

SKELETON = [
    (0,1),(1,20),(20,2),(2,3),
    (20,4),(4,5),(5,6),(6,7),(7,21),(21,22),
    (20,8),(8,9),(9,10),(10,11),(11,23),(23,24),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19)
]


# === Utils ===

def load_skel(path):
    data = []
    with open(path) as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            if len(coords) != 150:
                coords = (coords + [0.0]*150)[:150]
            data.append(coords)
    return np.array(data, dtype=np.float32)


def norm_skel(skel, mean, std):
    T = skel.shape[0]
    data = skel.reshape(T, 2, 25, 3)
    data = (data - mean) / std
    return data.reshape(T, 150)


def draw_skel(frame, skel, idx, mean, std):
    if idx >= len(skel): return frame
    h, w = frame.shape[:2]
    data = skel[idx:idx+1].reshape(1, 2, 25, 3)
    data = data * std + mean
    data = data.reshape(2, 25, 3)
    
    colors = [(0,0,255), (255,0,0)]
    for p in range(2):
        joints = data[p]
        pts = []
        for j in range(25):
            x = int((joints[j,0] + 1) * w / 2)
            y = int((1 - joints[j,1]) * h / 2)
            pts.append((x, y))
            if joints[j].any():
                cv2.circle(frame, (x,y), 3, colors[p], -1)
        for (i,j) in SKELETON:
            if joints[i].any() and joints[j].any():
                cv2.line(frame, pts[i], pts[j], colors[p], 1)
    return frame


# === Visualization ===

def get_action_colors():
    df = pd.read_excel(os.path.join(DATA_ROOT, "Actions.xlsx"))
    names, colors = {0: "Background"}, {0: (128,128,128)}
    for _, row in df.iterrows():
        aid = int(row['Label'])
        names[aid] = row['Action']
        hue = (aid * 137.5) % 360
        bgr = cv2.cvtColor(np.uint8([[[hue,255,255]]]), cv2.COLOR_HSV2BGR)[0,0]
        colors[aid] = tuple(int(c) for c in bgr)
    return names, colors


def draw_panel(width, height, action, names, colors, frame_id):
    panel = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "PKU-MMD Action Recognition", (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Color boxes
    box_w, box_h = 30, 30
    for aid in range(1, 52):
        row, col = (aid-1)//13, (aid-1)%13
        x = 50 + col * (box_w + 5)
        y = 80 + row * (box_h + 5)
        color = colors[aid]
        if aid == action:
            color = tuple(min(255, int(c*1.5)) for c in color)
            cv2.rectangle(panel, (x-2,y-2), (x+box_w+2,y+box_h+2), color, 2)
        cv2.rectangle(panel, (x,y), (x+box_w,y+box_h), color, -1)
        cv2.putText(panel, f"{aid}", (x+5, y+box_h-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    # Current action
    if action != 0:
        name = names[action][:25] + "..." if len(names[action]) > 25 else names[action]
        cv2.putText(panel, f"ID: {action}", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[action], 2)
        cv2.putText(panel, f"Action: {name}", (50, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[action], 2)
    else:
        cv2.putText(panel, "Background", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,128,128), 2)
    
    cv2.putText(panel, f"Frame: {frame_id}", (50, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return panel


# === Inference ===

def run_inference(video_id):
    # Load model
    model = MS_TCN(4, 10, 64, IN_FEATURES, NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    
    mean = np.load("models/joint_mean.npy")
    std = np.load("models/joint_std.npy")
    names, colors = get_action_colors()
    
    # Load data
    vid_path = os.path.join(DATA_ROOT, "Data", "RGB_VIDEO", f"{video_id}.avi")
    skel_path = os.path.join(DATA_ROOT, "Data", "Skeleton", f"{video_id}.txt")
    cap = cv2.VideoCapture(vid_path)
    skel = norm_skel(load_skel(skel_path), mean, std)
    
    buffer = []
    frame_id = 0
    last_action = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize
        h, w = frame.shape[:2]
        scale = 700 / w
        frame = cv2.resize(frame, (700, int(h*scale)))
        
        # Draw skeleton
        frame = draw_skel(frame, skel, frame_id, mean, std)
        
        # Inference
        if frame_id < len(skel):
            buffer.append(skel[frame_id])
            if len(buffer) > WINDOW_SIZE:
                buffer.pop(0)
            
            if len(buffer) == WINDOW_SIZE:
                inp = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(inp)
                    probs = F.softmax(out, dim=-1)[0, WINDOW_SIZE//2].cpu().numpy()
                    pred = np.argmax(probs)
                    if probs[pred] >= CONF_THRES:
                        last_action = pred
        
        # Display
        panel = draw_panel(900, frame.shape[0], last_action, names, colors, frame_id)
        disp = np.hstack((frame, panel))
        cv2.imshow("PKU-MMD Action Recognition", disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # List videos
    vid_dir = os.path.join(DATA_ROOT, "Data", "RGB_VIDEO")
    videos = [f.replace(".avi", "") for f in os.listdir(vid_dir) if f.endswith("-M.avi")]
    print("Available videos:", videos[:5], "...")
    
    vid = input("Enter video ID (e.g., 0002-M): ").strip()
    if vid in videos:
        run_inference(vid)
    else:
        print("Video not found")
