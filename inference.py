#inference.py
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd
import time
import math
from collections import deque
import torch.nn.functional as F

# MODEL DEFINITION
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
        return x + out

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
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.stage1(x)
        for stage in self.stages:
            out = stage(out) + out
        out = self.classifier(out)
        return out.permute(0, 2, 1)

# ========================
# CONFIGURATION
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/ms_tcn_pku.pth" 
DATA_ROOT = r"E:/CARET_Project/CARET/PKUMMDv2"
WINDOW_NAME = "PKU-MMD Action Recognition"
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
VIDEO_WIDTH = 700
PANEL_WIDTH = 900

# Model parameters
NUM_CLASSES = 52
IN_FEATURES = 150
NUM_STAGES = 4
NUM_LAYERS = 10
NUM_F_MAPS = 64

# Inference parameters
WINDOW_SIZE = 128
CONFIDENCE_THRESHOLD = 0.15
MIN_ACTION_DURATION = 3

# OpenCV properties
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_COUNT = 7
CV_CAP_PROP_POS_FRAMES = 1

# Skeleton connections
SKELETON_CONNECTIONS = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (21, 22),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (23, 24),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19)
]

# ========================
# LOAD MODEL & NORMALIZATION
# ========================
print("Loading model and normalization parameters...")
model = MS_TCN(
    num_stages=NUM_STAGES,
    num_layers=NUM_LAYERS,
    num_f_maps=NUM_F_MAPS,
    features_dim=IN_FEATURES,
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

joint_mean = np.load("models/joint_mean.npy")
joint_std = np.load("models/joint_std.npy")

print(f"Model loaded on {DEVICE}")
print(f"Normalization parameters loaded")

# ========================
# LOAD ACTION MAPPING
# ========================
print("Loading action mapping...")
action_df = pd.read_excel(os.path.join(DATA_ROOT, "Actions.xlsx"))
action_names = {}
action_colors = {}

for idx, row in action_df.iterrows():
    action_id = int(row['Label'])
    action_name = row['Action']
    action_names[action_id] = action_name
    
    hue = (action_id * 137.5) % 360
    color = tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[hue, 255, 255]]]), 
        cv2.COLOR_HSV2BGR
    )[0][0])
    action_colors[action_id] = color

action_names[0] = "Background"
action_colors[0] = (128, 128, 128)

print(f"Loaded {len(action_names)} actions")

# ========================
# DATA LOADING UTILITIES
# ========================
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

def normalize_skeleton_per_joint(skeleton_data, joint_mean, joint_std):
    """Normalize skeleton data per joint across all frames"""
    T = skeleton_data.shape[0]
    data_4d = skeleton_data.reshape(T, 2, 25, 3)
    data_4d = (data_4d - joint_mean) / joint_std
    return data_4d.reshape(T, 150)

# ========================
# VISUALIZATION UTILITIES
# ========================
def draw_color_id_chart(panel, current_action, frame_count):
    """Draw a chart of 51 colored ID boxes in 4 rows with improved blinking"""
    box_size = 35
    box_margin = 5
    boxes_per_row = 13
    rows = 4
    
    total_width = boxes_per_row * (box_size + box_margin) - box_margin
    start_x = (panel.shape[1] - total_width) // 2
    start_y = 60
    
    chart_height = rows * (box_size + box_margin) + 20
    cv2.rectangle(panel, (start_x - 15, start_y - 15), 
                 (start_x + total_width + 15, start_y + chart_height), 
                 (40, 40, 40), -1)
    
    cv2.putText(panel, "Action IDs", (panel.shape[1] // 2 - 50, start_y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    for action_id in range(1, 52):
        idx = action_id - 1
        row = idx // boxes_per_row
        col = idx % boxes_per_row
        
        x = start_x + col * (box_size + box_margin)
        y = start_y + row * (box_size + box_margin)
        
        color = action_colors[action_id]
        
        # Apply blinking effect for current action
        if action_id == current_action:
            blink_state = (frame_count // 2) % 2
            if blink_state == 0:
                color = tuple(min(255, int(c * 1.8)) for c in color)
                cv2.rectangle(panel, (x - 4, y - 4), 
                             (x + box_size + 4, y + box_size + 4), 
                             color, 3)
            else:
                color = tuple(int(c * 0.5) for c in color)
        
        cv2.rectangle(panel, (x, y), (x + box_size, y + box_size), color, -1)
        cv2.rectangle(panel, (x, y), (x + box_size, y + box_size), (255, 255, 255), 1)
        cv2.putText(panel, f"ID{action_id}", (x + 5, y + box_size - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return start_y + chart_height + 15

def draw_action_segments_bars(panel, current_action, frame_count, bar_y):
    """Draw two rows of color bars representing actions with improved blinking"""
    bar_height = 30
    bar_width = panel.shape[1] - 30
    start_x = 24
    
    row1_y = bar_y
    row2_y = bar_y + bar_height + 10
    
    cv2.rectangle(panel, (start_x, row1_y - 10), 
                 (start_x + bar_width, row2_y + bar_height + 10), 
                 (40, 40, 40), -1)
    
    cv2.putText(panel, "Action Segments", (start_x, row1_y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # First row (actions 1-25)
    segment_width = bar_width // 25
    for action_id in range(1, 26):
        idx = action_id - 1
        x = start_x + idx * segment_width
        
        color = action_colors[action_id]
        
        if action_id == current_action:
            blink_state = (frame_count // 2) % 2
            if blink_state == 0:
                color = tuple(min(255, int(c * 1.8)) for c in color)
                cv2.rectangle(panel, (x - 3, row1_y - 3), 
                             (x + segment_width + 3, row1_y + bar_height + 3), 
                             color, 3)
            else:
                color = tuple(int(c * 0.5) for c in color)
        
        cv2.rectangle(panel, (x, row1_y), (x + segment_width, row1_y + bar_height), color, -1)
        cv2.rectangle(panel, (x, row1_y), (x + segment_width, row1_y + bar_height), (255, 255, 255), 1)
        
        if action_id % 5 == 0:
            cv2.putText(panel, str(action_id), (x + 5, row1_y + bar_height // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Second row (actions 26-51)
    segment_width = bar_width // 26
    for action_id in range(26, 52):
        idx = action_id - 26
        x = start_x + idx * segment_width
        
        color = action_colors[action_id]
        
        if action_id == current_action:
            blink_state = (frame_count // 2) % 2
            if blink_state == 0:
                color = tuple(min(255, int(c * 1.8)) for c in color)
                cv2.rectangle(panel, (x - 3, row2_y - 3), 
                             (x + segment_width + 3, row2_y + bar_height + 3), 
                             color, 3)
            else:
                color = tuple(int(c * 0.5) for c in color)
        
        cv2.rectangle(panel, (x, row2_y), (x + segment_width, row2_y + bar_height), color, -1)
        cv2.rectangle(panel, (x, row2_y), (x + segment_width, row2_y + bar_height), (255, 255, 255), 1)
        
        if action_id % 5 == 0:
            cv2.putText(panel, str(action_id), (x + 5, row2_y + bar_height // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return row2_y + bar_height + 20

def draw_action_panel(panel_height, panel_width, current_action, frame_count):
    """Draw the action panel with improved layout and blinking"""
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 30
    
    # Draw title
    title = "Human Action Recognition System"
    cv2.putText(panel, title, (panel_width // 2 - 180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Draw color ID chart
    chart_end_y = draw_color_id_chart(panel, current_action, frame_count)
    
    # Draw model name
    model_name = "MS-TCN Model"
    cv2.putText(panel, model_name, (panel_width // 2 - 70, chart_end_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Draw action segments bars
    bar_y = chart_end_y + 60
    bars_end_y = draw_action_segments_bars(panel, current_action, frame_count, bar_y)
    
    # Draw current action info
    if current_action != 0:
        current_name = action_names[current_action]
        current_color = action_colors[current_action]
        
        info_y = bars_end_y + 40
        
        # First row: ID no:
        cv2.putText(panel, "ID no:", (40, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(panel, str(current_action), (140, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 2)
        
        # Second row: Action name:
        cv2.putText(panel, "Action name:", (40, info_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        display_name = current_name
        if len(current_name) > 30:
            display_name = current_name[:27] + "..."
        
        cv2.putText(panel, display_name, (190, info_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 2)
        
        # Frame info
        debug_text = f"Frame: {frame_count} | Action: {current_action}"
        cv2.putText(panel, debug_text, (40, info_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    else:
        # Show background action
        info_y = bars_end_y + 40
        cv2.putText(panel, "No action detected (Background)", (40, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        
        debug_text = f"Frame: {frame_count} | Action: 0 (Background)"
        cv2.putText(panel, debug_text, (40, info_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return panel

def draw_skeleton_on_frame(frame, skeleton_data, frame_idx, joint_mean, joint_std,
                           coord_range='normalized', flip_x=False, invert_y=True, 
                           swap_persons=False, auto_fix_mirror=True, debug=False):
    """Draw skeleton joints and connections on the frame"""
    if frame_idx >= len(skeleton_data):
        return frame
    h, w = frame.shape[:2]
    
    skeleton_normalized = skeleton_data[frame_idx:frame_idx+1]
    try:
        skeleton_4d = skeleton_normalized.reshape(1, 2, 25, 3)
    except:
        skeleton_4d = skeleton_normalized.reshape(1, 2, 25, 3)
    
    skeleton_4d_denorm = skeleton_4d * joint_std + joint_mean
    skeleton_denorm = skeleton_4d_denorm.reshape(2, 25, 3)
    
    if swap_persons:
        skeleton_denorm = skeleton_denorm[::-1]
    
    person_colors = [(0, 0, 255), (255, 0, 0)]
    
    def project_to_image(joints, flip_x_local, invert_y_local):
        pts = []
        valid_mask = np.zeros(25, dtype=bool)
        xs = joints[:, 0].astype(float)
        ys = joints[:, 1].astype(float)
        
        if coord_range == 'normalized':
            if invert_y_local:
                ys = -ys
            if flip_x_local:
                xs = -xs
            img_xs = ((xs + 1.0) * 0.5) * (w - 1)
            img_ys = ((ys + 1.0) * 0.5) * (h - 1)
        else:
            if invert_y_local:
                ys = h - ys
            if flip_x_local:
                xs = w - xs
            img_xs = xs
            img_ys = ys
            
        for i in range(25):
            x_val = img_xs[i]
            y_val = img_ys[i]
            if np.isfinite(x_val) and np.isfinite(y_val):
                orig_x, orig_y, orig_z = joints[i]
                if not (abs(orig_x) < 1e-6 and abs(orig_y) < 1e-6 and abs(orig_z) < 1e-6):
                    valid_mask[i] = True
            else:
                valid_mask[i] = False
            xi = int(np.clip(round(x_val), 0, w - 1))
            yi = int(np.clip(round(y_val), 0, h - 1))
            pts.append((xi, yi))
        return np.array(pts), valid_mask
    
    def score(points, valid_mask):
        if valid_mask.sum() == 0:
            return -1
        inner_valid = 0
        centroid_x, centroid_y = [], []
        for i, v in enumerate(valid_mask):
            if not v:
                continue
            x, y = points[i]
            margin = min(0.05 * w, 5)
            if margin < x < w - margin and margin < y < h - margin:
                inner_valid += 1
            centroid_x.append(x)
            centroid_y.append(y)
        if centroid_x:
            cent_x = np.mean(centroid_x)
            cent_y = np.mean(centroid_y)
            center_dist = np.sqrt((cent_x - w/2)**2 + (cent_y - h/2)**2)
            centroid_bonus = max(0, inner_valid - center_dist / max(w, h) * 10)
            return inner_valid + centroid_bonus
        return inner_valid
    
    best_scores = []
    for person_idx in range(2):
        joints = skeleton_denorm[person_idx]
        if np.all(np.abs(joints) < 1e-6):
            best_scores.append(-1)
            continue
            
        if not auto_fix_mirror or (flip_x is not None and invert_y is not None):
            flip_x_local = flip_x
            invert_y_local = invert_y
            pts, valid = project_to_image(joints, flip_x_local, invert_y_local)
            sc = score(pts, valid)
            best_scores.append(sc)
            chosen_pts = pts
            chosen_valid = valid
        else:
            best_sc = -np.inf
            best_pts, best_valid = None, None
            for fx in [False, True]:
                for iy in [False, True]:
                    pts_cand, valid_cand = project_to_image(joints, fx, iy)
                    sc = score(pts_cand, valid_cand)
                    if sc > best_sc:
                        best_sc = sc
                        best_pts = pts_cand
                        best_valid = valid_cand
            chosen_pts, chosen_valid = best_pts, best_valid
            best_scores.append(best_sc)
            if best_sc > -1:
                for fx in [False, True]:
                    for iy in [False, True]:
                        pts_test, valid_test = project_to_image(joints, fx, iy)
                        if np.allclose(pts_test, chosen_pts) and np.array_equal(valid_test, chosen_valid):
                            break
                    else:
                        continue
                    break
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < 25 and end_idx < 25 and chosen_valid[start_idx] and chosen_valid[end_idx]:
                sp = tuple(chosen_pts[start_idx])
                ep = tuple(chosen_pts[end_idx])
                cv2.line(frame, sp, ep, person_colors[person_idx], 2, lineType=cv2.LINE_AA)
        for i in range(25):
            if chosen_valid[i]:
                pt = tuple(chosen_pts[i])
                cv2.circle(frame, pt, 4, person_colors[person_idx], -1)
                cv2.circle(frame, pt, 4, (255, 255, 255), 1)
    return frame

# ========================
# VIDEO PROCESSING
# ========================
def process_video(video_id):
    """Process a single video with action recognition"""
    video_path = os.path.join(DATA_ROOT, "Data", "RGB_VIDEO", f"{video_id}.avi")
    skeleton_path = os.path.join(DATA_ROOT, "Data", "Skeleton", f"{video_id}.txt")
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    if not os.path.exists(skeleton_path):
        print(f"Skeleton file not found: {skeleton_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    orig_width = int(cap.get(CV_CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(CV_CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(CV_CAP_PROP_FPS)
    
    print(f"Video info: {orig_width}x{orig_height}, {total_frames} frames, {video_fps:.2f} FPS")
    
    skeleton_data_normalized = load_skeleton(skeleton_path)
    skeleton_data_normalized = normalize_skeleton_per_joint(skeleton_data_normalized, joint_mean, joint_std)
    
    print(f"Loaded {len(skeleton_data_normalized)} skeleton frames")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    print("Processing video...")
    frame_count = 0
    start_time = time.time()
    last_action = -1
    action_duration = 0
    last_confident_action = 0
    action_confidence = 0.0
    
    # Initialize buffer for temporal processing
    window_buffer = deque(maxlen=WINDOW_SIZE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx = int(cap.get(CV_CAP_PROP_POS_FRAMES)) - 1
        
        target_height = int(orig_height * VIDEO_WIDTH / orig_width)
        frame_resized = cv2.resize(frame, (VIDEO_WIDTH, target_height))
        
        frame_with_skeleton = draw_skeleton_on_frame(
            frame_resized.copy(), 
            skeleton_data_normalized, 
            frame_idx, 
            joint_mean, 
            joint_std,
            swap_persons=False,
            flip_x=False,
            invert_y=True,
            auto_fix_mirror=True
        )
        
        # Get skeleton for current frame
        if frame_idx < len(skeleton_data_normalized):
            window_buffer.append(skeleton_data_normalized[frame_idx])
            
            # Predict action when we have enough frames
            if len(window_buffer) == WINDOW_SIZE:
                # Prepare input for model
                input_skel = torch.tensor(
                    np.array(window_buffer), 
                    dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                
                # Predict action
                with torch.no_grad():
                    output = model(input_skel)
                    
                    # Get probabilities using softmax
                    probabilities = F.softmax(output, dim=-1).cpu().numpy()[0, WINDOW_SIZE//2]
                    
                    # Get top prediction
                    pred = np.argmax(probabilities)
                    confidence = probabilities[pred]
                    
                    # Only update if confidence is above threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Check if this is a significant change from the last confident action
                        if pred != last_confident_action:
                            # If we've had the same action for a while, allow change
                            if action_duration >= MIN_ACTION_DURATION:
                                last_confident_action = pred
                                action_confidence = confidence
                                action_duration = 0
                        # If same action but higher confidence, update
                        elif pred == last_confident_action and confidence > action_confidence:
                            action_confidence = confidence
                    else:
                        # If confidence is low, keep the last confident action
                        pred = last_confident_action
                    
                    # Update action duration
                    if pred == last_confident_action:
                        action_duration += 1
                    else:
                        action_duration = 0
            else:
                # Not enough frames yet, use last confident action
                pred = last_confident_action
        else:
            # No skeleton data, use background
            pred = 0
        
        # Update last action for tracking
        if pred != last_action:
            last_action = pred
        
        # Create action panel
        panel = draw_action_panel(WINDOW_HEIGHT, PANEL_WIDTH, pred, frame_count)
        
        # Create a canvas with the same height as the panel
        canvas = np.zeros((WINDOW_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        # Calculate the vertical position to center the video
        y_offset = (WINDOW_HEIGHT - frame_resized.shape[0]) // 2
        
        # Place the resized video in the center of the canvas
        canvas[y_offset:y_offset + frame_resized.shape[0], :] = frame_with_skeleton
        
        # Add frame counter and FPS to the canvas
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine video canvas and panel
        combined = np.hstack((canvas, panel))
        
        # Display
        cv2.imshow(WINDOW_NAME, combined)
        
        # Exit on 'q' or ESC
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processing completed for {video_id}")

# ========================
# MAIN FUNCTION
# ========================
def main():
    print("=== PKU-MMD Action Recognition Inference ===")
    
    video_files = [f for f in os.listdir(os.path.join(DATA_ROOT, "Data", "RGB_VIDEO")) 
                   if f.endswith("-M.avi")]
    video_ids = [f.replace(".avi", "") for f in video_files]
    
    if not video_ids:
        print("No videos found in RGB_VIDEO directory")
        return
    
    print("Available videos (Middle view):")
    for i, vid in enumerate(video_ids[:10]):
        print(f"{i+1}. {vid}")
    if len(video_ids) > 10:
        print(f"... and {len(video_ids)-10} more")
    
    video_id = input("\nEnter video ID to process (e.g., '0002-M'): ").strip()
    
    if video_id not in video_ids:
        print(f"Invalid video ID. Available videos: {', '.join(video_ids[:5])}...")
        return
    
    process_video(video_id)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()