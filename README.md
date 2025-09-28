```markdown
# Human Action Recognition System

## Project Overview
This project implements a **Human Action Recognition System** using the **Multi-Stage Temporal Convolutional Network (MS-TCN)** on the **PKU-MMDv2 dataset**. The model performs **frame-level action classification** on skeleton data, recognizing **51 distinct human actions** plus a **background class** (total 52 classes).

The system processes 3D joint coordinates from two-person skeleton sequences and predicts actions in real time during inference with visual feedback.

---

## 📂 Directory Structure
```
ActionRecognition/
├── models/                   # Saved model weights and normalization stats
│   ├── ms_tcn_pku.pth        # Trained MS-TCN model checkpoint
│   ├── joint_mean.npy        # Skeleton normalization mean
│   ├── joint_std.npy         # Skeleton normalization std
│
├── logs/                     # Training logs and confusion matrices
│
├── train_ms_tcn.py           # Training script for MS-TCN model
├── inference.py              # Real-time inference with visualization
├── README.md                 # Project documentation
```

---

## Dependencies
Install required packages:
```bash
pip install torch opencv-python pandas numpy scikit-learn matplotlib seaborn
```

Ensure you have **PyTorch** and **OpenCV** installed with CUDA support (optional but recommended).

---

## Training the Model
To train the model, run:
```bash
python train_ms_tcn.py
```
This will:
- Load skeleton data and labels from PKU-MMDv2
- Normalize joint coordinates per joint
- Train the MS-TCN model using cross-subject split
- Save the best model based on F1-score
- Generate confusion matrices every 5 epochs

> ⚠️ Update `DATA_ROOT` in `train_ms_tcn.py` to point to your PKU-MMDv2 directory.

---

## Inference & Visualization
Run inference on any video from the dataset:
```bash
python inference.py
```
- Select a video ID (e.g., `0002-M`)
- View real-time skeleton overlay with joint connections
- See predicted action ID, name, and confidence
- Color-coded action panel with blinking highlight for current action
- Press `q` or `ESC` to exit

The system uses temporal smoothing and confidence thresholds to ensure stable predictions.

---

## Dataset
This project uses the **PKU-MMDv2 dataset** (skeleton modality):
- **Input**: 150D skeleton vectors (2 persons × 25 joints × 3 coordinates)
- **Labels**: Frame-wise action annotations (51 actions + background)
- **Splits**: Supports cross-subject and cross-view evaluation

> 📌 You must obtain the PKU-MMDv2 dataset separately and place it in the expected directory structure.

---

## Contributing
Feel free to open issues or submit pull requests for improvements.

---

## License
MIT License © 2025
```
