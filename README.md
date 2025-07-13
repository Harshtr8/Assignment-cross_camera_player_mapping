# ⚽ Cross-Camera Player Re-Identification – Liat.ai AI Intern Assignment

This AI project tackles the problem of tracking and re-identifying players across **multiple camera views** using deep visual features and YOLO-based object detection.

✅ Cross-view identity mapping  
✅ ResNet-based appearance embeddings  
✅ End-to-end video processing pipeline  
✅ Self-contained and easy to run  

---

## 🎬 Demo Video Outputs

**[📹 broadcast_tracked_output.mp4](./outputs/broadcast_tracked_output.mp4)**  
**[📹 tacticam_reid_output.mp4](./outputs/tacticam_reid_output.mp4)**

Each shows tracked player IDs with consistency across views.

---

## 🚀 Features

- 🎯 **YOLOv8 Object Detection**  
- 🧠 **Deep Feature Embeddings via ResNet-50**  
- 🔁 **Appearance-based Tracking**  
- 🔄 **Cross-Camera ID Mapping with Cosine Similarity**  
- 🎥 **Final Output Videos with Re-ID**  
- 📝 **Well-documented, reproducible pipeline**  

---

## 🛠️ Tech Stack

- Ultralytics YOLOv8 for detection  
- Torch + TorchVision (ResNet50) for embeddings  
- OpenCV for video I/O  
- FAISS/Numpy for similarity matching  
- YAML & JSON for configuration and data  
- TQDM for progress visualization  

---

## 📂 Folder Structure

```
cross_camera_player_mapping/
├── app.py                       # Main pipeline runner
├── draw.py                      # Visualize final re-ID outputs
├── config.yaml                  # Input/output configuration
├── README.md                    # This documentation file
├── report.md                    # Summary of methodology
├── requirements.txt             # Dependencies
│
├── data/
│   ├── broadcast.mp4            # Broadcast view input
│   └── tacticam.mp4             # Tacticam view input
│
├── outputs/
│   ├── broadcast_tracked.json
│   ├── tacticam_tracked.json
│   ├── tacticam_reid.json
│   ├── broadcast_tracked_output.mp4
│   └── tacticam_reid_output.mp4
│
├── src/
│   ├── config.py
│   ├── detect.py
│   ├── reid.py
│   ├── tracker.py
│   └── utils/
│       ├── __init__.py
│       └── feature_extractor.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/harshtr8/cross_camera_player_mapping.git
cd cross_camera_player_mapping
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 detection model (if not included)

Place the fine-tuned YOLOv8 model (`yolov8n.pt`) in the root or `models/` directory as defined in `config.yaml`.

---

## 🧪 Run the Pipeline

### Detect → Track → Re-ID → Visualize

```bash
python app.py
```

### Visualize Final Output Videos

```bash
python draw.py
```

---

## 🧾 Output Files

| File                               | Description                                      |
|------------------------------------|--------------------------------------------------|
| `broadcast_tracked.json`          | Tracked broadcast detections + ResNet features   |
| `tacticam_tracked.json`           | Tracked tacticam detections + features           |
| `tacticam_reid.json`              | tacticam detections re-ID'ed to match broadcast  |
| `broadcast_tracked_output.mp4`    | Video with tracked IDs in broadcast view         |
| `tacticam_reid_output.mp4`        | Video with consistent IDs from cross-view logic  |

---

## 📌 Configuration File

Edit `config.yaml` to change paths or model file:

```yaml
video:
  broadcast: data/broadcast.mp4
  tacticam: data/tacticam.mp4

output_dir: outputs
model_path: yolov8n.pt
```

---

## 📄 Report Highlights (See `report.md`)

- Used **YOLOv8** for detection and ResNet50 embeddings for visual identity.
- Matched players across cameras using **cosine similarity of appearance vectors**.
- Handled invalid crops and noise via bounding box filtering and thresholding.
- Code is modular, readable, and tracks all detections end-to-end.
- Can be extended to **multi-game/multi-view pipelines** with clustering and ID propagation.

---

## 📦 Dependencies

```txt
ultralytics==8.0.x
torch>=2.0
torchvision
opencv-python
scikit-learn
PyYAML
tqdm
numpy
```

Tested on: `Python 3.10`, `Windows 11`, `GPU optional`

---

## 📬 Submission

📤 Submit this folder (GitHub repo or Google Drive link) to:

- arshdeep@liat.ai  
- rishit@liat.ai

---

## 👨‍💻 Author

Submitted by: **Harsh Tripathi**  
For: **Liat.ai AI Intern Role**  
Date: **14-07-2025**