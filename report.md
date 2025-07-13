# 🎯 Liat.ai Assignment Report – Cross-Camera Player Mapping

## 👤 Author
Candidate for AI Intern – Liat.ai

## 📝 Task Overview
The objective was to assign consistent player IDs across two videos (`broadcast.mp4` and `tacticam.mp4`) taken from different camera angles. The solution must ensure that players keep the same identity even when moving out of frame or switching views.

---

## 🧠 Approach Summary

### 1. Object Detection
Used a fine-tuned YOLOv8 model to detect:
- Players
- Goalkeepers
- Referees
- Ball (filtered out for tracking)

Detections were filtered using a confidence threshold of **0.3**.

### 2. Improved Tracking with Appearance Features
Replaced naive IoU-only tracker with a feature-based tracker:
- Used **ResNet50** to extract 2048-D embeddings from each player crop
- Combined IoU and cosine similarity for assigning track IDs
- Prevented ID drift using appearance memory

### 3. Cross-Camera Re-Identification
For each tracked ID:
- Averaged the embeddings across multiple frames
- Matched players from `tacticam` to `broadcast` using **cosine similarity**

Mapping: `tacticam ID → broadcast ID`

### 4. Visualization
Generated final output video (`tacticam_reid_output.mp4`) with consistent player IDs across the second view.

---

## ✅ Results

- ✔️ Smooth tracking with consistent IDs even after occlusions
- ✔️ Cross-view ID mapping mostly accurate based on visual similarity
- ✔️ Modular code, clean structure, and efficient runtime
- ✔️ Output: `outputs/tacticam_reid_output.mp4`

---

## 🔧 Challenges & Limitations

- Minor ID drift in dense crowd frames
- Appearance-based re-ID works well but could be improved using domain-specific models (e.g., jersey ReID)
- Not yet using ball/player interactions or team info

---

## 🚀 Next Steps (If More Time Was Available)

- Integrate domain-specific ReID model (OSNet, FastReID)
- Combine temporal info (e.g., Kalman filtering)
- Improve track ID confidence smoothing
- Add team/jersey classification for better association

---

## 📦 Files Submitted

- `app.py` – pipeline orchestrator
- `reid.py` – cross-view ID mapping
- `tracker.py` – tracker with embeddings
- `feature_extractor.py` – ResNet50 embedding extractor
- `draw.py` – visualization script
- `outputs/` – final results
- `requirements.txt`, `README.md`, `config.yaml`