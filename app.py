import os
import sys
import yaml
sys.path.append("src")

from ultralytics import YOLO
from src.tracker import track_with_appearance
from src.reid import run_reid

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg["output_dir"], exist_ok=True)

def detect(video_path, model_path, out_json):
    model = YOLO(model_path)
    import cv2, json
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > cfg["frame_limit"]:
            break
        results = model(frame)
        for box in results[0].boxes:
            if int(box.cls[0]) != 0: continue  # Only detect class 0 (player)
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "frame": frame_idx,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            })
        frame_idx += 1

    with open(out_json, "w") as f:
        json.dump(detections, f, indent=2)

# 1. Run detection
print("[INFO] Running YOLOv8 detection...")
detect(cfg["broadcast_video"], cfg["model_path"], f"{cfg['output_dir']}/broadcast_detections.json")
detect(cfg["tacticam_video"], cfg["model_path"], f"{cfg['output_dir']}/tacticam_detections.json")

# 2. Run tracking
print("[INFO] Tracking players with appearance features...")
track_with_appearance(cfg["broadcast_video"],
                      f"{cfg['output_dir']}/broadcast_detections.json",
                      f"{cfg['output_dir']}/broadcast_tracked.json")

track_with_appearance(cfg["tacticam_video"],
                      f"{cfg['output_dir']}/tacticam_detections.json",
                      f"{cfg['output_dir']}/tacticam_tracked.json")

# 3. Re-ID
print("[INFO] Running Cross-Camera Re-Identification...")
run_reid(f"{cfg['output_dir']}/broadcast_tracked.json",
         f"{cfg['output_dir']}/tacticam_tracked.json",
         f"{cfg['output_dir']}/tacticam_reid.json")

print(f"[✅] Pipeline complete. Output saved in: {cfg['output_dir']}")