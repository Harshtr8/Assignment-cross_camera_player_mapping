import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm
import os
import yaml

def detect_players(video_path, model_path, output_json):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # class 0 = player
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({
                    "frame": frame_idx,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
        frame_idx += 1

    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    detect_players(cfg["broadcast_video"], cfg["model_path"], f"{cfg['output_dir']}/broadcast_detections.json")
    detect_players(cfg["tacticam_video"], cfg["model_path"], f"{cfg['output_dir']}/tacticam_detections.json")