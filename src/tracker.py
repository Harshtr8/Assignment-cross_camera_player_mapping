import json
import cv2
import numpy as np
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        from torchvision.models import resnet50, ResNet50_Weights
        import torch.nn as nn
        import torchvision.transforms as T

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # remove classification head
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bbox):
        import torch
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Clip to image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]

        # Check for valid crop size
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            print(f"[WARNING] Skipping invalid crop at {bbox}")
            return None

        try:
            input_tensor = self.transform(crop).unsqueeze(0)
            with torch.no_grad():
                features = self.model(input_tensor)
            return features[0].numpy().tolist()
        except Exception as e:
            print(f"[ERROR] ResNet feature extraction failed: {e}")
            return None

def track_with_appearance(video_path, detections_path, output_path):
    print(f"[INFO] Tracking players using appearance features in: {video_path}")
    cap = cv2.VideoCapture(video_path)
    with open(detections_path) as f:
        detections = json.load(f)

    detections_by_frame = {}
    for d in detections:
        frame = d["frame"]
        detections_by_frame.setdefault(frame, []).append(d)

    fe = FeatureExtractor()
    frame_idx = 0
    obj_id_counter = 0
    output = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = detections_by_frame.get(frame_idx, [])
        for det in frame_dets:
            bbox = det["bbox"]
            conf = det["confidence"]

            if conf < 0.3:
                continue  # Skip low confidence

            embedding = fe.extract(frame, bbox)
            if embedding is None:
                print(f"[DEBUG] No embedding for frame {frame_idx}, bbox: {bbox}")
                continue

            print(f"[DEBUG] ✓ Embedding extracted at frame {frame_idx}, ID {obj_id_counter}")

            det["id"] = obj_id_counter
            det["embedding"] = embedding
            obj_id_counter += 1
            output.append(det)

        frame_idx += 1

    cap.release()

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[✅] Tracked data saved to: {output_path}")