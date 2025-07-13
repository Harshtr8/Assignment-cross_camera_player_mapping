# draw.py

import json
import cv2
import os

def draw_video(input_video, detection_json, output_path):
    cap = cv2.VideoCapture(input_video)
    with open(detection_json) as f:
        data = json.load(f)

    data_by_frame = {}
    for d in data:
        frame = d['frame']
        data_by_frame.setdefault(frame, []).append(d)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for obj in data_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = map(int, obj["bbox"])
            pid = obj.get("id", -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✅] Output video saved to: {output_path}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # ▶️ Draw broadcast tracked output
    draw_video(
        input_video="data/broadcast.mp4",
        detection_json="outputs/broadcast_tracked.json",
        output_path="outputs/broadcast_tracked_output.mp4"
    )

    # ▶️ Draw tacticam re-id output
    draw_video(
        input_video="data/tacticam.mp4",
        detection_json="outputs/tacticam_reid.json",
        output_path="outputs/tacticam_reid_output.mp4"
    )