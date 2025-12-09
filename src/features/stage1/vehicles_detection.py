from collections import Counter, defaultdict
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

classes = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

model = YOLO("models/yolo11s.pt")


def extract_yolo_tracking_data(real_path):
    cap = cv2.VideoCapture(real_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // 3
    frame_count = 0

    extracted_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  classes=[2, 3, 5, 7], verbose=False)

            frame_detections = []

            for box in results[0].boxes:
                if box.id is not None:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id.cpu().numpy().item())
                    cls_id = int(box.cls.cpu().numpy().item())
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)

                    if cls_id in classes:
                        detection = {
                            "track_id": track_id,
                            "cls_name": classes[cls_id],
                            "bbox": (x1, y1, x2, y2),
                            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                            "area": (x2 - x1) * (y2 - y1)
                        }
                        frame_detections.append(detection)

            extracted_data.append({
                "frame_index": frame_count,
                "detections": frame_detections
            })

        frame_count += 1

    cap.release()
    return extracted_data


if __name__ == "__main__":
    video_file = "data/videos/normanniles3/normanniles3_2025-10-26-17-37-45.mp4"

    # Load your binary masks here (grayscale)
    entry_mask = cv2.imread(
        "src/mask/road_mask/normanniles3_entry_mask.png", 0)
    exit_mask = cv2.imread(
        "src/mask/road_mask/normanniles3_exit_mask.png", 0)
    round_mask = cv2.imread(
        "src/mask/road_mask/normanniles3_roundpoint_mask.png", 0)

    masks = {
        "entry": entry_mask,  # entry_mask
        "exit": exit_mask,  # exit_mask
        "roundpoint": round_mask  # round_mask
    }
    import time

    start_raw = time.time()
    raw_data = extract_yolo_tracking_data(video_file)
    end_raw = time.time()
