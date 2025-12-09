import os
import cv2
import time
import numpy as np
import torch
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from ultralytics import YOLO

from src.utils.saving import save_data_to_pickle
from src.features.mask import all_masks
from src.utils.utils import get_normanniles_id
from src.features.stage1.temporal_context import get_temporal_features

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

output_dir = "data/features/stage1/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

classes_dict = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


def get_background(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    current_idx = 0
    desired_indices = set(indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in desired_indices:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        current_idx += 1

    cap.release()

    if not frames:
        return None

    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)


def calculate_occupancy_frame(frame_gray, background_gray, mask, threshold=25):
    blurred_frame = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    blurred_bg = cv2.GaussianBlur(background_gray, (5, 5), 0)

    diff_intensity = cv2.absdiff(blurred_bg, blurred_frame)
    _, binary_intensity = cv2.threshold(
        diff_intensity, threshold, 255, cv2.THRESH_BINARY)

    edges_frame = cv2.Canny(blurred_frame, 50, 150)
    edges_bg = cv2.Canny(blurred_bg, 50, 150)

    edges_diff = cv2.subtract(edges_frame, edges_bg)

    combined_binary = cv2.bitwise_or(binary_intensity, edges_diff)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
    dilated_mask = cv2.dilate(closed_mask, kernel, iterations=2)

    motion_in_zone = cv2.bitwise_and(dilated_mask, dilated_mask, mask=mask)

    total_pixels_mask = cv2.countNonZero(mask)
    if total_pixels_mask == 0:
        return 0.0

    occupied_pixels = cv2.countNonZero(motion_in_zone)
    return occupied_pixels / total_pixels_mask


def process_video_optimized(video_path):
    try:
        cv2.setNumThreads(0)
        torch.set_num_threads(1)

        real_video_path = os.path.realpath(video_path)
        folder_name = os.path.basename(video_path).split(".")[0]
        current_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)

        normanniles_id = get_normanniles_id(real_video_path)
        masks = all_masks[normanniles_id]

        background_gray = get_background(real_video_path)
        if background_gray is None:
            return

        model = YOLO("models/yolo11s.pt")

        cap = cv2.VideoCapture(real_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30

        yolo_interval = fps // 3

        density_frames_per_sec = 3
        density_interval = fps // density_frames_per_sec

        yolo_data = []

        occupancy_accumulator = np.zeros((60, 4))
        occupancy_counts = np.zeros((60, 4))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. YOLO DETECTION
            if frame_idx % yolo_interval == 0:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                      classes=[2, 3, 5, 7], verbose=False)

                frame_detections = []
                for box in results[0].boxes:
                    if box.id is not None:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id.cpu().numpy().item())
                        cls_id = int(box.cls.cpu().numpy().item())

                        if cls_id in classes_dict:
                            detection = {
                                "track_id": track_id,
                                "cls_name": classes_dict[cls_id],
                                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                                "center": ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2),
                                "area": (float(x2) - float(x1)) * (float(y2) - float(y1))
                            }
                            frame_detections.append(detection)

                yolo_data.append({
                    "frame_index": frame_idx,
                    "detections": frame_detections
                })

            # 2. SPATIAL DENSITY
            # Logic: Collect samples and average them per second later
            if frame_idx % density_interval == 0:
                current_sec = frame_idx // fps
                if current_sec < 60:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for col_idx in range(4):
                        val = calculate_occupancy_frame(
                            frame_gray, background_gray, masks[col_idx])
                        occupancy_accumulator[current_sec, col_idx] += val
                        occupancy_counts[current_sec, col_idx] += 1

            frame_idx += 1

        cap.release()

        # Finalize Density Data (Average)
        with np.errstate(divide='ignore', invalid='ignore'):
            final_occupancy = occupancy_accumulator / occupancy_counts
            final_occupancy = np.nan_to_num(final_occupancy)

        # 3. TEMPORAL FEATURES (Lightweight, can stay separate or be merged if needed)
        temporal_features = get_temporal_features(video_path)

        # Saving
        save_data_to_pickle(yolo_data, os.path.join(
            current_output_dir, "yolo_detections.pkl"))
        save_data_to_pickle(final_occupancy, os.path.join(
            current_output_dir, "spacial_density.pkl"))
        save_data_to_pickle(temporal_features, os.path.join(
            current_output_dir, "temporal_features.pkl"))

    except Exception as e:
        print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    all_video_files = []
    search_dir = "data/videos"
    for root, dirs, files in os.walk(search_dir):
        for dir in dirs:
            for file in os.listdir(os.path.join(root, dir)):
                if file.endswith(".mp4"):
                    all_video_files.append(os.path.join(root, dir, file))

    videos_to_process = all_video_files
    num_cores = 16

    print("STARTING OPTIMIZED PROCESSING")
    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_video_optimized, videos_to_process),
             total=len(videos_to_process), desc="Processing videos"))
