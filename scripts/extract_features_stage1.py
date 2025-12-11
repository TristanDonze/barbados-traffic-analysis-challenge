from src.features.stage1.temporal_context import get_temporal_features
from src.features.stage1.spacial_density import spacial_density_analysis
from src.features.stage1.vehicles_detection import extract_yolo_tracking_data
from src.utils.saving import save_data_to_pickle, save_data_to_json
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import torch
import cv2
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


output_dir = "data/features/stage1/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def process_video_file(video_path):
    try:
        cv2.setNumThreads(0)
        torch.set_num_threads(1)

        real_video_path = os.path.realpath(video_path)
        folder_name = os.path.basename(video_path).split(".")[0]
        current_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(current_output_dir, exist_ok=True)

        detections_file = os.path.join(
            current_output_dir, "yolo_detections.pkl")
        occupancy_file = os.path.join(
            current_output_dir, "spacial_density.pkl")
        temporal_file = os.path.join(
            current_output_dir, "temporal_features.pkl")

        detections = extract_yolo_tracking_data(real_video_path)
        occupancy_data = spacial_density_analysis(real_video_path)
        temporal_features = get_temporal_features(video_path)

        save_data_to_pickle(detections, detections_file)
        save_data_to_pickle(occupancy_data, occupancy_file)
        save_data_to_pickle(temporal_features, temporal_file)
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

    videos_to_process = all_video_files[:128]

    num_cores = 8

    print("STARTING PROCESSING")
    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap_unordered(process_video_file, videos_to_process),
             total=len(videos_to_process), desc="Processing videos"))
