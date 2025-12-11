import os
import tqdm
import numpy as np
import json
from src.utils.saving import load_data_from_pickle, load_data_from_json
from src.features.stage2.spatial_density import extract_detection_features
from src.utils.utils import generate_videos_from_raw_data
from src.features.mask import all_masks

path = "data/features/stage1/"
output_dir = "data/features/stage2/"


def process_raw_data(folder_path, folder_name, id, video_path, real_video_path):
    video_id = id + '/' + folder_name + '.mp4'
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    detections_path = os.path.join(folder_path, "yolo_detections.pkl")
    spatial_density_path = os.path.join(folder_path, "spacial_density.pkl")
    temporal_features_path = os.path.join(folder_path, "temporal_features.pkl")

    detections_data = load_data_from_pickle(detections_path)
    spatial_density_data = load_data_from_pickle(spatial_density_path)
    temporal_features_data = load_data_from_pickle(temporal_features_path)

    detection_features = extract_detection_features(
        detections_data, masks=all_masks[id])

    spatial_density_data = spatial_density_data.tolist()

    all_features = {
        "video_path": video_path,
        "original_video_path": real_video_path,
        "spatial_density": spatial_density_data,
        "temporal_features": temporal_features_data,
        "detection_features": detection_features
    }
    output_data_path = os.path.join(output_dir, folder_name, "features.json")

    json.dump(all_features, open(output_data_path, "w"), indent=4)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    for root, dirs, files in os.walk(path):
        for dir in tqdm.tqdm(dirs, desc="Processing videos"):
            full_path = os.path.join(root, dir)
            folder_name = os.path.basename(full_path)
            normanniles_id = folder_name[:12]
            original_video_path = os.path.join("data/videos/",
                                               normanniles_id, folder_name + ".mp4")

            real_video_path = os.path.realpath(original_video_path)
            process_raw_data(full_path, folder_name, normanniles_id,
                             original_video_path, real_video_path)
