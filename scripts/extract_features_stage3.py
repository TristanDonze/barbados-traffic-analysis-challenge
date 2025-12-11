import os
import tqdm
import numpy as np
import json
from src.utils.saving import load_data_from_json
import pandas as pd

final_features = pd.DataFrame()

path = "data/features/stage2/"
output_file = "data/features/final_features.csv"


def process_spatial_data(spatial_minute_data):
    features = []

    spatial_minute_data_np = np.array(spatial_minute_data)  # shape (N, 4)

    # Iterate over the 4 zones (Entry, Exit, Roundpoint, Total)
    for zone_idx in range(4):
        zone_series = spatial_minute_data_np[:, zone_idx]

        mu = float(np.mean(zone_series))
        sigma = float(np.std(zone_series))
        maximum = float(np.max(zone_series))
        minimum = float(np.min(zone_series))

        x = np.arange(len(zone_series))
        if len(zone_series) > 1:
            slope, _ = np.polyfit(x, zone_series, 1)
            slope = float(slope)
        else:
            slope = 0.0
        features.append([mu, sigma, maximum, minimum, slope])

    return features


def process_temporal_data(temporal_features):
    """
    temporal_features_shape = (200)
    Return mean, std, min, max and slope
    """
    temporal_features_np = np.array(temporal_features)

    mean = float(np.mean(temporal_features_np, axis=0))
    std = float(np.std(temporal_features_np, axis=0))
    maximum = float(np.max(temporal_features_np, axis=0))
    minimum = float(np.min(temporal_features_np, axis=0))

    x = np.arange(temporal_features_np.shape[0])
    if temporal_features_np.shape[0] > 1:
        slope, _ = np.polyfit(x, temporal_features_np, 1)
        slope = float(slope)
    else:
        slope = 0.0

    features = [mean, std, maximum, minimum, slope]
    return features


def process_data(features, folder_name, id):
    temporal_stats_name = ["mean", "std", "max", "min", "slope"]
    video = id + '/' + folder_name + '.mp4'
    video_path = features["video_path"]
    original_video_path = features["original_video_path"]
    spatial_density = features["spatial_density"]
    temporal_features = features["temporal_features"]
    detection_features = features["detection_features"]

    stats_spatial = process_spatial_data(spatial_density)  # shape (4, 5)
    entry_road_occupancy_stats = stats_spatial[0]  # shape (5,)
    exit_road_occupancy_stats = stats_spatial[1]  # shape (5,)
    roundpoint_occupancy_stats = stats_spatial[2]  # shape (5,)
    total_occupancy_stats = stats_spatial[3]  # shape (5,)

    temporal_sin_time = temporal_features["sin_time"]
    temporal_cos_time = temporal_features["cos_time"]
    temporal_day_of_week_onehot = temporal_features["day_of_week_onehot"]

    car_count_total = detection_features["car_count_total"]
    bus_count_total = detection_features["bus_count_total"]
    truck_count_total = detection_features["truck_count_total"]
    total_area_stats = process_temporal_data(
        detection_features["total_area_bbox"])  # shape (5,)

    count_vehicles_entry_stats = process_temporal_data(
        detection_features["count_vehicles_entry"])  # shape (5,)
    count_vehicles_exit_stats = process_temporal_data(
        detection_features["count_vehicles_exit"])  # shape (5,)
    count_vehicles_roundpoint_stats = process_temporal_data(
        detection_features["count_vehicles_roundpoint"])  # shape (5,)

    average_speed_entry_stats = process_temporal_data(
        detection_features["average_speed_entry"])  # shape (5,)
    average_speed_exit_stats = process_temporal_data(
        detection_features["average_speed_exit"])  # shape (5,)
    average_speed_roundpoint_stats = process_temporal_data(
        detection_features["average_speed_roundpoint"])  # shape (5,)

    stopped_vehicles_entry_stats = process_temporal_data(
        detection_features["stopped_vehicles_entry"])  # shape (5,)
    stopped_vehicles_exit_stats = process_temporal_data(
        detection_features["stopped_vehicles_exit"])  # shape (5,)
    stopped_vehicles_roundpoint_stats = process_temporal_data(
        detection_features["stopped_vehicles_roundpoint"])  # shape (5,)

    temporal_features_map_name = {
        "entry_road_occupancy": entry_road_occupancy_stats,
        "exit_road_occupancy": exit_road_occupancy_stats,
        "roundpoint_occupancy": roundpoint_occupancy_stats,
        "total_occupancy": total_occupancy_stats,
        "total_area_stats": total_area_stats,
        "count_vehicles_entry": count_vehicles_entry_stats,
        "count_vehicles_exit": count_vehicles_exit_stats,
        "count_vehicles_roundpoint": count_vehicles_roundpoint_stats,
        "average_speed_entry": average_speed_entry_stats,
        "average_speed_exit": average_speed_exit_stats,
        "average_speed_roundpoint": average_speed_roundpoint_stats,
        "stopped_vehicles_entry": stopped_vehicles_entry_stats,
        "stopped_vehicles_exit": stopped_vehicles_exit_stats,
        "stopped_vehicles_roundpoint": stopped_vehicles_roundpoint_stats,
    }

    features_to_add = {
        "video": video,
        "video_path": video_path,
        "original_video_path": original_video_path,
        "temporal_sin_time": temporal_sin_time,
        "temporal_cos_time": temporal_cos_time,
        "temporal_day_of_week_onehot": temporal_day_of_week_onehot,
        "car_count_total": car_count_total,
        "bus_count_total": bus_count_total,
        "truck_count_total": truck_count_total,
    }

    for feature_name, stats in temporal_features_map_name.items():
        for stat_name, stat_values in zip(temporal_stats_name, stats):
            key = f"{feature_name}_{stat_name}"
            features_to_add[key] = stat_values

    # print(json.dumps(features_to_add, indent=4, ensure_ascii=False))
    # print(f"Total features extracted: {len(features_to_add)}")
    return features_to_add


if __name__ == "__main__":
    path = "data/features/stage2/"
    output_file = "data/features/final_features.csv"

    all_features_data = []

    for root, dirs, files in os.walk(path):
        for dir in tqdm.tqdm(dirs, desc="Processing files"):
            full_path = os.path.join(root, dir)
            folder_name = os.path.basename(full_path)
            normaniles_id = folder_name[:12]
            features_path = os.path.join(full_path, "features.json")

            if os.path.exists(features_path):
                features = load_data_from_json(features_path)
                features_to_add = process_data(
                    features, folder_name, normaniles_id)
                all_features_data.append(features_to_add)

    final_features = pd.DataFrame(all_features_data)
    final_features.to_csv(output_file, index=False)
    print(f"File saved to {output_file} with shape {final_features.shape}")
