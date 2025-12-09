from collections import Counter, defaultdict
import numpy as np
import math

classes = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


def extract_detection_features(detections_data, masks=None):
    id_class_history = defaultdict(list)

    for frame_data in detections_data:
        for detection in frame_data["detections"]:
            track_id = detection["track_id"]
            cls_name = detection["cls_name"]
            id_class_history[track_id].append(cls_name)

    final_class_counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0
    }

    for track_id, class_list in id_class_history.items():
        if class_list:
            most_common_class = Counter(class_list).most_common(1)[0][0]
            if most_common_class in final_class_counts:
                final_class_counts[most_common_class] += 1

    results = {
        "car_count_total": final_class_counts["car"],
        "bus_count_total": final_class_counts["bus"],
        "truck_count_total": final_class_counts["truck"],
        "total_area_bbox": [],

        "count_vehicles_entry": [],
        "count_vehicles_roundpoint": [],
        "count_vehicles_exit": [],

        "average_speed_entry": [],
        "average_speed_roundpoint": [],
        "average_speed_exit": [],

        "stopped_vehicles_entry": [],
        "stopped_vehicles_roundpoint": [],
        "stopped_vehicles_exit": []
    }

    mask_mapping = {
        0: "entry",
        2: "roundpoint",
        1: "exit"
    }

    for i, frame_data in enumerate(detections_data):
        current_detections = frame_data["detections"]

        frame_total_area = sum(float(d["area"]) for d in current_detections)
        results["total_area_bbox"].append(frame_total_area)

        previous_centroids = {}
        if i > 0:
            prev_detections = detections_data[i-1]["detections"]
            for d in prev_detections:
                previous_centroids[d["track_id"]] = d["center"]

        zone_stats = {
            name: {"count": 0, "total_speed": 0.0, "stopped_count": 0}
            for name in mask_mapping.values()
        }

        for det in current_detections:
            track_id = det["track_id"]
            center = det["center"]
            cx, cy = int(center[0]), int(center[1])

            speed = 0.0
            if track_id in previous_centroids:
                prev_cx, prev_cy = previous_centroids[track_id]
                dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                speed = float(dist)

            is_stopped = speed < 2.0

            if masks:
                for mask_id, zone_name in mask_mapping.items():
                    mask = masks.get(mask_id)
                    if mask is not None:
                        h, w = mask.shape[:2]
                        if 0 <= cx < w and 0 <= cy < h:
                            if mask[cy, cx] > 127:
                                zone_stats[zone_name]["count"] += 1
                                zone_stats[zone_name]["total_speed"] += speed
                                if is_stopped:
                                    zone_stats[zone_name]["stopped_count"] += 1

        for zone_name in mask_mapping.values():
            count = zone_stats[zone_name]["count"]
            total_speed = zone_stats[zone_name]["total_speed"]
            stopped = zone_stats[zone_name]["stopped_count"]

            avg_speed = (total_speed / count) if count > 0 else 0.0

            results[f"count_vehicles_{zone_name}"].append(int(count))
            results[f"average_speed_{zone_name}"].append(float(avg_speed))
            results[f"stopped_vehicles_{zone_name}"].append(int(stopped))

    return results
