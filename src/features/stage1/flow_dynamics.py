import numpy as np
import cv2


def calculate_global_pixel_motion(current_frame_gray: np.array, previous_frame_gray: np.array, road_mask: np.array):
    """
    Computes the mean absolute difference between two consecutive frames within the road area.
    This measures the 'global energy' of the scene.
    """
    if current_frame_gray.shape != previous_frame_gray.shape:
        raise ValueError(
            "Current and previous frames must have the same shape.")

    # Ensure mask is boolean or 0/1 for multiplication, but calculate area based on non-zero
    road_mask_bool = (road_mask > 0).astype(np.float32)

    # Use float32 to avoid uint8 overflow/underflow during subtraction
    frame_diff = np.abs(current_frame_gray.astype(
        np.float32) - previous_frame_gray.astype(np.float32))

    masked_diff = frame_diff * road_mask_bool

    # Use count_nonzero to be safe against masks with value 255
    road_area_pixels = np.count_nonzero(road_mask)

    if road_area_pixels == 0:
        return 0.0

    global_motion_intensity = np.sum(masked_diff) / road_area_pixels

    return global_motion_intensity


def match_centroids(current_centroids, previous_centroids, max_distance=100):
    """
    Helper function: Matches current centroids to previous ones using Nearest Neighbor.
    Returns a list of distances (pixels traveled) for matched vehicles.
    """
    if not current_centroids or not previous_centroids:
        return []

    curr_pts = np.array(current_centroids)  # Shape (N, 2)
    prev_pts = np.array(previous_centroids)  # Shape (M, 2)

    # Calculate Euclidean distance matrix between all N current and M previous points
    # resulting shape (N, M)
    dists = np.linalg.norm(curr_pts[:, None] - prev_pts[None, :], axis=2)

    # For each current point, find index of closest previous point
    closest_prev_indices = np.argmin(dists, axis=1)
    min_distances = np.min(dists, axis=1)

    valid_movements = []

    # Filter out matches that are too far (e.g., a car entering shouldn't match a car leaving)
    for d in min_distances:
        if d < max_distance:
            valid_movements.append(d)

    return valid_movements


# def estimate_average_speed(current_centroids: list, previous_centroids: list):
#     """
#     Calculates the average pixel distance traveled by vehicles between frames.
#     High speed = Fluid. Low Speed = Congestion.
#     """
#     movements = match_centroids(current_centroids, previous_centroids)

#     if not movements:
#         return 0.0

#     return np.mean(movements)


# def compute_stationary_ratio(current_centroids: list, previous_centroids: list, movement_threshold: int = 2):
#     """
#     Returns the percentage of vehicles that have moved less than 'threshold' pixels.
#     Ratio 1.0 = Total Gridlock. Ratio 0.0 = Everyone is moving.
#     """
#     movements = match_centroids(current_centroids, previous_centroids)

#     if not movements:
#         return 0.0

#     # Count how many moved LESS than threshold
#     stationary_count = sum(1 for m in movements if m < movement_threshold)

#     return stationary_count / len(movements)


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # --- Test Configuration ---
    video_path = "data/videos/normanniles1/normanniles1_2025-10-20-16-01-45.mp4"
    real_path = os.path.realpath(video_path)

    # Mask Loading (Assuming standard paths)
    entry_mask_path = "src/mask/road_mask/normanniles1_entry_mask.png"
    exit_mask_path = "src/mask/road_mask/normanniles1_exit_mask.png"
    round_mask_path = "src/mask/road_mask/normanniles1_roundpoint_mask.png"

    try:
        entry = cv2.imread(entry_mask_path, 0)
        exit_ = cv2.imread(exit_mask_path, 0)
        round_ = cv2.imread(round_mask_path, 0)

        # Merge masks
        merged = cv2.bitwise_or(entry, exit_)
        merged = cv2.bitwise_or(merged, round_)
        # Binarize strict (0 or 1)
        _, road_mask = cv2.threshold(merged, 127, 1, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"Error loading masks: {e}")
        exit()

    print(f"Processing video: {real_path}")

    cap = cv2.VideoCapture(real_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 3
    frame_interval = int(fps // target_fps)

    global_motion_intensities = []

    # Initial Frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Skip frames to reach target FPS
        for _ in range(frame_interval - 1):
            cap.grab()

        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Test Pixel Motion
        motion = calculate_global_pixel_motion(curr_gray, prev_gray, road_mask)
        global_motion_intensities.append(motion)

        # (Note: We cannot test estimate_average_speed here without YOLO results)

        prev_gray = curr_gray

    cap.release()

    # Visualization
    time_axis = [i / target_fps for i in range(len(global_motion_intensities))]

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, global_motion_intensities, label='Global Motion')
    plt.title("Motion Intensity (Pixel Difference) Analysis")
    plt.xlabel("Seconds")
    plt.ylabel("Intensity Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("global_motion_test.png")
    print("Graph saved to global_motion_test.png")
