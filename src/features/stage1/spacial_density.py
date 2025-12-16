import os
import cv2
import numpy as np
from src.features.mask import all_masks
from src.utils.utils import get_normanniles_id


def create_background_model(video_path, num_frames=50):
    """
    Generates a background image (empty road) by taking the median of random frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_indices = np.random.choice(total_frames, num_frames, replace=False)
    frames = []

    for idx in random_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)


def calculate_occupancy(frame_gray, background_gray, mask, threshold=25):
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


def spacial_density_analysis(real_path):
    normanniles_id = get_normanniles_id(real_path)
    masks = all_masks[normanniles_id]

    background = create_background_model(real_path)
    occupancy_data = np.zeros((60, 4))
    cap = cv2.VideoCapture(real_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for sec in range(60):
        temp_values = np.zeros((3, 4))
        offsets = [0.0, 0.33, 0.66]

        for i, offset in enumerate(offsets):
            target_frame_idx = int((sec + offset) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for col_idx in range(4):
                val = calculate_occupancy(
                    frame_gray, background, masks[col_idx])
                temp_values[i, col_idx] = val
        occupancy_data[sec] = np.mean(temp_values, axis=0)

    cap.release()
    return occupancy_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    output_dir = "/Data/barbados/plots"
    os.makedirs(output_dir, exist_ok=True)

    video_path = "/Data/barbados/barbados-traffic-analysis-challenge/data/videos/normanniles1/normanniles1_2025-10-20-06-00-45.mp4"
    real_path = os.path.realpath(video_path)
    occupancy_data = spacial_density_analysis(real_path)
    zones_names = ["Entry", "Exit", "Roundabout", "Total Combined"]
    print("\n--- Occupancy Averages Over the Minute ---")
    for i, name in enumerate(zones_names):
        print(
            f"{name} : {np.mean(occupancy_data[:, i]):.4f} ({np.mean(occupancy_data[:, i])*100:.1f}%)")

    plt.figure(figsize=(12, 12))
    time_axis = np.arange(60)

    for i, name in enumerate(zones_names):
        plt.subplot(4, 1, i + 1)
        plt.plot(time_axis, occupancy_data[:, i], label=name, alpha=0.8)
        plt.title(f"Occupancy Rate: {name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Occupancy Rate")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occupancy_analysis.png"))
