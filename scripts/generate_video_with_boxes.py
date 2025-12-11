import cv2
import os
import pickle


def get_normanniles_id(video_path: str) -> str:
    """Extract 'normannilesX' camera id from the full video path."""
    index = video_path.find("normanniles")
    cam_id = video_path[index:index + 12]  # e.g. 'normanniles1'
    return cam_id




def load_raw_data(detections_path: str):
    with open(detections_path, "rb") as f:
        raw_data = pickle.load(f)
    return raw_data


def generate_videos_from_raw_data(raw_data, video_path):
    real_path = os.path.realpath(video_path)
    cap = cv2.VideoCapture(real_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 3)  # same sampling as in Stage 1
    frame_count = 0

    output_dir = "/Data/barbados/plots"
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(
        output_dir,
        os.path.basename(video_path).replace(".mp4", "_annotated.mp4")
    )
    out = None

    data_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # We only have detections for every frame_interval-th frame
        if frame_count % frame_interval == 0 and data_index < len(raw_data):
            frame_data = raw_data[data_index]
            data_index += 1

            for det in frame_data["detections"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                track_id = det["track_id"]
                cls_name = det["cls_name"]

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # draw label
                label = f"{cls_name} ID:{track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # init VideoWriter once we know the frame size
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(out_video_path, fourcc, fps // 3, (width, height))

            out.write(frame)

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()

    print(f"Annotated video saved to: {out_video_path}")


if __name__ == "__main__":
    video_path = (
        "/Data/barbados/barbados-traffic-analysis-challenge/data/videos/"
        "normanniles1/normanniles1_2025-10-20-10-17-45.mp4"
    )

    detections_path = '/Data/barbados/barbados-traffic-analysis-challenge/data/features/stage1/normanniles1_2025-10-20-10-17-45/yolo_detections.pkl'
    print("Using detections from:", detections_path)

    raw_data = load_raw_data(detections_path)
    generate_videos_from_raw_data(raw_data, video_path)