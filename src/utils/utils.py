import cv2
import os


def get_normanniles_id(video_path):
    index = video_path.find("normanniles")
    id = video_path[index:index+12]
    return id


def generate_videos_from_raw_data(raw_data, video_path):
    real_path = os.path.realpath(video_path)
    cap = cv2.VideoCapture(real_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // 3
    frame_count = 0

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(output_dir, "annotated_output.mp4")
    out = None

    data_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and data_index < len(raw_data):
            frame_data = raw_data[data_index]
            data_index += 1

            for det in frame_data["detections"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                track_id = det["track_id"]
                cls_name = det["cls_name"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(
                    out_video_path, fourcc, fps // 3, (width, height))

            out.write(frame)

        frame_count += 1

    cap.release()
    if out:
        out.release()
    print(f"Annotated video saved to {out_video_path}")


if __name__ == "__main__":
    path = "data/videos/normanniles1/normanniles1_2025-10-20-06-01-45.mp4"
    import os
    real_path = os.path.realpath(path)
    print(get_normanniles_id(real_path))
