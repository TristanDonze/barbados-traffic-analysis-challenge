import os
import cv2
from PIL import Image


def extract_first_frame(video_path, output_path):
    try:
        real_video_path = os.path.realpath(video_path)
        if not os.path.exists(real_video_path):
            print(f"Video file does not exist: {real_video_path}\n")
            return
        else:
            print(f"Processing video file: {real_video_path}")
        cap = cv2.VideoCapture(real_video_path)
        ret, frame = cap.read()
        print(ret, frame)
        if ret:
            cv2.imwrite(output_path, frame)
        cap.release()

    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")


if __name__ == "__main__":
    folder_path = "./data/videos/"
    all_normanniles = [
        "normanniles1/normanniles1_",
        "normanniles2/normanniles2_",
        "normanniles3/normanniles3_",
        "normanniles4/normanniles4_",
    ]
    path = "2025-10-21-14-00-45.mp4"
    output_folder_path = "src/mask/extracted_images/"
    for normanniles in all_normanniles:
        file_path = os.path.join(folder_path, normanniles + path)
        normanniles_name = normanniles.split("/")[0]
        output_file_path = os.path.join(
            output_folder_path, normanniles_name + "_first_frame.jpg")
        extract_first_frame(file_path, output_file_path)
