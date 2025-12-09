def get_normanniles_id(video_path):
    index = video_path.find("normanniles")
    id = video_path[index:index+12]
    return id


if __name__ == "__main__":
    path = "data/videos/normanniles1/normanniles1_2025-10-20-06-01-45.mp4"
    import os
    real_path = os.path.realpath(path)
    print(get_normanniles_id(real_path))
