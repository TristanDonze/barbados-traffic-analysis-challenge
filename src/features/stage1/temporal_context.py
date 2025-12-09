import datetime
import math
import os


def get_temporal_features(video_path: str):
    """Extracts temporal features from the video file path.
    Args:
        video_path (str): Path to the video file. (e.g "data/videos/normanniles1/normanniles1_2025-10-20-06-01-45.mp4")
    Returns:
        dict: A dictionary containing temporal features.
    """
    # Extract timestamp from the video filename
    filename = os.path.basename(video_path)
    timestamp_str = filename.split("_")[-1].replace(".mp4", "")

    # Parse the timestamp
    dt_object = parse_timestamp(timestamp_str)
    # print(f"Parsed datetime: {dt_object}")
    # Encode cyclical time
    sin_time, cos_time = encode_cyclical_time(dt_object)

    # Get one-hot encoded day of the week
    day_of_week_onehot = get_day_of_week_onehot(dt_object)

    return {
        "sin_time": sin_time,
        "cos_time": cos_time,
        "day_of_week_onehot": day_of_week_onehot,
    }


def parse_timestamp(timestamp_str: str):
    """Converts string timestamp to datetime object.
    Args:
        timestamp_str (str): Timestamp string in the format "YYYY-MM-DD-HH-MM-SS".
    Returns:
        datetime.datetime: Parsed datetime object.
    """
    return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")


def encode_cyclical_time(dt_object):
    """
    Returns [sin_time, cos_time] based on the minute of the day.
    Normalizes time to a continuous circle (23:59 is close to 00:00).
    Args:
        dt_object (datetime.datetime): Datetime object.
    Returns:
        tuple: (sin_time, cos_time)
    """
    minutes_in_day = dt_object.hour * 60 + dt_object.minute
    radians = 2 * math.pi * minutes_in_day / (24 * 60)
    sin_time = math.sin(radians)
    cos_time = math.cos(radians)
    return sin_time, cos_time


def get_day_of_week_onehot(dt_object):
    """Returns a one-hot encoded vector for the day of the week.
    Args:
        dt_object (datetime.datetime): Datetime object.
    Returns:
        list: One-hot encoded vector for the day of the week (0=Monday, 6=Sunday).
    """
    day_of_week = dt_object.weekday()  # Monday=0, Sunday=6
    onehot = [0] * 7
    onehot[day_of_week] = 1
    return onehot


if __name__ == "__main__":
    video_path = "data/videos/normanniles1/normanniles1_2025-10-20-06-01-45.mp4"
    features = get_temporal_features(video_path)
    print(features)
