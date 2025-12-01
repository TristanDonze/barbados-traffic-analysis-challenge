import os
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

BUCKET_NAME = 'brb-traffic'
LOCAL_DIR = 'data/videos'
CSV_PATH = 'data/Train.csv'

def get_blob_name(video_path):
    filename = video_path.split('/')[-1]
    camera_id = filename.split('_')[0]
    return f"{camera_id}/{filename}"

def download_dataset():
    # if not os.path.exists(LOCAL_DIR):
    #     os.makedirs(LOCAL_DIR)
    #     print(f"Directory '{LOCAL_DIR}' created.")

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Reading Train.csv...")
    df = pd.read_csv(CSV_PATH)
    df['blob_path'] = df['videos'].apply(get_blob_name)
    all_blobs = df['blob_path'].unique().tolist()

    print(f"Starting download of {len(all_blobs)} files from bucket '{BUCKET_NAME}'...")

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
    except Exception as e:
        print(f"Authentication/Connection error: {e}")
        return

    for blob_path in tqdm(all_blobs, unit="file"):
        file_name = os.path.basename(blob_path)
        local_file_path = os.path.join(LOCAL_DIR, file_name)
        if os.path.exists(local_file_path):
            continue
        try:
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_file_path)
        except Exception as e:
            print(f"\nError downloading {blob_path}: {e}")

    print(f"\nDownload finished. Files are in '{LOCAL_DIR}'.")

if __name__ == "__main__":
    download_dataset()
