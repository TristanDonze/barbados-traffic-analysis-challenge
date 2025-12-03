import os
from pathlib import Path

os.environ["KAGGLEHUB_CACHE"] = "/Data/barbados/barbados-traffic-analysis-challenge/kaggle_cache"

try:
    import kagglehub
except ImportError:
    raise SystemExit("Please install kagglehub first: pip install kagglehub")

DATASET_MAPPING = {
    "juliusmwangi/brb-traffic-compressed-n1": "normanniles1",
    "juliusmwangi/brb-traffic-compressed-n2": "normanniles2",
    "juliusmwangi/brb-traffic-compressed-n3": "normanniles3",
    "juliusmwangi/brb-traffic-compressed-n4": "normanniles4",
    "juliusmwangi/brb-traffic-compressed-added": "added",
}


def main():
    print("Organizing symlinks in data/videos...\n")

    base_project_dir = Path(
        "/Data/barbados/barbados-traffic-analysis-challenge")
    target_dir = base_project_dir / "data" / "videos"

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}\n")

    for ds_name, folder_name in DATASET_MAPPING.items():
        print(f"Processing: {ds_name}")

        cache_path = kagglehub.dataset_download(ds_name)

        src = Path(cache_path) / folder_name

        link_name = target_dir / folder_name

        if not src.exists():
            print(f"  [ERROR] Source folder not found: {src}")
            continue

        try:
            if not link_name.exists():
                link_name.symlink_to(src, target_is_directory=True)
                print(f"  -> Created link: {link_name} -> {src}")
            else:
                print(f"  -> Link already exists: {link_name}")
        except OSError as e:
            print(f"  [ERROR] Could not create symlink: {e}")

        print()

    print("Organization complete.")
    print("\nListing of data/videos:")
    os.system(f"ls -l {target_dir}")


if __name__ == "__main__":
    main()
