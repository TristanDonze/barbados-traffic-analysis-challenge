from pathlib import Path
import os

os.environ["KAGGLEHUB_CACHE"] = "/Data/barbados/barbados-traffic-analysis-challenge/kaggle_cache"

try:
    import kagglehub
except ImportError:
    raise SystemExit(
        "The 'kagglehub' package is not installed.\n"
        "Install it first with:\n"
        "    pip install kagglehub\n"
    )

# List of datasets to download (same as in the notebook)
DATASETS = [
    "juliusmwangi/brb-traffic-compressed-n1",
    "juliusmwangi/brb-traffic-compressed-n2",
    "juliusmwangi/brb-traffic-compressed-n3",
    "juliusmwangi/brb-traffic-compressed-n4",
    "juliusmwangi/brb-traffic-compressed-added",
]


def main():
    print("Starting download of BRB traffic datasets via kagglehub...\n")

    # Optional: directory where you want to *organise* symlinks to the data
    target_dir = Path(
        "/Data/barbados/barbados-traffic-analysis-challenge/data/full_data")
    target_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        print(f"Downloading dataset: {ds}")
        # This downloads (or reuses cache) and returns the local path
        local_path = kagglehub.dataset_download(ds)
        print(f"  -> Kagglehub stored it at: {local_path}")

        # Optionally, create a symlink into target_dir for convenience
        # so you have everything under ./data/brb_traffic
        src = Path(local_path)
        link_name = target_dir / src.name
        try:
            if not link_name.exists():
                link_name.symlink_to(src, target_is_directory=True)
                print(f"  -> Linked into: {link_name}")
            else:
                print(f"  -> Link already exists: {link_name}")
        except OSError:
            # On Windows without admin / symlink rights just skip this part
            print(
                "  (Could not create symlink; you can access data via the kagglehub path above.)")

        print()

    print("All datasets processed.")


if __name__ == "__main__":
    main()
