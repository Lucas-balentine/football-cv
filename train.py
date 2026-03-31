"""
Fine-tune YOLOv8m on American football datasets from Roboflow.

Downloads multiple datasets, remaps their class labels to a unified schema
(player, referee, ball), merges them, and trains YOLOv8m.

Usage:
    python train.py                    # download datasets + train
    python train.py --download-only    # just download and merge datasets
    python train.py --skip-download    # train on already-downloaded data
    python train.py --epochs 100       # customize training
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────

DATASETS_DIR = Path("datasets")
MERGED_DIR = DATASETS_DIR / "merged"

# Unified class schema
UNIFIED_CLASSES = ["player", "referee", "ball"]

# Datasets to download and how to remap their classes
# Format: (workspace, project, version, class_mapping)
# class_mapping: {original_class_name: unified_class_name_or_None_to_skip}
DATASET_CONFIGS = [
    {
        "workspace": "nflfootball",
        "project": "nfl-players-qs3y8",
        "version": 1,
        "class_map": {
            "Player": "player",
            "Referee": "referee",
        },
    },
    {
        "workspace": "bronkscottema",
        "project": "football-player-detection",
        "version": 1,
        "class_map": {
            # All positions → player
            "Center": "player",
            "QB": "player",
            "db": "player",
            "lb": "player",
            "Runningback": "player",
            "Fullback": "player",
            "Tightend": "player",
            "H-back": "player",
            "Wide Receiver": "player",
            "wide receiver": "player",
            "wide-receiver": "player",
            "tight-end": "player",
            "running-back": "player",
            "fullback": "player",
            "center": "player",
            "qb": "player",
            "h-back": "player",
        },
    },
    {
        "workspace": "fh-technikum-wien-m15r2",
        "project": "american-football-player-detection",
        "version": 1,
        "class_map": {
            "american-football-players": "player",
            "american-football-player": "player",
            "player": "player",
            "referee": "referee",
            "ball": "ball",
            # Skip non-relevant classes
            "whitehat": None,
        },
    },
]


def download_datasets(api_key: str) -> list[Path]:
    """Download all configured datasets from Roboflow in YOLOv8 format."""
    rf = Roboflow(api_key=api_key)
    downloaded = []

    for config in DATASET_CONFIGS:
        name = config["project"]
        print(f"\n{'='*60}")
        print(f"Downloading: {config['workspace']}/{name} v{config['version']}")
        print(f"{'='*60}")

        try:
            project = rf.workspace(config["workspace"]).project(config["project"])
            version = project.version(config["version"])
            dataset = version.download("yolov8", location=str(DATASETS_DIR / name))
            downloaded.append(DATASETS_DIR / name)
            print(f"  Downloaded to {DATASETS_DIR / name}")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")
            continue

    return downloaded


def read_dataset_yaml(dataset_path: Path) -> dict:
    """Read the data.yaml from a downloaded dataset."""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        # Some datasets put it at the root with a different name
        for candidate in dataset_path.glob("*.yaml"):
            yaml_path = candidate
            break

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def remap_labels(
    label_file: Path,
    original_classes: list[str],
    class_map: dict[str, str | None],
) -> list[str]:
    """Remap label file class indices to the unified schema.

    Returns the remapped lines, skipping any classes mapped to None.
    """
    unified_index = {name: i for i, name in enumerate(UNIFIED_CLASSES)}
    remapped_lines = []

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_class_idx = int(parts[0])
            if orig_class_idx >= len(original_classes):
                continue

            orig_class_name = original_classes[orig_class_idx]

            # Look up in the class map (try original name and lowercase)
            new_class = class_map.get(orig_class_name)
            if new_class is None:
                new_class = class_map.get(orig_class_name.lower())
            if new_class is None:
                # Skip this annotation
                continue

            new_idx = unified_index[new_class]
            parts[0] = str(new_idx)
            remapped_lines.append(" ".join(parts))

    return remapped_lines


def merge_datasets_with_configs(downloaded_paths: list[Path], configs: list[dict]) -> Path:
    """Merge multiple downloaded datasets into a single unified dataset."""
    print(f"\n{'='*60}")
    print("Merging datasets...")
    print(f"{'='*60}")

    # Clean and recreate merged directory
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in ["train", "valid", "test"]:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_annotations = 0

    for dataset_path, config in zip(downloaded_paths, configs):
        name = config["project"]
        print(f"\n  Processing {name}...")

        # Read original class list
        try:
            data_yaml = read_dataset_yaml(dataset_path)
        except Exception as e:
            print(f"    Failed to read data.yaml: {e}")
            continue

        original_classes = data_yaml.get("names", [])
        if isinstance(original_classes, dict):
            max_idx = max(original_classes.keys())
            original_classes = [original_classes.get(i, f"unknown_{i}") for i in range(max_idx + 1)]

        print(f"    Original classes: {original_classes}")

        class_map = config["class_map"]
        dataset_images = 0

        # Collect all image/label pairs from whatever splits exist
        all_pairs = []  # list of (img_path, lbl_path)
        found_splits = []

        for split in ["train", "valid", "test"]:
            img_dir = dataset_path / split / "images"
            lbl_dir = dataset_path / split / "labels"

            if not img_dir.exists():
                continue
            found_splits.append(split)

            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    all_pairs.append((img_file, lbl_file, split))

        print(f"    Found {len(all_pairs)} labeled images in splits: {found_splits}")

        # If dataset only has train (no valid), auto-split 85/15
        has_valid = any(s in found_splits for s in ["valid", "val"])
        if not has_valid and len(all_pairs) > 10:
            import random
            random.seed(42)
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * 0.85)
            # Reassign splits
            reassigned = []
            for i, (img, lbl, _) in enumerate(all_pairs):
                new_split = "train" if i < split_idx else "valid"
                reassigned.append((img, lbl, new_split))
            all_pairs = reassigned
            print(f"    Auto-split: {split_idx} train, {len(all_pairs) - split_idx} valid")

        for img_file, lbl_file, split in all_pairs:
            remapped = remap_labels(lbl_file, original_classes, class_map)
            if not remapped:
                continue

            prefix = name.replace("-", "_")[:10]
            new_img_name = f"{prefix}_{img_file.name}"
            new_lbl_name = f"{prefix}_{img_file.stem}.txt"

            shutil.copy2(img_file, MERGED_DIR / split / "images" / new_img_name)

            with open(MERGED_DIR / split / "labels" / new_lbl_name, "w") as f:
                f.write("\n".join(remapped) + "\n")

            dataset_images += 1
            total_annotations += len(remapped)

        total_images += dataset_images
        print(f"    Added {dataset_images} images")

    # Write merged data.yaml
    data_yaml_content = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }

    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print(f"\n  Merged dataset: {total_images} images, {total_annotations} annotations")
    print(f"  Classes: {UNIFIED_CLASSES}")
    print(f"  Saved to {MERGED_DIR}")

    # Print split breakdown
    for split in ["train", "valid", "test"]:
        img_dir = MERGED_DIR / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.iterdir()))
            print(f"    {split}: {count} images")

    return MERGED_DIR


def train(
    data_path: Path,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "mps",
) -> Path:
    """Fine-tune YOLOv8m on the merged dataset."""
    print(f"\n{'='*60}")
    print(f"Training YOLOv8m for {epochs} epochs")
    print(f"  Dataset: {data_path}")
    print(f"  Batch size: {batch}, Image size: {imgsz}, Device: {device}")
    print(f"{'='*60}\n")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=str(data_path / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="football_detector",
        project="runs",
        patience=10,
        save=True,
        plots=True,
    )

    # The best weights path
    best_weights = Path("runs/football_detector/weights/best.pt")
    if best_weights.exists():
        # Copy to project root for easy access
        shutil.copy2(best_weights, "football_detector.pt")
        print(f"\n  Best weights saved to football_detector.pt")

    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m on football data")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download and merge datasets, don't train")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, train on existing merged data")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: mps, cpu, or cuda (default: mps)")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not args.skip_download:
        if not api_key or api_key == "your_key_here":
            print("Error: Set ROBOFLOW_API_KEY in .env file")
            print("  Get a free key at https://app.roboflow.com/settings/api")
            return

        downloaded = download_datasets(api_key)
        if not downloaded:
            print("No datasets downloaded successfully")
            return

        # Only pass configs that actually downloaded
        successful_configs = []
        for config in DATASET_CONFIGS:
            path = DATASETS_DIR / config["project"]
            if path.exists() and (path / "data.yaml").exists():
                successful_configs.append(config)

        # Temporarily swap DATASET_CONFIGS for merge
        downloaded_paths = [DATASETS_DIR / c["project"] for c in successful_configs]
        merge_datasets_with_configs(downloaded_paths, successful_configs)

        if args.download_only:
            return

    # Verify merged dataset exists
    if not (MERGED_DIR / "data.yaml").exists():
        print(f"Error: No merged dataset found at {MERGED_DIR}")
        print("  Run without --skip-download first")
        return

    train(
        data_path=MERGED_DIR,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
