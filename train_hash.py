"""
Train YOLOv11 on hash-yards intersection dataset for local inference.

Uses a pre-exported Roboflow dataset (YOLO11 format) already on disk.
No Roboflow API key needed for training — the dataset is local.

Usage:
    python train_hash.py                        # train with defaults
    python train_hash.py --model yolo11n.pt     # use nano variant
    python train_hash.py --epochs 30            # fewer epochs
    python train_hash.py --resume               # resume interrupted training
"""

import argparse
import os
import shutil
from pathlib import Path

# MPS fallback: let unsupported tensor-indexing ops run on CPU automatically.
# Fixes TAL shape-mismatch crash on Apple Silicon while keeping conv/matmul on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner


# ── MPS workaround ─────────────────────────────────────────────────────────
# Apple Silicon MPS has indexing bugs in the Task Aligned Assigner (TAL) that
# cause shape mismatches and out-of-bounds errors.  We monkey-patch TAL's
# forward() to move tensors to CPU for the assignment math, then move results
# back to MPS.  Conv/matmul (the heavy stuff) still runs on MPS.

_tal_original_forward = TaskAlignedAssigner.forward

@torch.no_grad()
def _tal_cpu_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    device = pd_scores.device
    if device.type == "mps":
        result = _tal_original_forward(
            self,
            pd_scores.cpu(), pd_bboxes.cpu(), anc_points.cpu(),
            gt_labels.cpu(), gt_bboxes.cpu(), mask_gt.cpu(),
        )
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in result)
    return _tal_original_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

TaskAlignedAssigner.forward = _tal_cpu_forward


# ── Configuration ───────────────────────────────────────────────────────────

DATASET_DIR = Path("Hash Yards Intersection.v6i.yolov11")
OUTPUT_MODEL = Path("models/hash_intersection.pt")

# Default training hyperparameters
DEFAULTS = {
    "model": "yolo11s.pt",     # small variant — good accuracy for batch processing
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "device": "mps",
    "patience": 15,
    "project": "runs/hash_intersection",
    "name": "train",
}


def fix_data_yaml(dataset_dir: Path) -> Path:
    """Ensure data.yaml has absolute paths so ultralytics resolves them correctly.

    The Roboflow export uses relative paths like '../train/images' which can
    resolve incorrectly depending on the working directory.  We create a copy
    that points directly to the actual directories inside the dataset folder.

    Returns path to the fixed data.yaml.
    """
    original = dataset_dir / "data.yaml"
    with open(original) as f:
        data = yaml.safe_load(f)

    # Map data.yaml keys to actual subdirectory names inside the dataset folder
    base = dataset_dir.resolve()
    key_to_subdir = {"train": "train", "val": "valid", "test": "test"}

    for key, subdir in key_to_subdir.items():
        images_dir = base / subdir / "images"
        if images_dir.exists():
            data[key] = str(images_dir)
        elif key in data:
            # Fallback: try resolving the original relative path
            raw = Path(data[key])
            if not raw.is_absolute():
                resolved = (base / raw).resolve()
                data[key] = str(resolved)

    # Write fixed yaml next to original
    fixed_path = dataset_dir / "data_abs.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Fixed data.yaml with absolute paths → {fixed_path}")
    return fixed_path


def train(
    model_name: str = DEFAULTS["model"],
    epochs: int = DEFAULTS["epochs"],
    batch: int = DEFAULTS["batch"],
    imgsz: int = DEFAULTS["imgsz"],
    device: str = DEFAULTS["device"],
    patience: int = DEFAULTS["patience"],
    resume: bool = False,
) -> Path:
    """Train YOLOv11 on hash intersection dataset."""

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}\n"
            f"Expected Roboflow YOLO11 export with train/valid/test splits."
        )

    data_yaml = fix_data_yaml(DATASET_DIR)

    print(f"\n{'='*60}")
    print(f"Training {model_name} for hash intersection detection")
    print(f"  Dataset:   {DATASET_DIR}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch}")
    print(f"  ImgSz:     {imgsz}")
    print(f"  Device:    {device}")
    print(f"  Patience:  {patience}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)

    # Minimal augmentation — dataset already has 5× augmentation from
    # Roboflow (rotation ±15°, shear ±15°, brightness ±25%, exposure ±15%,
    # Gaussian blur, salt & pepper noise).
    # Disable AMP on MPS — Apple Silicon's half-precision causes shape
    # mismatches in the Task Assignment Layer (TAL) IoU computation.
    use_amp = device not in ("mps", "MPS")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=DEFAULTS["project"],
        name=DEFAULTS["name"],
        patience=patience,
        save=True,
        plots=True,
        exist_ok=True,
        resume=resume,
        amp=use_amp,
        # Reduced augmentation — dataset is pre-augmented
        degrees=0.0,           # no additional rotation
        shear=0.0,             # no additional shear
        hsv_h=0.005,           # minimal hue shift
        hsv_s=0.1,             # minimal saturation shift
        hsv_v=0.05,            # minimal value shift
        translate=0.05,        # slight translation only
        scale=0.1,             # slight scale only
        flipud=0.0,            # no vertical flip (field has orientation)
        fliplr=0.5,            # horizontal flip is fine
        mosaic=0.3,            # reduced mosaic (some is still helpful)
        mixup=0.0,             # no mixup
    )

    # Copy best weights to standard location
    # YOLO may nest under runs/detect/; check both paths
    best_weights = Path(DEFAULTS["project"]) / DEFAULTS["name"] / "weights" / "best.pt"
    if not best_weights.exists():
        alt = Path("runs/detect") / DEFAULTS["project"] / DEFAULTS["name"] / "weights" / "best.pt"
        if alt.exists():
            best_weights = alt
    if best_weights.exists():
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, OUTPUT_MODEL)
        print(f"\n✓ Best weights saved to {OUTPUT_MODEL}")
        print(f"  (also at {best_weights})")
    else:
        print(f"\nWarning: best.pt not found at {best_weights}")

    return OUTPUT_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on hash-yards intersection dataset"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULTS["model"],
        help=f"Base model (default: {DEFAULTS['model']}). "
             f"Options: yolo11n.pt, yolo11s.pt, yolo11m.pt",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULTS["epochs"],
        help=f"Training epochs (default: {DEFAULTS['epochs']})",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULTS["batch"],
        help=f"Batch size (default: {DEFAULTS['batch']})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULTS["imgsz"],
        help=f"Image size (default: {DEFAULTS['imgsz']})",
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULTS["device"],
        help=f"Device: mps, cpu, or cuda (default: {DEFAULTS['device']})",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULTS["patience"],
        help=f"Early stopping patience (default: {DEFAULTS['patience']})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
