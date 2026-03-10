"""
Baseline player detection using COCO-pretrained YOLOv8m.

Runs person detection (COCO class 0) on input images, filters detections
to only those standing on the green playing field, and saves annotated
results to the output/ directory.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class 0 = person
PERSON_CLASS_ID = 0

# HSV range for green grass (tuned for broadcast football footage)
GREEN_LOW = np.array([30, 40, 40])
GREEN_HIGH = np.array([80, 255, 255])

# Minimum fraction of green pixels in the region below a detection's
# foot position for it to be considered "on the field"
FIELD_GREEN_THRESHOLD = 0.3

# Size of the sampling region (in pixels) around the foot position
FOOT_SAMPLE_RADIUS = 15


def build_field_mask(image: np.ndarray) -> np.ndarray:
    """Create a binary mask where True = green field pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    # Dilate to fill small gaps in the grass (yard lines, paint)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def is_on_field(box_xyxy: list[float], field_mask: np.ndarray) -> bool:
    """Check if a detection's foot position is on the green field.

    Samples a small region around the bottom-center of the bounding box
    and checks if enough of it overlaps with the field mask.
    """
    h, w = field_mask.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    # Foot position = bottom-center of bounding box
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)

    # Sample a region around the foot
    r = FOOT_SAMPLE_RADIUS
    sx1 = max(0, foot_x - r)
    sx2 = min(w, foot_x + r)
    sy1 = max(0, foot_y - r)
    sy2 = min(h, foot_y + r)

    region = field_mask[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return False

    green_ratio = np.count_nonzero(region) / region.size
    return green_ratio >= FIELD_GREEN_THRESHOLD


def detect_players(image_paths: list[str], confidence: float = 0.3) -> None:
    model = YOLO("yolov8m.pt")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            print(f"Skipping {image_path}: file not found")
            continue

        print(f"\nProcessing {path.name}...")
        image = cv2.imread(str(path))
        field_mask = build_field_mask(image)

        # Save field mask for debugging
        mask_path = output_dir / f"{path.stem}_field_mask{path.suffix}"
        cv2.imwrite(str(mask_path), field_mask)

        results = model(str(path), conf=confidence, classes=[PERSON_CLASS_ID])
        result = results[0]

        total = len(result.boxes)
        kept_indices = []
        for i, box in enumerate(result.boxes):
            coords = box.xyxy[0].tolist()
            if is_on_field(coords, field_mask):
                kept_indices.append(i)

        filtered = len(kept_indices)
        removed = total - filtered
        print(f"  Detected {total} people, kept {filtered} on-field ({removed} filtered out)")

        # Print kept detections
        for rank, i in enumerate(kept_indices):
            box = result.boxes[i]
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            print(f"  [{rank+1}] conf={conf:.2f}  size={w:.0f}x{h:.0f}px")

        # Draw only kept detections on the image
        annotated = image.copy()
        for i in kept_indices:
            box = result.boxes[i]
            conf = box.conf.item()
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 144, 30), 2)
            label = f"person {conf:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 144, 30), 2,
            )

        output_path = output_dir / f"{path.stem}_detections{path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        print(f"  Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect football players in images")
    parser.add_argument(
        "images",
        nargs="*",
        default=["steelers.jpg", "texas.jpg"],
        help="Image paths to process (default: steelers.jpg texas.jpg)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    args = parser.parse_args()
    detect_players(args.images, confidence=args.conf)


if __name__ == "__main__":
    main()
