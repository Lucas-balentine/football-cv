"""
Gradio web UI for the football CV pipeline.

Drag-and-drop images and see outputs from every pipeline stage:
  - Player detection + team classification (YOLOv8m & Roboflow)
  - Field homography (bird's-eye warp)
  - Yard number + directional arrow detection

Usage:
    python app.py          # opens http://localhost:7860
    python app.py --share  # creates a public URL
"""

import base64
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from detect_players import build_field_mask, is_on_field, PERSON_CLASS_ID
from field_homography import (
    detect_white_lines_mask,
    estimate_homography,
)
from field_markings import (
    detect_field_markings, draw_markings_debug, draw_markings_panel,
    detect_white_mask_clean, detect_local_contrast_mask, FieldMarkings,
    _build_player_mask,
)
# field_numbers import removed — using Roboflow model (football-field-tkgnq/13) instead
from interactive_homography import (
    run_interactive_homography,
    draw_field_template,
    yard_to_template_y,
    template_y_to_yard,
    _segmentation_detect,
    _yolo_detect,
    _roboflow_detect,
    _bbox_iou,
    detect_player_positions,
    ROLE_COLORS_BGR,
    TEMPLATE_H,
    TEMPLATE_W,
    _hash_intersection_predict,
    _recover_grid_candidates,
)
from field_homography import TEMPLATE_SCALE, FIELD_WIDTH_YD
from playbook_renderer import render_playbook
from team_classifier import classify_teams, classify_teams_multi

load_dotenv()

# ── Lazy-loaded YOLO model ───────────────────────────────────────────────────

_yolo_model = None


def _get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8m.pt")
    return _yolo_model


# ── Roboflow API helper ──────────────────────────────────────────────────────

_RF_API_URL = "https://detect.roboflow.com/football-presnap-tracker/6"


def _roboflow_predict(image_bgr: np.ndarray, confidence: int, overlap: int = 40) -> list[dict]:
    """Call Roboflow hosted inference API directly (avoids SDK coordinate bugs)."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set in .env")

    _, buf = cv2.imencode(".jpg", image_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")

    url = f"{_RF_API_URL}?api_key={api_key}&confidence={confidence}&overlap={overlap}"
    req = urllib.request.Request(
        url, data=b64.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["predictions"]


# ── BGR <-> RGB helpers ──────────────────────────────────────────────────────

def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _placeholder(h: int = 400, w: int = 600, text: str = "") -> np.ndarray:
    """Return a dark gray placeholder image (RGB)."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    if text:
        cv2.putText(img, text, (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return img


# ── Pipeline stages ──────────────────────────────────────────────────────────

TEAM_COLORS = {
    0: (0, 165, 255),    # orange (BGR)
    1: (255, 50, 50),    # blue
    -1: (180, 180, 180), # gray
}


# ── HSV-based team classification ────────────────────────────────────────────

def hsv_classify_teams(
    crops: list[np.ndarray],
) -> tuple[list[int], list[tuple]]:
    """Classify players into 2 teams using dominant jersey HSV color + whiteness.

    For each crop: take torso region (15-55%, skipping helmet), convert to HSV,
    categorize pixels as white/colored/green/black. Cluster on a 3D feature
    vector [hue_cos, hue_sin, white_ratio] so that white-jersey vs colored-
    jersey teams are properly separated even when helmets match.

    Args:
        crops: list of BGR numpy arrays (may have black-masked background).

    Returns:
        team_labels: list of ints (0, 1, or -1 if unclassifiable).
        dominant_hsv: list of (H, S, V, white_ratio) tuples per player (or None).
    """
    from sklearn.cluster import KMeans

    dominant_hsv: list[tuple | None] = []
    feature_vectors: list[np.ndarray | None] = []

    for crop in crops:
        h_crop, w_crop = crop.shape[:2]
        if h_crop < 10:
            dominant_hsv.append(None)
            feature_vectors.append(None)
            continue

        # Torso region: skip helmet (top 15%) and legs (bottom 45%)
        y_start = max(0, int(h_crop * 0.15))
        y_end = max(y_start + 1, int(h_crop * 0.55))
        jersey = crop[y_start:y_end, :]
        if jersey.size == 0:
            dominant_hsv.append(None)
            feature_vectors.append(None)
            continue

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Pixel categories
        green = ((h_ch >= 35) & (h_ch <= 85)) & (s_ch >= 20) & (v_ch >= 30)
        black = v_ch < 30
        white = ~green & ~black & (s_ch < 40) & (v_ch >= 80)
        colored = ~green & ~black & ~white & ((v_ch > 5) | (s_ch > 5))

        n_white = int(np.count_nonzero(white))
        n_colored = int(np.count_nonzero(colored))
        total_meaningful = n_white + n_colored

        if total_meaningful < 50:
            dominant_hsv.append(None)
            feature_vectors.append(None)
            continue

        white_ratio = n_white / total_meaningful

        # Dominant hue from colored pixels only
        if n_colored >= 30:
            valid_h = h_ch[colored].astype(float)
            valid_s = s_ch[colored].astype(float)
            valid_v = v_ch[colored].astype(float)

            hist, _ = np.histogram(valid_h, bins=180, range=(0, 180))
            kernel = np.ones(5) / 5
            hist_smooth = np.convolve(hist, kernel, mode="same")
            dom_hue = float(np.argmax(hist_smooth))
            dom_s = float(np.mean(valid_s))
            dom_v = float(np.mean(valid_v))
        else:
            # Mostly white jersey
            dom_hue = 0.0
            dom_s = 0.0
            dom_v = float(np.mean(v_ch[white])) if n_white > 0 else 0.0

        dominant_hsv.append((dom_hue, dom_s, dom_v, white_ratio))

        # 3D feature vector: [hue_cos, hue_sin, white_ratio * 2]
        # Scaling white_ratio by 2 gives it comparable weight to hue components
        angle_rad = dom_hue * 2.0 * np.pi / 180.0
        feature_vectors.append(np.array([
            np.cos(angle_rad), np.sin(angle_rad), white_ratio * 2.0,
        ]))

    # Collect classifiable players
    classifiable_indices = [i for i, v in enumerate(feature_vectors) if v is not None]

    if len(classifiable_indices) < 2:
        return [-1] * len(crops), dominant_hsv

    # Stack feature vectors and cluster
    X = np.array([feature_vectors[i] for i in classifiable_indices])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(X)

    labels = [-1] * len(crops)
    for idx, ki in enumerate(classifiable_indices):
        labels[ki] = int(kmeans.labels_[idx])

    return labels, dominant_hsv


def run_detection(
    image_bgr: np.ndarray, conf: float,
) -> tuple[np.ndarray, str]:
    """Player detection + team classification.  Returns (annotated_bgr, summary)."""
    model = _get_yolo()
    field_mask = build_field_mask(image_bgr)

    results = model(image_bgr, conf=conf, classes=[PERSON_CLASS_ID])
    result = results[0]

    # Filter to on-field detections
    kept = []
    for i, box in enumerate(result.boxes):
        if is_on_field(box.xyxy[0].tolist(), field_mask):
            kept.append(i)

    # Extract crops
    crops = []
    for i in kept:
        x1, y1, x2, y2 = [int(v) for v in result.boxes[i].xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)
        crops.append(image_bgr[y1:y2, x1:x2])

    # Classify teams
    if len(crops) >= 2:
        team_labels, _ = classify_teams(crops)
    else:
        team_labels = [-1] * len(crops)

    # Draw
    annotated = image_bgr.copy()
    for rank, i in enumerate(kept):
        box = result.boxes[i]
        conf_val = box.conf.item()
        team = team_labels[rank]
        color = TEAM_COLORS.get(team, (180, 180, 180))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"T{team} {conf_val:.2f}" if team >= 0 else f"??? {conf_val:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Summary text
    total = len(result.boxes)
    on_field = len(kept)
    team_counts = {}
    for t in team_labels:
        team_counts[t] = team_counts.get(t, 0) + 1
    parts = [f"Detected {total} people, {on_field} on-field"]
    for t in sorted(team_counts):
        tag = f"Team {t}" if t >= 0 else "Unclassified"
        parts.append(f"  {tag}: {team_counts[t]}")
    summary = "\n".join(parts)

    return annotated, summary


ROBOFLOW_COLORS = {
    "qb": (0, 0, 255),       # red (BGR)
    "oline": (0, 200, 0),    # green
    "skill": (255, 0, 0),    # blue
    "defense": (0, 165, 255),  # orange
    "ref": (0, 255, 255),    # yellow
}


def run_roboflow_detection(
    image_bgr: np.ndarray, conf: float,
) -> tuple[np.ndarray, str]:
    """Roboflow football-presnap-tracker detection.  Returns (annotated_bgr, summary)."""
    predictions = _roboflow_predict(image_bgr, confidence=int(conf * 100))

    annotated = image_bgr.copy()
    class_counts: dict[str, int] = {}

    for pred in predictions:
        cls = pred["class"]
        c = pred["confidence"]
        x = int(pred["x"] - pred["width"] / 2)
        y = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        color = ROBOFLOW_COLORS.get(cls, (180, 180, 180))
        cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
        label = f"{cls} {c:.2f}"
        cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        class_counts[cls] = class_counts.get(cls, 0) + 1

    parts = [f"Roboflow: {len(predictions)} detections"]
    for cls in sorted(class_counts):
        parts.append(f"  {cls}: {class_counts[cls]}")
    summary = "\n".join(parts)

    return annotated, summary


def run_homography(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Field homography.  Returns (lines_debug, birdseye, blend, summary)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "input.jpg"
        cv2.imwrite(str(tmp_path), image_bgr)
        out = Path(tmpdir)

        H = estimate_homography(str(tmp_path), out)

        def _read(suffix: str) -> np.ndarray | None:
            p = out / f"input_{suffix}.jpg"
            if p.exists():
                return cv2.imread(str(p))
            return None

        lines_img = _read("lines_debug")
        birds_img = _read("birdseye")
        blend_img = _read("blend")

        if H is None:
            msg = "Homography failed (not enough lines detected)"
            ph = _placeholder(text=msg)
            return (
                lines_img if lines_img is not None else ph,
                ph,
                ph,
                msg,
            )

        inliers = "unknown"
        summary = f"Homography computed successfully"
        return (
            lines_img if lines_img is not None else _placeholder(),
            birds_img if birds_img is not None else _placeholder(),
            blend_img if blend_img is not None else _placeholder(),
            summary,
        )


def run_numbers(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Field number detection via Roboflow model.  Returns (annotated, summary)."""
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.getenv("ROBOFLOW_API_KEY", "WCQqMdQpSXVfNVwr7Xcj"),
    )

    # Encode image for API
    _, buf = cv2.imencode(".jpg", image_bgr)
    import tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(buf.tobytes())
        tmp_path = f.name

    try:
        result = client.infer(tmp_path, model_id="football-field-tkgnq/13")
    finally:
        os.unlink(tmp_path)

    preds = result.get("predictions", [])

    # Color map by class type
    CLASS_COLORS = {
        "player": (0, 200, 0),       # green
        "ref": (200, 200, 200),      # gray
        "ball": (0, 165, 255),       # orange
    }
    NUMBER_COLOR = (0, 255, 255)     # yellow for field numbers

    annotated = image_bgr.copy()
    class_counts: dict[str, int] = {}
    number_dets = []

    for p in preds:
        cls = p["class"]
        conf = p["confidence"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

        cx, cy = int(p["x"]), int(p["y"])
        w, h = int(p["width"]), int(p["height"])
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        # Determine color — field number classes have patterns like tl-30, tr-40, bl-10, t-50
        if cls in CLASS_COLORS:
            color = CLASS_COLORS[cls]
            thickness = 2
        else:
            color = NUMBER_COLOR
            thickness = 3
            number_dets.append((cls, conf, cx, cy))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Build summary
    parts = [f"Football Field Model (football-field-tkgnq/13)", f"Total: {len(preds)} detections", ""]
    if number_dets:
        parts.append("Field numbers:")
        for cls, conf, cx, cy in sorted(number_dets, key=lambda x: x[0]):
            parts.append(f"  {cls} @ ({cx}, {cy}) conf={conf:.3f}")
    parts.append("")
    parts.append("All classes:")
    for cls, count in sorted(class_counts.items()):
        parts.append(f"  {cls}: {count}")

    return annotated, "\n".join(parts)


def run_markings(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Field marking detection.  Returns (debug_annotated, debug_panel, white_mask, lc_mask, summary)."""
    field_mask = build_field_mask(image_bgr)
    player_mask = _build_player_mask(image_bgr)

    markings = detect_field_markings(image_bgr, field_mask, player_mask)
    debug = draw_markings_debug(image_bgr, markings)
    panel = draw_markings_panel(image_bgr, markings)

    # Debug masks
    white_mask = detect_white_mask_clean(image_bgr, field_mask, player_mask)
    lc_mask = detect_local_contrast_mask(image_bgr, field_mask, player_mask)

    parts = [
        f"Dominant angle: {markings.dominant_angle:.1f}°",
        f"Yard lines: {len(markings.yard_lines)}",
        f"Sidelines: {len(markings.sidelines)}",
        f"Hash marks: {len(markings.hash_marks)}",
        f"Hash mark rows: {len(markings.hash_rows)}",
    ]
    for row in markings.hash_rows:
        label = row.label or "unlabeled"
        parts.append(f"  Row '{label}': {len(row.marks)} marks, lateral={row.lateral_position:.0f}")
    parts.append(f"Total segments: {len(markings.all_segments)}")
    summary = "\n".join(parts)
    return debug, panel, white_mask, lc_mask, summary


# ── Hash-yards intersection detection ────────────────────────────────────

_HASH_CLASS_COLORS = {
    "hash-yard-intersection": (0, 255, 255),   # cyan
    "hash-mark": (255, 255, 0),                # yellow
    "yard-line": (0, 255, 0),                  # green
}
_HASH_DEFAULT_COLOR = (255, 128, 0)            # orange fallback


def run_hash_intersections(
    image_bgr: np.ndarray,
    high_conf: int = 40,
    low_conf: int = 15,
    snap_radius: float = 40.0,
) -> tuple[np.ndarray, str]:
    """Detect hash-yard intersections via Roboflow model with grid-based recovery.

    1. Query model at *low_conf* to get all possible detections
    2. Split into anchors (>= high_conf) and candidates
    3. Recover candidates that fall on the grid pattern of anchors
    4. Return annotated image + summary

    Returns (annotated_image_bgr, summary_text).
    """
    all_preds = _hash_intersection_predict(image_bgr, confidence=low_conf)

    high_threshold = high_conf / 100.0
    anchors = [p for p in all_preds if p["confidence"] >= high_threshold]
    candidates = [p for p in all_preds if p["confidence"] < high_threshold]

    recovered = _recover_grid_candidates(anchors, candidates, snap_radius)

    # Tag anchors so we can distinguish them in drawing
    for p in anchors:
        p["_recovered"] = False

    final_preds = anchors + recovered

    # ── Draw ──
    annotated = image_bgr.copy()
    class_counts: dict[str, int] = {}
    recovered_count = 0

    for p in final_preds:
        cls = p["class"]
        conf = p["confidence"]
        is_recovered = p.get("_recovered", False)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if is_recovered:
            recovered_count += 1

        color = _HASH_CLASS_COLORS.get(cls, _HASH_DEFAULT_COLOR)
        if is_recovered:
            # Dimmer shade for recovered detections
            color = tuple(max(0, c - 55) for c in color)

        # Roboflow returns center (x, y) + width/height
        cx, cy = int(p["x"]), int(p["y"])
        w, h = int(p["width"]), int(p["height"])
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        thickness = 2 if not is_recovered else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        suffix = " (grid)" if is_recovered else ""
        label = f"{cls} {conf:.2f}{suffix}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Center dot on recovered marks for visibility
        if is_recovered:
            cv2.circle(annotated, (cx, cy), 4, color, -1)

    # ── Summary ──
    anchor_count = len(anchors)
    parts = [
        f"Hash-Yards Intersection Model",
        f"High-confidence (>={high_conf}%): {anchor_count}",
        f"Recovered from grid pattern: {recovered_count}",
        f"Total: {len(final_preds)}",
    ]
    for cls, count in sorted(class_counts.items()):
        parts.append(f"  {cls}: {count}")
    if recovered_count > 0:
        parts.append(f"\nRecovered marks matched the grid pattern of high-confidence detections")
    summary = "\n".join(parts)

    return annotated, summary


# ── Segmentation visualization ───────────────────────────────────────────

SEG_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (100, 255, 255), (255, 100, 255),
    (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
    (255, 200, 150), (150, 255, 200), (200, 150, 255),
]


def run_segmentation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Segmentation model visualization.

    Returns (overlay_image, masks_panel, summary).
    """
    seg_dets = _segmentation_detect(image_bgr)

    if not seg_dets:
        ph = image_bgr.copy()
        cv2.putText(ph, "Segmentation unavailable", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return ph, ph, "Segmentation failed (check API key / inference-sdk)"

    # Overlay: semi-transparent colored masks on the image
    overlay = image_bgr.copy()
    mask_vis = np.zeros_like(image_bgr)
    class_counts = {}

    for i, det in enumerate(seg_dets):
        cls = det["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        color = SEG_COLORS[i % len(SEG_COLORS)]
        mask = det.get("mask")

        x1, y1, x2, y2 = det["bbox"]
        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {det['conf']:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw mask overlay
        if mask is not None:
            colored = np.zeros_like(image_bgr)
            colored[:] = color
            mask_region = mask > 0
            overlay[mask_region] = cv2.addWeighted(
                overlay[mask_region], 0.6,
                colored[mask_region], 0.4, 0,
            )
            mask_vis[mask_region] = color

    parts = [f"Segmentation: {len(seg_dets)} detections"]
    for cls in sorted(class_counts):
        parts.append(f"  {cls}: {class_counts[cls]}")

    return overlay, mask_vis, "\n".join(parts)


# ── Roboflow field segmentation (field + end zones) ─────────────────────────

_FIELD_SEG_API_URL = "https://detect.roboflow.com/football-field-xqbmo/1"

_FIELD_SEG_COLORS = {
    "field":          (0, 200, 0),     # green (BGR)
    "touchdown_zone": (0, 100, 255),   # orange (BGR)
}


def _field_seg_predict(
    image_bgr: np.ndarray,
    confidence: int = 25,
    overlap: int = 40,
) -> list[dict]:
    """Call Roboflow football-field segmentation model.

    Returns list of predictions, each with 'class', 'confidence',
    and 'points' (polygon vertices as [{x, y}, ...]).
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set in .env")

    _, buf = cv2.imencode(".jpg", image_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")

    url = (
        f"{_FIELD_SEG_API_URL}"
        f"?api_key={api_key}&confidence={confidence}&overlap={overlap}"
    )
    req = urllib.request.Request(
        url,
        data=b64.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["predictions"]


def run_field_segmentation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Detect field and end zones via Roboflow segmentation model.

    Returns (overlay_image, masks_panel, summary_text).
    """
    preds = _field_seg_predict(image_bgr)

    if not preds:
        ph = image_bgr.copy()
        cv2.putText(ph, "No field detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return ph, ph, "No field or end zone detected"

    overlay = image_bgr.copy()
    mask_vis = np.zeros_like(image_bgr)
    h, w = image_bgr.shape[:2]
    class_counts: dict[str, list[float]] = {}

    for pred in preds:
        cls = pred["class"]
        conf = pred["confidence"]
        pts_raw = pred.get("points", [])
        if not pts_raw:
            continue

        class_counts.setdefault(cls, []).append(conf)

        # Convert points to numpy polygon
        poly = np.array(
            [[int(p["x"]), int(p["y"])] for p in pts_raw],
            dtype=np.int32,
        )

        color = _FIELD_SEG_COLORS.get(cls, (180, 180, 180))
        alpha = 0.25 if cls == "field" else 0.4

        # Semi-transparent filled polygon on overlay
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 255)
        colored = np.zeros_like(image_bgr)
        colored[:] = color
        region = poly_mask > 0
        overlay[region] = cv2.addWeighted(
            overlay[region], 1.0 - alpha,
            colored[region], alpha, 0,
        )

        # Polygon outline
        cv2.polylines(overlay, [poly], True, color, 2)

        # Masks-only panel
        mask_vis[region] = color

        # Label
        cx = int(pred["x"])
        cy = int(pred["y"])
        label = f"{cls} {conf:.0%}"
        cv2.putText(overlay, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Summary
    parts = [f"Field segmentation: {len(preds)} detection(s)"]
    for cls in sorted(class_counts):
        confs = class_counts[cls]
        conf_str = ", ".join(f"{c:.0%}" for c in confs)
        parts.append(f"  {cls}: {len(confs)} ({conf_str})")

    return overlay, mask_vis, "\n".join(parts)


# ── Team assignment via segmentation masks ───────────────────────────────────

SEG_TEAM_COLORS = {
    0: (0, 165, 255),    # orange (BGR) — Team A
    1: (255, 50, 50),    # blue (BGR)   — Team B
    -1: (180, 180, 180), # gray         — unclassified
}


def _build_team_overlay(
    image_bgr: np.ndarray,
    players: list[dict],
    labels: list[int],
    method_prefix: str,
    diagnostics: list[dict] | None = None,
    dominant_hsv: list[tuple | None] | None = None,
) -> np.ndarray:
    """Draw bounding boxes + mask tints colored by team label on the image.

    Args:
        image_bgr: original image.
        players: list of player dicts with 'bbox' and optional 'mask'.
        labels: team label per player (0, 1, or -1).
        method_prefix: "C" for Color-clustering, "H" for HSV — used in on-image labels.
        diagnostics: optional list of per-player diagnostic dicts (position + color).
        dominant_hsv: optional HSV tuples for the HSV method overlay.
    """
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    for i, (p, lbl) in enumerate(zip(players, labels)):
        color = SEG_TEAM_COLORS.get(lbl, (180, 180, 180))
        x1, y1, x2, y2 = p["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        # Semi-transparent mask fill if available
        mask = p.get("mask")
        if mask is not None:
            region = mask > 0
            colored = np.zeros_like(image_bgr)
            colored[:] = color
            overlay[region] = cv2.addWeighted(
                overlay[region], 0.7,
                colored[region], 0.3, 0,
            )

        # Bounding box
        cv2.rectangle(overlay, (x1c, y1c), (x2c, y2c), color, 2)

        # Label text
        if method_prefix == "C" and diagnostics:
            diag = diagnostics[i]
            team_name = "OFF" if lbl == 0 else ("DEF" if lbl == 1 else "?")
            # Show which signals contributed: P=position zone, C=color
            signals = ""
            zone = diag.get("position_zone")
            if zone is not None:
                signals += zone[0].upper()  # L/N/M/D for lineman/near/mid/deep
            if diag["color_signal"] is not None:
                signals += "C"
            conf = int(diag["confidence"] * 100)
            label_text = f"{team_name} [{signals}] {conf}%"
        elif method_prefix == "H" and dominant_hsv and dominant_hsv[i] is not None:
            label_text = f"H:T{lbl} H{int(dominant_hsv[i][0])}" if lbl >= 0 else "H:?"
        else:
            label_text = f"{method_prefix}:T{lbl}" if lbl >= 0 else f"{method_prefix}:?"

        # Draw text with outline for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        tx, ty = x1c, max(y1c - 8, 15)
        cv2.putText(overlay, label_text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(overlay, label_text, (tx, ty), font, font_scale, color, thickness)

    return overlay


def _build_crops_panel(
    crops: list[np.ndarray],
    multi_labels: list[int],
    hsv_labels: list[int],
    diagnostics: list[dict],
    dominant_hsv: list[tuple | None],
    cols: int = 5,
) -> np.ndarray:
    """Build a grid of player crops annotated with team labels.

    Each crop gets a colored border (left half = color-clustering, right half = HSV color)
    and a text label showing signal breakdown.
    """
    if not crops:
        return np.zeros((200, 400, 3), dtype=np.uint8)

    # Uniform cell size
    cell_w, cell_h = 100, 180
    label_h = 35  # space for text below crop
    total_cell_h = cell_h + label_h
    rows = (len(crops) + cols - 1) // cols
    panel = np.zeros((rows * total_cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, crop in enumerate(crops):
        r, c = divmod(idx, cols)
        # Resize crop to fit cell (preserving aspect ratio within bounds)
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            continue
        scale = min((cell_w - 8) / cw, (cell_h - 8) / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh))

        # Center in cell
        y_off = r * total_cell_h + (cell_h - nh) // 2
        x_off = c * cell_w + (cell_w - nw) // 2
        panel[y_off:y_off + nh, x_off:x_off + nw] = resized

        # Colored border: left half = color-clustering, right half = HSV color
        m_color = SEG_TEAM_COLORS.get(multi_labels[idx], (180, 180, 180))
        h_color = SEG_TEAM_COLORS.get(hsv_labels[idx], (180, 180, 180))
        cell_x = c * cell_w
        cell_y = r * total_cell_h
        # Left border (color-clustering)
        cv2.rectangle(panel, (cell_x, cell_y), (cell_x + cell_w // 2 - 1, cell_y + cell_h - 1), m_color, 2)
        # Right border (HSV)
        cv2.rectangle(panel, (cell_x + cell_w // 2, cell_y), (cell_x + cell_w - 1, cell_y + cell_h - 1), h_color, 2)

        # Label line 1: zone + signals
        diag = diagnostics[idx]
        signals = ""
        zone = diag.get("position_zone")
        if zone is not None:
            signals += zone[0].upper()  # L/N/M/D for lineman/near/mid/deep
        if diag["color_signal"] is not None:
            signals += "C"
        team_name = "OFF" if multi_labels[idx] == 0 else ("DEF" if multi_labels[idx] == 1 else "?")
        line1 = f"{team_name}[{signals}] H:{hsv_labels[idx]}"
        ty = cell_y + cell_h + 14
        tx = cell_x + 3
        cv2.putText(panel, line1, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)

        # Label line 2: hue + confidence
        conf = int(diag["confidence"] * 100)
        hue_str = ""
        if dominant_hsv[idx] is not None:
            hue_str = f"H{int(dominant_hsv[idx][0])} "
        line2 = f"{hue_str}{conf}%"
        if diag["conflict"]:
            line2 += " !"
        cv2.putText(panel, line2, (tx, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 140), 1)

    return panel


def run_team_assignment(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[tuple[tuple[int, int, int, int], int]]]:
    """Run color-only team assignment using segmentation masks.

    Uses _segmentation_detect() for player masks (no presnap roles).
    Runs classify_teams_multi() with no ball_yard/direction, which falls
    through to unsupervised K-Means on [hue, white_ratio]. Compares
    against the standalone hsv_classify_teams() for reference.

    Returns (color_overlay, hsv_overlay, crops_panel, summary_text, bbox_labels).
    bbox_labels is a list of (bbox_xyxy, team_label) for passing to the
    Field Mapping tab so team assignments stay consistent.
    """
    # Step 1: Get player detections with masks (seg only, no roles)
    seg_dets = _segmentation_detect(image_bgr)

    if not seg_dets:
        ph = image_bgr.copy()
        cv2.putText(ph, "Segmentation unavailable", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        empty_panel = np.zeros((200, 400, 3), dtype=np.uint8)
        return ph, ph.copy(), empty_panel, "Segmentation failed — no detections", []

    # Filter to players only (exclude refs)
    players = [
        {"bbox": d["bbox"], "conf": d.get("conf", 0.5),
         "source": "seg", "mask": d.get("mask")}
        for d in seg_dets if d.get("class") == "player"
    ]

    if len(players) < 2:
        ph = image_bgr.copy()
        cv2.putText(ph, f"Only {len(players)} player(s) — need >= 2", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        empty_panel = np.zeros((200, 400, 3), dtype=np.uint8)
        return ph, ph.copy(), empty_panel, f"Only {len(players)} player(s) detected — need at least 2 for clustering", []

    # Step 2: Color-only classification (no position, unsupervised K-Means)
    multi_labels, diagnostics = classify_teams_multi(
        players, image_bgr, ball_yard=None, offense_direction=None,
    )

    # Step 3: Extract masked crops for HSV comparison
    h, w = image_bgr.shape[:2]
    masked_crops = []
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = image_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            masked_crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
            continue
        player_mask = p.get("mask")
        if player_mask is not None:
            crop_mask = player_mask[y1c:y2c, x1c:x2c]
            crop = crop.copy()
            crop[crop_mask == 0] = 0
        masked_crops.append(crop)

    # Step 4: HSV-only classification for comparison
    hsv_labels, dominant_hsv = hsv_classify_teams(masked_crops)

    # Step 5: Build overlays
    color_overlay = _build_team_overlay(
        image_bgr, players, multi_labels, "C", diagnostics=diagnostics,
    )
    hsv_overlay = _build_team_overlay(
        image_bgr, players, hsv_labels, "H", dominant_hsv=dominant_hsv,
    )

    # Step 6: Build crops panel
    crops_panel = _build_crops_panel(
        masked_crops, multi_labels, hsv_labels, diagnostics, dominant_hsv,
    )

    # Step 7: Build summary
    total = len(players)

    m_t0 = sum(1 for l in multi_labels if l == 0)
    m_t1 = sum(1 for l in multi_labels if l == 1)
    m_unk = sum(1 for l in multi_labels if l == -1)

    h_t0 = sum(1 for l in hsv_labels if l == 0)
    h_t1 = sum(1 for l in hsv_labels if l == 1)
    h_unk = sum(1 for l in hsv_labels if l == -1)

    color_used = sum(1 for d in diagnostics if d["color_signal"] is not None)

    parts = [
        f"Detected: {total} players (segmentation only)",
        "",
        f"Color clustering: T0={m_t0}, T1={m_t1}, unknown={m_unk}",
        f"  Color signals: {color_used}/{total}",
        f"  (Unsupervised — cluster IDs are arbitrary, no offense/defense semantics)",
        f"  (Use Field Mapping tab with LOS + direction for offense/defense labels)",
        "",
        f"HSV-only: T0={h_t0}, T1={h_t1}, unknown={h_unk}",
    ]

    # Build bbox→label pairs for passing to Field Mapping tab
    bbox_labels = [
        (tuple(p["bbox"]), int(label))
        for p, label in zip(players, multi_labels)
    ]

    return color_overlay, hsv_overlay, crops_panel, "\n".join(parts), bbox_labels


def run_blended_comparison(
    image_bgr: np.ndarray,
    conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Side-by-side comparison of all three models + merged view.

    Returns (yolo_img, presnap_img, seg_img, blended_img, summary).
    """
    h, w = image_bgr.shape[:2]

    # Run all three
    yolo_dets = _yolo_detect(image_bgr, conf)
    rf_dets = _roboflow_detect(image_bgr, confidence=int(conf * 100))
    seg_dets = _segmentation_detect(image_bgr)

    # ── YOLO-only view ──
    yolo_img = image_bgr.copy()
    for d in yolo_dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(yolo_img, f"{d['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(yolo_img, f"YOLO: {len(yolo_dets)} people", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── Presnap-only view ──
    presnap_img = image_bgr.copy()
    for d in rf_dets:
        x1, y1, x2, y2 = d["bbox"]
        color = ROBOFLOW_COLORS.get(d["role"], (180, 180, 180))
        cv2.rectangle(presnap_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(presnap_img, f"{d['role']} {d['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(presnap_img, f"Presnap: {len(rf_dets)} detections", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ── Segmentation-only view ──
    seg_img = image_bgr.copy()
    for i, d in enumerate(seg_dets):
        x1, y1, x2, y2 = d["bbox"]
        color = SEG_COLORS[i % len(SEG_COLORS)]
        cv2.rectangle(seg_img, (x1, y1), (x2, y2), color, 2)
        mask = d.get("mask")
        if mask is not None:
            colored = np.zeros_like(image_bgr)
            colored[:] = color
            region = mask > 0
            seg_img[region] = cv2.addWeighted(
                seg_img[region], 0.6, colored[region], 0.4, 0,
            )
    cv2.putText(seg_img, f"Segmentation: {len(seg_dets)} players", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

    # ── Blended view: color by source ──
    # Match YOLO ↔ presnap
    yolo_matched_rf = set()
    rf_matched = set()
    for ri, rd in enumerate(rf_dets):
        for yi, yd in enumerate(yolo_dets):
            if yi in yolo_matched_rf:
                continue
            if _bbox_iou(rd["bbox"], yd["bbox"]) >= 0.3:
                yolo_matched_rf.add(yi)
                rf_matched.add(ri)
                break
    # Match YOLO ↔ seg
    yolo_matched_seg = set()
    seg_matched = set()
    for si, sd in enumerate(seg_dets):
        for yi, yd in enumerate(yolo_dets):
            if yi in yolo_matched_seg:
                continue
            if _bbox_iou(sd["bbox"], yd["bbox"]) >= 0.3:
                yolo_matched_seg.add(yi)
                seg_matched.add(si)
                break

    blended_img = image_bgr.copy()
    # Source colors: green=all three, cyan=yolo+presnap, yellow=yolo+seg,
    # blue=presnap only, red=seg only, white=yolo only
    source_counts = {"all_three": 0, "yolo+presnap": 0, "yolo+seg": 0,
                     "yolo_only": 0, "presnap_only": 0, "seg_only": 0}

    for yi, yd in enumerate(yolo_dets):
        x1, y1, x2, y2 = yd["bbox"]
        has_rf = yi in yolo_matched_rf
        has_seg = yi in yolo_matched_seg
        if has_rf and has_seg:
            color, tag = (0, 255, 0), "3x"
            source_counts["all_three"] += 1
        elif has_rf:
            color, tag = (255, 255, 0), "Y+P"
            source_counts["yolo+presnap"] += 1
        elif has_seg:
            color, tag = (0, 255, 255), "Y+S"
            source_counts["yolo+seg"] += 1
        else:
            color, tag = (255, 255, 255), "Y"
            source_counts["yolo_only"] += 1
        cv2.rectangle(blended_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(blended_img, tag, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for ri, rd in enumerate(rf_dets):
        if ri in rf_matched:
            continue
        x1, y1, x2, y2 = rd["bbox"]
        cv2.rectangle(blended_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(blended_img, "P", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        source_counts["presnap_only"] += 1

    for si, sd in enumerate(seg_dets):
        if si in seg_matched:
            continue
        x1, y1, x2, y2 = sd["bbox"]
        cv2.rectangle(blended_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(blended_img, "S", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        source_counts["seg_only"] += 1

    total = sum(source_counts.values())
    cv2.putText(blended_img, f"Blended: {total} total", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Summary
    parts = [
        f"YOLO: {len(yolo_dets)} | Presnap: {len(rf_dets)} | Seg: {len(seg_dets)}",
        f"Blended total: {total}",
    ]
    for src, cnt in sorted(source_counts.items()):
        if cnt > 0:
            parts.append(f"  {src}: {cnt}")

    return yolo_img, presnap_img, seg_img, blended_img, "\n".join(parts)


# ── Game Setup: LOS & first down line annotation ────────────────────────

def _draw_nfl_line(
    image: np.ndarray,
    point: tuple[int, int],
    angle_deg: float,
    color_bgr: tuple,
    label: str,
    alpha: float = 0.4,
    field_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a semi-transparent NFL-style line from the sideline point inward.

    The line starts at the clicked sideline point and extends only across
    the green field area (stops at the far sideline or image edge).
    """
    h, w = image.shape[:2]
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)

    # Build field mask if not provided (for clipping the line to the field)
    if field_mask is None:
        from field_homography import build_field_mask
        field_mask = build_field_mask(image)

    # Sample both directions from the point, find how far the field extends
    t_max = max(h, w) * 2
    num_samples = 500

    def _find_field_extent(sign):
        """Walk along the line in one direction, return the furthest on-field point."""
        best_t = 0
        for i in range(num_samples):
            t = sign * (i / num_samples) * t_max
            sx = int(point[0] + t * dx)
            sy = int(point[1] + t * dy)
            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                break
            if field_mask[sy, sx] > 0:
                best_t = t
        return best_t

    t_pos = _find_field_extent(+1)
    t_neg = _find_field_extent(-1)

    # Use the full field extent (from negative end to positive end)
    x1 = int(point[0] + t_neg * dx)
    y1 = int(point[1] + t_neg * dy)
    x2 = int(point[0] + t_pos * dx)
    y2 = int(point[1] + t_pos * dy)

    # Draw the thick line on an overlay for transparency
    overlay = image.copy()
    cv2.line(overlay, (x1, y1), (x2, y2), color_bgr, 6)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw a thin solid center line
    cv2.line(result, (x1, y1), (x2, y2), color_bgr, 2)

    # Label near the clicked point
    lx = point[0] + int(20 * dy)
    ly = point[1] - int(20 * dx)
    cv2.putText(result, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    return result


# Hash mark physical positions (distance from near sideline in yards)
HASH_POSITIONS = {
    "college": (20.0, 33.33),   # 60ft (20 yd) from each sideline per NCAA rules
    "nfl": (23.58, 29.75),      # 70'9" from each sideline
}


def compute_game_setup_data(
    los_point: tuple[int, int],
    fd_point: tuple[int, int],
    los_yard: int,
    fd_yard: int,
    direction: str,
    image_bgr: np.ndarray,
    near_hash_point: tuple[int, int] | None = None,
    far_hash_point: tuple[int, int] | None = None,
    field_type: str = "college",
    markings: "FieldMarkings | None" = None,
) -> dict:
    """Compute all derived data from sideline + hash mark points.

    Correspondence points for homography:
      - LOS sideline point → (0, los_yard) on template
      - 1st down sideline point → (0, fd_yard) on template
      - Near hash mark → (near_hash_x, los_yard) on template
      - Far hash mark → (far_hash_x, los_yard) on template
      - Detected yard lines (if available) → full-width line correspondences
    """
    dx = fd_point[0] - los_point[0]
    dy = fd_point[1] - los_point[1]
    pixel_dist = np.hypot(dx, dy)
    yard_dist = abs(fd_yard - los_yard)

    sideline_angle = np.degrees(np.arctan2(dy, dx)) % 180
    yard_line_angle = (sideline_angle + 90) % 180
    pixels_per_yard = pixel_dist / yard_dist if yard_dist > 0 else 0

    los_template_y = yard_to_template_y(los_yard)
    fd_template_y = yard_to_template_y(fd_yard)

    near_hash_yd, far_hash_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    near_hash_template_x = int(near_hash_yd * TEMPLATE_SCALE)
    far_hash_template_x = int(far_hash_yd * TEMPLATE_SCALE)

    # Build correspondence points
    homography_points = [
        (los_point, (0, los_template_y)),
        (fd_point, (0, fd_template_y)),
    ]

    # Hash mark correspondences — at the LOS yardage
    if near_hash_point is not None:
        homography_points.append(
            (near_hash_point, (near_hash_template_x, los_template_y))
        )
    if far_hash_point is not None:
        homography_points.append(
            (far_hash_point, (far_hash_template_x, los_template_y))
        )

    # Detected yard line correspondences — each yard line at a known yardage
    # gives a point on the sideline (x=0 on template).
    # Assign yard numbers by measuring perpendicular distance from the LOS
    # in units of 5-yard gaps.
    if markings and markings.yard_lines and pixels_per_yard > 0:
        # Direction vector along the sideline (LOS → 1st down)
        side_dx = dx / pixel_dist if pixel_dist > 0 else 0
        side_dy = dy / pixel_dist if pixel_dist > 0 else 0
        five_yd_px = pixels_per_yard * 5

        for yl in markings.yard_lines:
            ext = yl.extended
            if ext is None or len(ext) != 4:
                continue
            mx = (ext[0] + ext[2]) / 2
            my = (ext[1] + ext[3]) / 2
            # Project yard line midpoint onto sideline direction
            proj = (mx - los_point[0]) * side_dx + (my - los_point[1]) * side_dy
            # How many 5-yard increments from LOS?
            increments = round(proj / five_yd_px) if five_yd_px > 0 else 0
            # Determine the yard number
            if direction == "right":
                yl_yard = los_yard + increments * 5
            else:
                yl_yard = los_yard - increments * 5

            if yl_yard < 0 or yl_yard > 100:
                continue

            yl_template_y = yard_to_template_y(yl_yard)
            # Use the intersection of the yard line with the sideline edge
            # (the endpoint closest to the sideline / los_point)
            p1 = np.array([ext[0], ext[1]], dtype=float)
            p2 = np.array([ext[2], ext[3]], dtype=float)
            los_arr = np.array(los_point, dtype=float)
            # Pick the endpoint closest to the LOS sideline point
            if np.linalg.norm(p1 - los_arr) < np.linalg.norm(p2 - los_arr):
                yl_img_pt = (int(p1[0]), int(p1[1]))
            else:
                yl_img_pt = (int(p2[0]), int(p2[1]))

            homography_points.append(
                (yl_img_pt, (0, yl_template_y))
            )

    return {
        "los_point": los_point,
        "fd_point": fd_point,
        "los_yard": los_yard,
        "fd_yard": fd_yard,
        "distance": yard_dist,
        "sideline_angle": sideline_angle,
        "yard_line_angle": yard_line_angle,
        "pixels_per_yard": pixels_per_yard,
        "direction": direction,
        "field_type": field_type,
        "los_template_pos": (0, los_template_y),
        "fd_template_pos": (0, fd_template_y),
        "near_hash_point": near_hash_point,
        "far_hash_point": far_hash_point,
        "near_hash_template_x": near_hash_template_x,
        "far_hash_template_x": far_hash_template_x,
        "homography_points": homography_points,
    }


def _nearest_yard_line_angle(
    point: tuple[int, int],
    markings: FieldMarkings,
) -> float | None:
    """Find the angle of the detected yard line nearest to a point.

    In perspective, each yard line has a slightly different angle.
    Returns the angle of the closest yard line's representative segment,
    or None if no yard lines exist.
    """
    if not markings.yard_lines:
        return None

    px, py = point
    best_dist = float('inf')
    best_angle = None

    for yl in markings.yard_lines:
        ex = yl.extended
        if len(ex) != 4:
            continue
        # Distance from point to this line segment
        x1, y1, x2, y2 = ex.astype(float)
        # Project point onto line
        ldx, ldy = x2 - x1, y2 - y1
        line_len_sq = ldx * ldx + ldy * ldy
        if line_len_sq < 1:
            continue
        t = max(0, min(1, ((px - x1) * ldx + (py - y1) * ldy) / line_len_sq))
        proj_x = x1 + t * ldx
        proj_y = y1 + t * ldy
        dist = np.hypot(px - proj_x, py - proj_y)

        if dist < best_dist:
            best_dist = dist
            # Angle of this specific yard line
            best_angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180

    return best_angle


def draw_game_setup(
    image_bgr: np.ndarray,
    los_point: tuple[int, int] | None,
    fd_point: tuple[int, int] | None,
    los_yard: int,
    fd_yard: int,
    direction: str,
    markings: FieldMarkings | None = None,
    yard_line_angle: float | None = None,
    near_hash_point: tuple[int, int] | None = None,
    far_hash_point: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw NFL-style LOS (blue) and first down (yellow) lines on the image.

    Uses the angle of the nearest detected yard line for each point, so
    lines match the local perspective. Falls back to the global dominant
    angle if per-line angles aren't available.
    Also draws hash mark indicator dots (green) if provided.
    """
    result = image_bgr.copy()

    # Auto-detect yard lines if not provided
    if markings is None and (los_point or fd_point):
        try:
            markings = detect_field_markings(image_bgr)
        except Exception:
            markings = None

    # Global fallback angle
    if yard_line_angle is None:
        if markings is not None:
            yard_line_angle = markings.dominant_angle
        else:
            yard_line_angle = 90.0

    # Build field mask once for clipping lines to the field
    from field_homography import build_field_mask
    field_mask = build_field_mask(image_bgr)

    if los_point is not None:
        # Use the angle of the nearest yard line for perspective-correct drawing
        los_angle = yard_line_angle
        if markings is not None:
            local = _nearest_yard_line_angle(los_point, markings)
            if local is not None:
                los_angle = local

        result = _draw_nfl_line(
            result, los_point, los_angle,
            color_bgr=(255, 100, 0),  # blue
            label=f"LOS {los_yard}yd",
            alpha=0.45,
            field_mask=field_mask,
        )
        # Draw sideline marker dot
        cv2.circle(result, los_point, 8, (255, 100, 0), -1)
        cv2.circle(result, los_point, 8, (255, 255, 255), 2)

    if fd_point is not None:
        fd_angle = yard_line_angle
        if markings is not None:
            local = _nearest_yard_line_angle(fd_point, markings)
            if local is not None:
                fd_angle = local

        result = _draw_nfl_line(
            result, fd_point, fd_angle,
            color_bgr=(0, 255, 255),  # yellow
            label=f"1st Down {fd_yard}yd",
            alpha=0.45,
            field_mask=field_mask,
        )
        # Draw sideline marker dot
        cv2.circle(result, fd_point, 8, (0, 255, 255), -1)
        cv2.circle(result, fd_point, 8, (255, 255, 255), 2)

    # Draw hash mark indicator dots (green)
    if near_hash_point is not None:
        cv2.circle(result, near_hash_point, 10, (0, 220, 0), -1)
        cv2.circle(result, near_hash_point, 10, (255, 255, 255), 2)
        cv2.putText(result, "Near Hash", (near_hash_point[0] + 14, near_hash_point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
    if far_hash_point is not None:
        cv2.circle(result, far_hash_point, 10, (0, 180, 0), -1)
        cv2.circle(result, far_hash_point, 10, (255, 255, 255), 2)
        cv2.putText(result, "Far Hash", (far_hash_point[0] + 14, far_hash_point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

    # Draw sideline between the two points
    if los_point is not None and fd_point is not None:
        cv2.line(result, los_point, fd_point, (255, 255, 255), 2)
        # Direction arrow
        mid_x = (los_point[0] + fd_point[0]) // 2
        mid_y = (los_point[1] + fd_point[1]) // 2
        dx = fd_point[0] - los_point[0]
        dy = fd_point[1] - los_point[1]
        norm = np.hypot(dx, dy)
        if norm > 0:
            # Arrow pointing in direction of play
            if direction == "right":
                ax, ay = int(dx / norm * 30), int(dy / norm * 30)
            else:
                ax, ay = int(-dx / norm * 30), int(-dy / norm * 30)
            cv2.arrowedLine(result, (mid_x - ax, mid_y - ay),
                           (mid_x + ax, mid_y + ay),
                           (255, 255, 255), 3, tipLength=0.4)

    return result


def draw_game_setup_template(
    los_yard: int,
    fd_yard: int,
    direction: str,
    field_type: str = "college",
    near_hash_point: tuple[int, int] | None = None,
    far_hash_point: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw the field template with LOS, first down lines, and hash marks."""
    template = draw_field_template(field_type=field_type)

    los_y = yard_to_template_y(los_yard)
    fd_y = yard_to_template_y(fd_yard)

    near_hash_yd, far_hash_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    near_hash_x = int(near_hash_yd * TEMPLATE_SCALE)
    far_hash_x = int(far_hash_yd * TEMPLATE_SCALE)

    # Draw hash mark lateral lines (thin dashed green) across the full template height
    for hx in (near_hash_x, far_hash_x):
        for y in range(0, TEMPLATE_H, 8):
            cv2.line(template, (hx, y), (hx, min(y + 4, TEMPLATE_H - 1)), (0, 140, 0), 1)

    # LOS line (blue)
    overlay = template.copy()
    cv2.line(overlay, (0, los_y), (TEMPLATE_W - 1, los_y), (255, 100, 0), 4)
    template = cv2.addWeighted(overlay, 0.5, template, 0.5, 0)
    cv2.line(template, (0, los_y), (TEMPLATE_W - 1, los_y), (255, 100, 0), 2)
    cv2.putText(template, f"LOS {los_yard}", (TEMPLATE_W // 2 - 30, los_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # First down line (yellow)
    overlay = template.copy()
    cv2.line(overlay, (0, fd_y), (TEMPLATE_W - 1, fd_y), (0, 255, 255), 4)
    template = cv2.addWeighted(overlay, 0.5, template, 0.5, 0)
    cv2.line(template, (0, fd_y), (TEMPLATE_W - 1, fd_y), (0, 255, 255), 2)
    cv2.putText(template, f"1st {fd_yard}", (TEMPLATE_W // 2 - 30, fd_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Hash mark dots at LOS yardage (bright green if marked)
    if near_hash_point is not None:
        cv2.circle(template, (near_hash_x, los_y), 6, (0, 220, 0), -1)
        cv2.circle(template, (near_hash_x, los_y), 6, (255, 255, 255), 2)
    if far_hash_point is not None:
        cv2.circle(template, (far_hash_x, los_y), 6, (0, 220, 0), -1)
        cv2.circle(template, (far_hash_x, los_y), 6, (255, 255, 255), 2)

    # Direction arrow
    arrow_y = (los_y + fd_y) // 2
    if direction == "right":
        cv2.arrowedLine(template, (TEMPLATE_W // 2 - 40, arrow_y),
                       (TEMPLATE_W // 2 + 40, arrow_y),
                       (255, 255, 255), 2, tipLength=0.3)
    else:
        cv2.arrowedLine(template, (TEMPLATE_W // 2 + 40, arrow_y),
                       (TEMPLATE_W // 2 - 40, arrow_y),
                       (255, 255, 255), 2, tipLength=0.3)

    # Shade the area between LOS and first down
    y_top = min(los_y, fd_y)
    y_bot = max(los_y, fd_y)
    overlay = template.copy()
    cv2.rectangle(overlay, (0, y_top), (TEMPLATE_W - 1, y_bot), (0, 200, 255), -1)
    template = cv2.addWeighted(overlay, 0.15, template, 0.85, 0)

    return template


# ── Top-level runner ─────────────────────────────────────────────────────────

def run_pipeline(
    image_rgb: np.ndarray, conf: float,
) -> tuple:
    """Run all pipeline stages.  Gradio passes RGB; we convert internally."""
    if image_rgb is None:
        ph = _placeholder(text="Drop an image above")
        empty = ""
        return (ph, empty, ph, empty, ph, ph, ph, empty,
                ph, ph, ph, ph, empty, ph, empty,
                ph, empty,
                ph, ph, empty,
                ph, ph, empty,
                ph, ph, ph, empty,
                ph, ph, ph, ph, empty,
                [])

    image_bgr = _rgb_to_bgr(image_rgb)

    # Player detection (YOLOv8m)
    try:
        det_img, det_text = run_detection(image_bgr, conf)
        det_img = _bgr_to_rgb(det_img)
    except Exception as e:
        det_img = _placeholder(text=f"Detection error: {e}")
        det_text = str(e)

    # Player detection (Roboflow)
    try:
        rf_img, rf_text = run_roboflow_detection(image_bgr, conf)
        rf_img = _bgr_to_rgb(rf_img)
    except Exception as e:
        rf_img = _placeholder(text=f"Roboflow error: {e}")
        rf_text = str(e)

    # Homography
    try:
        lines_img, birds_img, blend_img, hom_text = run_homography(image_bgr)
        lines_img = _bgr_to_rgb(lines_img)
        birds_img = _bgr_to_rgb(birds_img)
        blend_img = _bgr_to_rgb(blend_img)
    except Exception as e:
        ph = _placeholder(text=f"Homography error: {e}")
        lines_img = birds_img = blend_img = ph
        hom_text = str(e)

    # Field markings
    try:
        mk_img, mk_panel_img, mk_white_img, mk_lc_img, mk_text = run_markings(image_bgr)
        mk_img = _bgr_to_rgb(mk_img)
        mk_panel_img = _bgr_to_rgb(mk_panel_img)
    except Exception as e:
        mk_img = _placeholder(text=f"Markings error: {e}")
        mk_panel_img = mk_img
        mk_white_img = mk_img
        mk_lc_img = mk_img
        mk_text = str(e)

    # Hash-yards intersections (Roboflow model)
    try:
        hash_img, hash_text = run_hash_intersections(image_bgr)
        hash_img = _bgr_to_rgb(hash_img)
    except Exception as e:
        hash_img = _placeholder(text=f"Hash intersection error: {e}")
        hash_text = str(e)

    # Field numbers (Roboflow model)
    try:
        num_img, num_text = run_numbers(image_bgr)
        num_img = _bgr_to_rgb(num_img)
    except Exception as e:
        num_img = _placeholder(text=f"Numbers error: {e}")
        num_text = str(e)

    # Segmentation
    try:
        seg_overlay, seg_masks, seg_text = run_segmentation(image_bgr)
        seg_overlay = _bgr_to_rgb(seg_overlay)
        seg_masks = _bgr_to_rgb(seg_masks)
    except Exception as e:
        seg_overlay = _placeholder(text=f"Segmentation error: {e}")
        seg_masks = seg_overlay
        seg_text = str(e)

    # Field segmentation (field + end zones)
    try:
        fseg_overlay, fseg_masks, fseg_text = run_field_segmentation(image_bgr)
        fseg_overlay = _bgr_to_rgb(fseg_overlay)
        fseg_masks = _bgr_to_rgb(fseg_masks)
    except Exception as e:
        fseg_overlay = _placeholder(text=f"Field seg error: {e}")
        fseg_masks = fseg_overlay
        fseg_text = str(e)

    # Team assignment (segmentation-based)
    ta_bbox_labels = []
    try:
        ta_resnet, ta_hsv, ta_crops, ta_text, ta_bbox_labels = run_team_assignment(image_bgr)
        ta_resnet = _bgr_to_rgb(ta_resnet)
        ta_hsv = _bgr_to_rgb(ta_hsv)
        ta_crops = _bgr_to_rgb(ta_crops)
    except Exception as e:
        ta_resnet = _placeholder(text=f"Team assignment error: {e}")
        ta_hsv = ta_resnet
        ta_crops = ta_resnet
        ta_text = str(e)

    # Blended comparison
    try:
        b_yolo, b_presnap, b_seg, b_blend, blend_text = run_blended_comparison(
            image_bgr, conf,
        )
        b_yolo = _bgr_to_rgb(b_yolo)
        b_presnap = _bgr_to_rgb(b_presnap)
        b_seg = _bgr_to_rgb(b_seg)
        b_blend = _bgr_to_rgb(b_blend)
    except Exception as e:
        ph = _placeholder(text=f"Blend error: {e}")
        b_yolo = b_presnap = b_seg = b_blend = ph
        blend_text = str(e)

    return (
        det_img, det_text,
        rf_img, rf_text,
        lines_img, birds_img, blend_img, hom_text,
        mk_img, mk_panel_img, mk_white_img, mk_lc_img, mk_text,
        hash_img, hash_text,
        num_img, num_text,
        seg_overlay, seg_masks, seg_text,
        fseg_overlay, fseg_masks, fseg_text,
        ta_resnet, ta_hsv, ta_crops, ta_text,
        b_yolo, b_presnap, b_seg, b_blend, blend_text,
        ta_bbox_labels,
    )


# ── Gradio interface ─────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Football CV Pipeline") as demo:
        gr.Markdown("# Football CV Pipeline")
        gr.Markdown(
            "Drop a sideline football image below and click **Run** to see "
            "player detection, field homography, and yard-number detection."
        )

        with gr.Row():
            input_image = gr.Image(
                type="numpy", label="Input image",
                height=360,
            )
            with gr.Column(scale=0, min_width=200):
                field_type_radio = gr.Radio(
                    choices=["college", "nfl"],
                    value="college",
                    label="Field type",
                )
                conf_slider = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05,
                    label="YOLO confidence",
                )
                run_btn = gr.Button("Run Pipeline", variant="primary")

        with gr.Tabs():
            # ── Tab 1: YOLOv8m player detection ──────────────────────────
            with gr.TabItem("YOLOv8m + Teams"):
                det_output = gr.Image(label="Team-colored detections")
                det_summary = gr.Textbox(label="Summary", lines=4)

            # ── Tab 2: Roboflow player detection ─────────────────────────
            with gr.TabItem("Roboflow Presnap"):
                rf_output = gr.Image(label="Role-classified detections")
                rf_summary = gr.Textbox(label="Summary", lines=6)
                gr.Markdown(
                    "*Classes: qb (red), oline (green), skill (blue), "
                    "defense (orange), ref (yellow)*"
                )

            # ── Tab 3: Field homography ──────────────────────────────────
            with gr.TabItem("Field Homography"):
                with gr.Row():
                    lines_output = gr.Image(label="Detected lines")
                    birds_output = gr.Image(label="Bird's-eye view")
                    blend_output = gr.Image(label="Blended overlay")
                hom_summary = gr.Textbox(label="Summary", lines=2)

            # ── Tab 4: Field markings ──────────────────────────────────
            with gr.TabItem("Field Markings"):
                mk_output = gr.Image(label="Classified markings")
                mk_panel = gr.Image(label="Debug panel (4-stage pipeline)")
                with gr.Row():
                    mk_white = gr.Image(label="White mask (strict HSV)")
                    mk_lc = gr.Image(label="Hash mark mask (local contrast)")
                mk_summary = gr.Textbox(label="Summary", lines=5)
                gr.Markdown(
                    "*Colors: green=yard lines, blue=sidelines, "
                    "yellow=hash marks, orange=tick marks*"
                )
                gr.Markdown("---")
                gr.Markdown(
                    "### Hash-Yards Intersections (Roboflow)\n"
                    "High-confidence detections establish the grid pattern. "
                    "Lower-confidence detections are recovered if they fall at "
                    "expected grid positions. Recovered marks shown with thinner "
                    "boxes, dimmer color, and *(grid)* label."
                )
                hash_output = gr.Image(label="Hash-yard intersection detections")
                hash_summary = gr.Textbox(label="Intersection summary", lines=6)

            # ── Tab 5: Field Numbers (Roboflow) ──────────────────────────
            with gr.TabItem("Field Numbers"):
                gr.Markdown(
                    "### Roboflow Field Detection Model\n"
                    "Detects field yard numbers (`tl-30`, `tr-40`, `t-50`, etc.), "
                    "players, refs, and ball. Yellow = field numbers, "
                    "green = players, gray = refs, orange = ball."
                )
                num_output = gr.Image(label="Detections")
                num_summary = gr.Textbox(label="Summary", lines=10)

            # ── Tab 6: Segmentation ───────────────────────────────────
            with gr.TabItem("Segmentation"):
                gr.Markdown(
                    "**Roboflow segmentation workflow** — instance segmentation "
                    "with per-player polygon masks."
                )
                with gr.Row():
                    seg_overlay_output = gr.Image(label="Mask overlay")
                    seg_masks_output = gr.Image(label="Masks only")
                seg_summary = gr.Textbox(label="Summary", lines=3)

            # ── Tab 7: Field segments ─────────────────────────────────
            with gr.TabItem("Field Segments"):
                gr.Markdown(
                    "**Roboflow field segmentation** — detects playing field "
                    "and end zone boundaries."
                )
                with gr.Row():
                    fseg_overlay_output = gr.Image(label="Field overlay")
                    fseg_masks_output = gr.Image(label="Masks only")
                fseg_summary = gr.Textbox(label="Summary", lines=3)

            # ── Tab 8: Team Assignment ──────────────────────────────
            with gr.TabItem("Team Assignment"):
                gr.Markdown(
                    "**Color-based team clustering** — unsupervised jersey color "
                    "classification using HSV hue + white ratio. "
                    "Compared against standalone HSV-only clustering."
                )
                with gr.Row():
                    ta_resnet_output = gr.Image(label="Color Clustering Teams")
                    ta_hsv_output = gr.Image(label="HSV Color Teams")
                ta_crops_output = gr.Image(label="Player Crops Comparison")
                ta_summary = gr.Textbox(label="Summary", lines=8)
                gr.Markdown(
                    "*Colors: orange = team 0, blue = team 1, gray = unknown. "
                    "Labels show team [C=color] confidence%. "
                    "Crop borders: left = color clustering, right = HSV. "
                    "Use Field Mapping tab with LOS + direction for offense/defense semantics.*"
                )

            # ── Tab 9: Blended comparison ─────────────────────────────
            with gr.TabItem("All Three Models"):
                gr.Markdown(
                    "**Side-by-side comparison** of YOLO, Presnap, and Segmentation, "
                    "plus a merged view showing which models agree."
                )
                with gr.Row():
                    blend_yolo = gr.Image(label="YOLO (person)")
                    blend_presnap = gr.Image(label="Presnap (roles)")
                with gr.Row():
                    blend_seg = gr.Image(label="Segmentation (masks)")
                    blend_merged = gr.Image(label="Blended (color = source)")
                blend_summary = gr.Textbox(label="Summary", lines=6)
                gr.Markdown(
                    "*Blended colors: green=all 3, cyan=YOLO+presnap, "
                    "yellow=YOLO+seg, white=YOLO only, blue=presnap only, red=seg only*"
                )

            # ── Tab 8: Game Setup ─────────────────────────────────────────
            with gr.TabItem("Game Setup"):
                gr.Markdown(
                    "## Line of Scrimmage, First Down & Hash Marks\n"
                    "Mark points on the field to establish the game situation and "
                    "build correspondence points for homography.\n\n"
                    "**Step 1:** Click the LOS on the sideline. "
                    "**Step 2:** Click the first down marker on the sideline. "
                    "**Step 3:** (Optional) Mark the near and far hash marks at the LOS yardage. "
                    "**Step 4:** Set yard numbers, direction, and field type. "
                    "**Step 5:** Click **Apply**."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        setup_image_view = gr.Image(
                            type="numpy",
                            label="Click to mark points",
                            interactive=True,
                            height=420,
                        )
                    with gr.Column(scale=1):
                        setup_click_mode = gr.Radio(
                            choices=["Mark LOS", "Mark 1st Down", "Mark Near Hash", "Mark Far Hash"],
                            value="Mark LOS",
                            label="Click mode",
                        )
                        los_yard_input = gr.Number(
                            value=25, label="LOS yard line",
                            minimum=1, maximum=99, precision=0,
                        )
                        fd_yard_input = gr.Number(
                            value=35, label="First down yard line",
                            minimum=1, maximum=99, precision=0,
                        )
                        direction_toggle = gr.Radio(
                            choices=["left", "right"],
                            value="right",
                            label="Direction of play",
                        )
                        field_type_toggle = gr.Radio(
                            choices=["college", "nfl"],
                            value="college",
                            label="Field type",
                        )
                        with gr.Row():
                            setup_apply_btn = gr.Button(
                                "Apply", variant="primary", size="lg",
                            )
                            auto_hash_btn = gr.Button(
                                "Use Detected Hash Marks", variant="secondary",
                            )

                with gr.Row():
                    setup_annotated = gr.Image(label="NFL-style line overlay")
                    setup_template = gr.Image(label="Field template")

                setup_data_display = gr.Textbox(
                    label="Derived data", lines=14, interactive=False,
                )

                # Hidden state
                los_point_state = gr.State(value=None)
                fd_point_state = gr.State(value=None)
                near_hash_state = gr.State(value=None)
                far_hash_state = gr.State(value=None)
                # Cache detected field markings so we can use per-line angles
                markings_state = gr.State(value=None)

                # Copy input image to setup view
                def on_setup_input_change(image_rgb):
                    return image_rgb, None, None, None, None, None

                input_image.change(
                    on_setup_input_change,
                    inputs=[input_image],
                    outputs=[setup_image_view, los_point_state, fd_point_state,
                             near_hash_state, far_hash_state, markings_state],
                )

                # Handle clicks on the setup image
                def on_setup_click(image_rgb, mode, los_pt, fd_pt, near_hash_pt, far_hash_pt,
                                   los_yd, fd_yd, direction, cached_markings, evt: gr.SelectData):
                    if image_rgb is None or evt.index is None:
                        return image_rgb, los_pt, fd_pt, near_hash_pt, far_hash_pt, cached_markings

                    ix, iy = int(evt.index[0]), int(evt.index[1])

                    # Auto-detect field markings on first click
                    if cached_markings is None:
                        try:
                            image_bgr = _rgb_to_bgr(image_rgb)
                            cached_markings = detect_field_markings(image_bgr)
                        except Exception:
                            cached_markings = None

                    if mode == "Mark LOS":
                        los_pt = (ix, iy)
                    elif mode == "Mark 1st Down":
                        fd_pt = (ix, iy)
                    elif mode == "Mark Near Hash":
                        near_hash_pt = (ix, iy)
                    elif mode == "Mark Far Hash":
                        far_hash_pt = (ix, iy)

                    # Redraw with current markers using per-line angles
                    image_bgr = _rgb_to_bgr(image_rgb)
                    annotated = draw_game_setup(
                        image_bgr, los_pt, fd_pt,
                        int(los_yd), int(fd_yd), direction,
                        markings=cached_markings,
                        near_hash_point=near_hash_pt,
                        far_hash_point=far_hash_pt,
                    )
                    return (_bgr_to_rgb(annotated), los_pt, fd_pt,
                            near_hash_pt, far_hash_pt, cached_markings)

                setup_image_view.select(
                    on_setup_click,
                    inputs=[
                        input_image, setup_click_mode,
                        los_point_state, fd_point_state,
                        near_hash_state, far_hash_state,
                        los_yard_input, fd_yard_input, direction_toggle,
                        markings_state,
                    ],
                    outputs=[setup_image_view, los_point_state, fd_point_state,
                             near_hash_state, far_hash_state, markings_state],
                )

                # Auto-fill hash marks using YOLO hash intersection model
                def on_auto_hash(image_rgb, los_pt, fd_pt, near_hash_pt, far_hash_pt,
                                 los_yd, fd_yd, direction, cached_markings):
                    if image_rgb is None or los_pt is None:
                        return image_rgb, near_hash_pt, far_hash_pt, cached_markings

                    image_bgr = _rgb_to_bgr(image_rgb)

                    # Detect field markings if not cached (for yard line angles)
                    if cached_markings is None:
                        try:
                            cached_markings = detect_field_markings(image_bgr)
                        except Exception:
                            cached_markings = None

                    found = False

                    # Primary: use YOLO hash intersection model
                    try:
                        preds = _hash_intersection_predict(image_bgr, confidence=30)
                        if len(preds) >= 2:
                            # Get the LOS yard line angle for perpendicular projection
                            yl_angle = 90.0
                            if cached_markings and cached_markings.dominant_angle:
                                yl_angle = cached_markings.dominant_angle
                            sideline_angle = (yl_angle + 90) % 180
                            side_rad = np.deg2rad(sideline_angle)
                            side_dx, side_dy = np.cos(side_rad), np.sin(side_rad)

                            # Project each detection onto the yard line direction
                            # (perpendicular to sideline) to find the two closest
                            # to the LOS yardage
                            yl_rad = np.deg2rad(yl_angle)
                            yl_dx, yl_dy = np.cos(yl_rad), np.sin(yl_rad)

                            # Score each detection by distance along sideline from LOS
                            scored = []
                            for p in preds:
                                cx, cy = p["x"], p["y"]
                                # Project onto sideline direction from LOS
                                along_sideline = (cx - los_pt[0]) * side_dx + (cy - los_pt[1]) * side_dy
                                # Lateral distance (along yard line direction)
                                lateral = (cx - los_pt[0]) * yl_dx + (cy - los_pt[1]) * yl_dy
                                scored.append((abs(along_sideline), lateral, cx, cy))

                            # Sort by distance along sideline — closest to LOS first
                            scored.sort(key=lambda s: s[0])

                            # Take detections near the LOS (within ~3 yard-lines worth)
                            # Group by lateral position to find the two hash rows
                            nearby = scored[:min(20, len(scored))]

                            # Cluster into 2 lateral groups using simple split
                            nearby.sort(key=lambda s: s[1])  # sort by lateral position
                            laterals = [s[1] for s in nearby]
                            # Find the biggest gap
                            best_gap_idx = 0
                            best_gap = 0
                            for i in range(len(laterals) - 1):
                                gap = laterals[i + 1] - laterals[i]
                                if gap > best_gap:
                                    best_gap = gap
                                    best_gap_idx = i

                            if best_gap > 0 and len(nearby) >= 2:
                                group_near = nearby[:best_gap_idx + 1]
                                group_far = nearby[best_gap_idx + 1:]

                                # Pick the detection closest to LOS in each group
                                group_near.sort(key=lambda s: s[0])
                                group_far.sort(key=lambda s: s[0])

                                if group_near and group_far:
                                    near_hash_pt = (int(group_near[0][2]), int(group_near[0][3]))
                                    far_hash_pt = (int(group_far[0][2]), int(group_far[0][3]))
                                    found = True
                    except Exception:
                        pass

                    # Fallback: use CC-based hash mark rows
                    if not found and cached_markings and len(cached_markings.hash_rows) >= 2:
                        rows = cached_markings.hash_rows
                        row_marks = []
                        for row in rows:
                            best_mark = None
                            best_dist = float("inf")
                            for seg in row.marks:
                                mx, my = seg.midpoint
                                dist = abs(mx - los_pt[0]) + abs(my - los_pt[1])
                                if dist < best_dist:
                                    best_dist = dist
                                    best_mark = seg
                            if best_mark is not None:
                                row_marks.append(best_mark)

                        if len(row_marks) >= 2:
                            row_marks.sort(key=lambda s: np.hypot(
                                s.midpoint[0] - los_pt[0], s.midpoint[1] - los_pt[1]))
                            near_hash_pt = (int(row_marks[0].midpoint[0]), int(row_marks[0].midpoint[1]))
                            far_hash_pt = (int(row_marks[1].midpoint[0]), int(row_marks[1].midpoint[1]))

                    annotated = draw_game_setup(
                        image_bgr, los_pt, fd_pt,
                        int(los_yd), int(fd_yd), direction,
                        markings=cached_markings,
                        near_hash_point=near_hash_pt,
                        far_hash_point=far_hash_pt,
                    )
                    return _bgr_to_rgb(annotated), near_hash_pt, far_hash_pt, cached_markings

                auto_hash_btn.click(
                    on_auto_hash,
                    inputs=[
                        input_image, los_point_state, fd_point_state,
                        near_hash_state, far_hash_state,
                        los_yard_input, fd_yard_input, direction_toggle,
                        markings_state,
                    ],
                    outputs=[setup_image_view, near_hash_state, far_hash_state, markings_state],
                )

                # Handle Apply button
                def on_setup_apply(image_rgb, los_pt, fd_pt, near_hash_pt, far_hash_pt,
                                   los_yd, fd_yd, direction, field_type, cached_markings):
                    if image_rgb is None:
                        ph = _placeholder(text="Drop an image first")
                        return ph, ph, "No image provided"

                    los_yd = int(los_yd)
                    fd_yd = int(fd_yd)
                    image_bgr = _rgb_to_bgr(image_rgb)

                    # Draw annotated image with per-line perspective angles
                    annotated = draw_game_setup(
                        image_bgr, los_pt, fd_pt,
                        los_yd, fd_yd, direction,
                        markings=cached_markings,
                        near_hash_point=near_hash_pt,
                        far_hash_point=far_hash_pt,
                    )

                    # Draw template with hash mark positions
                    template = draw_game_setup_template(
                        los_yd, fd_yd, direction,
                        field_type=field_type,
                        near_hash_point=near_hash_pt,
                        far_hash_point=far_hash_pt,
                    )

                    # Compute derived data
                    parts = [
                        f"LOS: {los_yd}-yard line",
                        f"First Down: {fd_yd}-yard line",
                        f"Distance to 1st: {abs(fd_yd - los_yd)} yards",
                        f"Direction of play: {direction}",
                        f"Field type: {field_type}",
                    ]

                    if los_pt and fd_pt:
                        data = compute_game_setup_data(
                            los_pt, fd_pt, los_yd, fd_yd, direction, image_bgr,
                            near_hash_point=near_hash_pt,
                            far_hash_point=far_hash_pt,
                            field_type=field_type,
                            markings=cached_markings,
                        )
                        parts.extend([
                            "",
                            "--- Derived from sideline points ---",
                            f"LOS sideline point: ({los_pt[0]}, {los_pt[1]})",
                            f"1st down sideline point: ({fd_pt[0]}, {fd_pt[1]})",
                            f"Sideline angle: {data['sideline_angle']:.1f}\u00b0",
                            f"Yard line angle: {data['yard_line_angle']:.1f}\u00b0",
                            f"Pixels per yard: {data['pixels_per_yard']:.1f}",
                            f"Pixel distance between marks: {np.hypot(fd_pt[0]-los_pt[0], fd_pt[1]-los_pt[1]):.0f}px",
                        ])

                        # Hash mark info
                        near_yd, far_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
                        if near_hash_pt:
                            parts.append(f"Near hash: image ({near_hash_pt[0]}, {near_hash_pt[1]}) @ {near_yd:.1f}yd from sideline")
                        if far_hash_pt:
                            parts.append(f"Far hash: image ({far_hash_pt[0]}, {far_hash_pt[1]}) @ {far_yd:.1f}yd from sideline")

                        # Correspondence points
                        parts.append("")
                        parts.append(f"--- Homography correspondence points ({len(data['homography_points'])}) ---")
                        for i, (img_pt, tmpl_pt) in enumerate(data["homography_points"]):
                            labels = ["LOS sideline", "1st down sideline", "Near hash", "Far hash"]
                            label = labels[i] if i < len(labels) else f"Point {i}"
                            parts.append(f"{label}: image ({img_pt[0]}, {img_pt[1]}) \u2192 template ({tmpl_pt[0]}, {tmpl_pt[1]})")
                    elif los_pt:
                        parts.append(f"\nLOS sideline point: ({los_pt[0]}, {los_pt[1]})")
                        parts.append("Mark the first down point for full data.")
                    elif fd_pt:
                        parts.append(f"\n1st down sideline point: ({fd_pt[0]}, {fd_pt[1]})")
                        parts.append("Mark the LOS point for full data.")
                    else:
                        parts.append("\nClick the image to mark sideline points.")

                    return (
                        _bgr_to_rgb(annotated),
                        _bgr_to_rgb(template),
                        "\n".join(parts),
                    )

                setup_apply_btn.click(
                    on_setup_apply,
                    inputs=[
                        input_image, los_point_state, fd_point_state,
                        near_hash_state, far_hash_state,
                        los_yard_input, fd_yard_input, direction_toggle,
                        field_type_toggle, markings_state,
                    ],
                    outputs=[setup_annotated, setup_template, setup_data_display],
                )

            # ── Tab 9: Interactive field mapping ─────────────────────────
            with gr.TabItem("Field Mapping"):
                gr.Markdown(
                    "**Step 1:** Set the offense direction. "
                    "**Step 2:** Click the ball in the **image** (left). "
                    "**Step 3:** Click the matching position on the **field template** (right). "
                    "**Step 4:** Click **Map Players**."
                )
                fm_direction_toggle = gr.Radio(
                    choices=["right", "left"],
                    value="right",
                    label="Offense direction (which way is the offense going?)",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        # Clickable image for ball-in-image position
                        ball_image_view = gr.Image(
                            type="numpy",
                            label="Step 1: Click ball in image",
                            interactive=True,
                            height=360,
                        )
                        ball_image_display = gr.Textbox(
                            value="Click the ball in the image",
                            label="Ball in image",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        # Field template for clicking (landscape, redraws on field_type/direction change)
                        field_template_img = gr.Image(
                            value=_bgr_to_rgb(draw_field_template(
                                orientation="horizontal", field_type="college", direction="right",
                            )),
                            type="numpy",
                            label="Step 3: Click ball on field",
                            interactive=True,
                            height=360,
                        )
                        ball_yard_display = gr.Textbox(
                            value="Click the field to set ball position",
                            label="Ball on field",
                            interactive=False,
                        )
                map_btn = gr.Button("Map Players", variant="primary", size="lg")
                with gr.Row():
                    field_result = gr.Image(label="Players on field")
                    playbook_result = gr.Image(label="Playbook View")
                with gr.Row():
                    overlay_result = gr.Image(label="Field overlay")
                    warped_result = gr.Image(label="Warped view")
                with gr.Row():
                    corr_result = gr.Image(label="Yard line assignments")
                map_summary = gr.Textbox(label="Summary", lines=10)

                # ── Coordinate export ──────────────────────────────
                with gr.Accordion("Export Coordinates", open=False):
                    export_btn = gr.Button("Export as JSON", size="sm")
                    coord_json = gr.Code(label="Player coordinates (JSON)", language="json", lines=15)
                    export_file = gr.File(label="Download JSON")

                # Hidden state
                # ball_image_pos: (img_x, img_y) or None
                ball_image_state = gr.State(value=None)
                # ball_field_state: (yard, template_x, template_y)
                ball_field_state = gr.State(value=(25, TEMPLATE_W // 2, 350))
                # Team labels from Team Assignment tab: list of (bbox_xyxy, label)
                ta_labels_state = gr.State(value=[])
                # Playbook coordinate data for export
                playbook_data_state = gr.State(value=None)

                # Handle click on the image to mark ball position
                def on_image_click(image_rgb, evt: gr.SelectData):
                    if image_rgb is None or evt.index is None:
                        return image_rgb, "Click the ball in the image", None
                    ix, iy = evt.index
                    ix, iy = int(ix), int(iy)

                    # Draw marker on image
                    marked = image_rgb.copy()
                    cv2.circle(marked, (ix, iy), 12, (255, 165, 0), 3)
                    cv2.circle(marked, (ix, iy), 4, (255, 165, 0), -1)
                    cv2.putText(marked, "BALL", (ix + 15, iy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                    return (
                        marked,
                        f"Ball in image at ({ix}, {iy})",
                        (ix, iy),
                    )

                ball_image_view.select(
                    on_image_click,
                    inputs=[ball_image_view],
                    outputs=[ball_image_view, ball_image_display, ball_image_state],
                )

                # Copy input image to ball_image_view when input changes
                def on_input_change(image_rgb):
                    return image_rgb, "Click the ball in the image", None

                input_image.change(
                    on_input_change,
                    inputs=[input_image],
                    outputs=[ball_image_view, ball_image_display, ball_image_state],
                )

                # Handle click on field template
                def on_field_click(image, ft, direction, evt: gr.SelectData):
                    if evt.index is None:
                        return image, "Click the field to set ball position", (25, TEMPLATE_W // 2, 350)
                    click_x, click_y = evt.index

                    # Landscape: X axis = yards, Y axis = lateral (inverted)
                    # Reverse-map landscape click → portrait template coords
                    if direction == "left":
                        portrait_ty = TEMPLATE_H - 1 - int(click_x)
                    else:
                        portrait_ty = int(click_x)
                    portrait_tx = TEMPLATE_W - 1 - int(click_y)

                    yard = template_y_to_yard(portrait_ty)
                    yard = max(1, min(99, yard))
                    tx = max(0, min(TEMPLATE_W - 1, portrait_tx))
                    ty = yard_to_template_y(yard)

                    lateral_pct = tx / TEMPLATE_W
                    if lateral_pct < 0.35:
                        loc = "near sideline"
                    elif lateral_pct > 0.65:
                        loc = "far sideline"
                    elif 0.45 < lateral_pct < 0.55:
                        loc = "center"
                    elif lateral_pct < 0.45:
                        loc = "near hash"
                    else:
                        loc = "far hash"

                    # Redraw landscape template with ball marker
                    tmpl = draw_field_template(
                        orientation="horizontal", field_type=ft, direction=direction,
                    )
                    # Convert portrait (tx, ty) → landscape pixel for drawing
                    lx = ty  # template_y → landscape x
                    ly = TEMPLATE_W - 1 - tx  # template_x → landscape y
                    if direction == "left":
                        lx = TEMPLATE_H - 1 - lx
                    cv2.circle(tmpl, (lx, ly), 10, (0, 165, 255), -1)
                    cv2.putText(tmpl, f"{yard} yd ({loc})", (lx + 15, ly + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    return (
                        _bgr_to_rgb(tmpl),
                        f"Ball at {yard}-yard line, {loc}",
                        (yard, tx, ty),
                    )

                field_template_img.select(
                    on_field_click,
                    inputs=[field_template_img, field_type_radio, fm_direction_toggle],
                    outputs=[field_template_img, ball_yard_display, ball_field_state],
                )

                # Redraw template when field type or direction changes
                def on_field_setting_change(ft, direction):
                    return (
                        _bgr_to_rgb(draw_field_template(
                            orientation="horizontal", field_type=ft, direction=direction,
                        )),
                        "Click the field to set ball position",
                        (25, TEMPLATE_W // 2, 350),
                    )

                field_type_radio.change(
                    on_field_setting_change,
                    inputs=[field_type_radio, fm_direction_toggle],
                    outputs=[field_template_img, ball_yard_display, ball_field_state],
                )
                fm_direction_toggle.change(
                    on_field_setting_change,
                    inputs=[field_type_radio, fm_direction_toggle],
                    outputs=[field_template_img, ball_yard_display, ball_field_state],
                )

                # Handle "Map Players" click
                def _build_playbook_data(field_players, team_labels, ball_yard, ft, direction):
                    """Build structured coordinate data for export."""
                    _TEAM_MAP = {0: "offense", 1: "defense", -1: "unknown"}
                    players = []
                    for i, (p, lbl) in enumerate(zip(field_players, team_labels)):
                        yd = p.get("yard")
                        lat = p.get("lateral")
                        if yd is None:
                            continue
                        entry = {
                            "id": i + 1,
                            "team": _TEAM_MAP.get(lbl, "unknown"),
                            "yard_line": round(float(yd), 1),
                            "lateral_yards": round(float(lat), 1) if lat is not None else None,
                        }
                        players.append(entry)
                    return {
                        "ball_yard_line": int(ball_yard),
                        "offense_direction": direction,
                        "field_type": ft,
                        "player_count": len(players),
                        "players": players,
                    }

                def on_map_players(image_rgb, conf, ball_field_info, ball_img_pos, ft, direction, ta_labels):
                    if image_rgb is None:
                        ph = _placeholder(text="Drop an image first")
                        return ph, ph, ph, ph, ph, "No image provided", None
                    ball_yard, ball_tx, ball_ty = ball_field_info
                    image_bgr = _rgb_to_bgr(image_rgb)
                    # Pass pre-computed team labels from the Team Assignment tab
                    # so that team classification is consistent between tabs.
                    precomputed = ta_labels if ta_labels else None
                    try:
                        field_img, overlay, corr_img, warped, summary, field_players, team_labels = (
                            run_interactive_homography(
                                image_bgr,
                                int(ball_yard),
                                ball_template_pos=(int(ball_tx), int(ball_ty)),
                                ball_image_pos=tuple(int(v) for v in ball_img_pos) if ball_img_pos else None,
                                conf=conf,
                                field_type=ft,
                                offense_direction=direction,
                                precomputed_team_labels=precomputed,
                            )
                        )
                        # Render playbook-style diagram (landscape, direction-aware)
                        playbook_pil = render_playbook(
                            field_players, team_labels,
                            ball_yard=int(ball_yard),
                            field_type=ft,
                            orientation="horizontal",
                            direction=direction,
                        )
                        playbook_rgb = np.array(playbook_pil)
                        # Build coordinate data for export
                        pb_data = _build_playbook_data(
                            field_players, team_labels,
                            int(ball_yard), ft, direction,
                        )
                        return (
                            _bgr_to_rgb(field_img),
                            playbook_rgb,
                            _bgr_to_rgb(overlay),
                            _bgr_to_rgb(corr_img),
                            _bgr_to_rgb(warped),
                            summary,
                            pb_data,
                        )
                    except Exception as e:
                        ph = _placeholder(text=str(e))
                        return ph, ph, ph, ph, ph, str(e), None

                def on_export_coords(pb_data):
                    """Export playbook coordinates as JSON string + downloadable file."""
                    import json, tempfile, os
                    if not pb_data:
                        return "No data — click Map Players first.", None
                    json_str = json.dumps(pb_data, indent=2)
                    # Write to temp file for download
                    tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", prefix="playbook_",
                        delete=False, dir=tempfile.gettempdir(),
                    )
                    tmp.write(json_str)
                    tmp.close()
                    return json_str, tmp.name

                map_btn.click(
                    on_map_players,
                    inputs=[input_image, conf_slider, ball_field_state, ball_image_state, field_type_radio, fm_direction_toggle, ta_labels_state],
                    outputs=[field_result, playbook_result, overlay_result, corr_result, warped_result, map_summary, playbook_data_state],
                )
                export_btn.click(
                    on_export_coords,
                    inputs=[playbook_data_state],
                    outputs=[coord_json, export_file],
                )

            # ── Tab 10: Video Frame Browser ──────────────────────────────
            with gr.TabItem("Video Frames"):
                gr.Markdown(
                    "## Video Frame Browser\n"
                    "Load a video, scrub to specific frames, and extract presnap frames "
                    "for analysis. Extracted frames are saved to `videos/frames/` and "
                    "can be loaded into the pipeline via the main image input."
                )

                def _scan_videos():
                    import glob as _glob
                    files = []
                    for p in ["videos/*.mp4", "videos/*.avi", "videos/*.mov", "*.mp4"]:
                        files.extend(_glob.glob(p))
                    return sorted(set(files))

                with gr.Row():
                    video_path_input = gr.Dropdown(
                        label="Video file",
                        choices=_scan_videos(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_videos_btn = gr.Button("Refresh", size="sm")

                with gr.Row():
                    with gr.Column(scale=3):
                        frame_display = gr.Image(
                            label="Current frame",
                            type="numpy",
                            interactive=False,
                            height=400,
                        )
                    with gr.Column(scale=1):
                        frame_slider = gr.Slider(
                            minimum=0, maximum=1, step=1, value=0,
                            label="Frame number",
                        )
                        time_display = gr.Textbox(
                            label="Timestamp", interactive=False, value="0:00.000",
                        )
                        video_info = gr.Textbox(
                            label="Video info", interactive=False, lines=4,
                        )
                        with gr.Row():
                            prev_btn = gr.Button("◀ -1", size="sm")
                            prev10_btn = gr.Button("◀ -10", size="sm")
                            next10_btn = gr.Button("+10 ▶", size="sm")
                            next_btn = gr.Button("+1 ▶", size="sm")
                        with gr.Row():
                            prev30_btn = gr.Button("◀ -1s", size="sm")
                            next30_btn = gr.Button("+1s ▶", size="sm")
                            prev300_btn = gr.Button("◀ -10s", size="sm")
                            next300_btn = gr.Button("+10s ▶", size="sm")
                        extract_btn = gr.Button(
                            "Extract Frame", variant="primary", size="lg",
                        )
                        use_frame_btn = gr.Button(
                            "Use as Pipeline Input", variant="secondary",
                        )
                        extract_status = gr.Textbox(
                            label="Status", interactive=False,
                        )

                # Hidden state for video metadata
                video_meta_state = gr.State(value=None)  # {path, fps, total_frames, w, h}

                def _refresh_video_list():
                    videos = _scan_videos()
                    return gr.update(choices=videos, value=videos[0] if videos else None)

                def _load_video(path):
                    """Load video and return first frame + metadata."""
                    if not path or not os.path.exists(path):
                        return None, gr.update(maximum=1), f"Video not found: {path!r}", "", None

                    cap = cv2.VideoCapture(path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total / fps if fps > 0 else 0

                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        return None, gr.update(maximum=1), "Failed to read video", "", None

                    meta = {"path": path, "fps": fps, "total_frames": total, "w": w, "h": h}
                    info = (
                        f"Resolution: {w}x{h}\n"
                        f"FPS: {fps:.1f}\n"
                        f"Frames: {total}\n"
                        f"Duration: {int(duration//60)}m {int(duration%60)}s"
                    )

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return (
                        frame_rgb,
                        gr.update(maximum=total - 1, value=0),
                        info,
                        "0:00.000",
                        meta,
                    )

                def _seek_frame(frame_num, meta):
                    """Read a specific frame from the video."""
                    if meta is None:
                        return None, "No video loaded"

                    frame_num = max(0, min(int(frame_num), meta["total_frames"] - 1))
                    cap = cv2.VideoCapture(meta["path"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        return None, f"Failed to read frame {frame_num}"

                    secs = frame_num / meta["fps"] if meta["fps"] > 0 else 0
                    mins = int(secs // 60)
                    secs_rem = secs % 60
                    timestamp = f"{mins}:{secs_rem:06.3f}"

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb, timestamp

                def _make_stepper(delta):
                    def _step(current, meta):
                        if meta is None:
                            return current, None, "No video loaded"
                        new_frame = max(0, min(int(current + delta), meta["total_frames"] - 1))
                        img, ts = _seek_frame(new_frame, meta)
                        return new_frame, img, ts
                    return _step

                def _extract_frame(frame_num, meta):
                    """Save current frame as an image file."""
                    if meta is None:
                        return "No video loaded"

                    frame_num = int(frame_num)
                    cap = cv2.VideoCapture(meta["path"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        return f"Failed to read frame {frame_num}"

                    out_dir = Path("videos/frames")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    video_name = Path(meta["path"]).stem
                    out_path = out_dir / f"{video_name}_frame{frame_num:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    return f"Saved: {out_path}"

                def _use_as_input(frame_num, meta):
                    """Return the current frame as RGB for the main pipeline input."""
                    if meta is None:
                        return None, "No video loaded"

                    frame_num = int(frame_num)
                    cap = cv2.VideoCapture(meta["path"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        return None, f"Failed to read frame {frame_num}"

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb, f"Loaded frame {frame_num} into pipeline"

                # Wire up events
                refresh_videos_btn.click(
                    _refresh_video_list,
                    outputs=[video_path_input],
                )

                video_path_input.change(
                    _load_video,
                    inputs=[video_path_input],
                    outputs=[frame_display, frame_slider, video_info, time_display, video_meta_state],
                )
                # Also bind input event — some Gradio versions fire input instead of change
                if hasattr(video_path_input, "input"):
                    video_path_input.input(
                        _load_video,
                        inputs=[video_path_input],
                        outputs=[frame_display, frame_slider, video_info, time_display, video_meta_state],
                    )

                frame_slider.release(
                    _seek_frame,
                    inputs=[frame_slider, video_meta_state],
                    outputs=[frame_display, time_display],
                )

                # Step buttons
                for btn, delta in [
                    (prev_btn, -1), (next_btn, 1),
                    (prev10_btn, -10), (next10_btn, 10),
                    (prev30_btn, -30), (next30_btn, 30),
                    (prev300_btn, -300), (next300_btn, 300),
                ]:
                    btn.click(
                        _make_stepper(delta),
                        inputs=[frame_slider, video_meta_state],
                        outputs=[frame_slider, frame_display, time_display],
                    )

                extract_btn.click(
                    _extract_frame,
                    inputs=[frame_slider, video_meta_state],
                    outputs=[extract_status],
                )

                use_frame_btn.click(
                    _use_as_input,
                    inputs=[frame_slider, video_meta_state],
                    outputs=[input_image, extract_status],
                )

        # Sync Game Setup direction → Field Mapping direction
        direction_toggle.change(
            lambda d: d,
            inputs=[direction_toggle],
            outputs=[fm_direction_toggle],
        )

        run_btn.click(
            run_pipeline,
            inputs=[input_image, conf_slider],
            outputs=[
                det_output, det_summary,
                rf_output, rf_summary,
                lines_output, birds_output, blend_output, hom_summary,
                mk_output, mk_panel, mk_white, mk_lc, mk_summary,
                hash_output, hash_summary,
                num_output, num_summary,
                seg_overlay_output, seg_masks_output, seg_summary,
                fseg_overlay_output, fseg_masks_output, fseg_summary,
                ta_resnet_output, ta_hsv_output, ta_crops_output, ta_summary,
                blend_yolo, blend_presnap, blend_seg, blend_merged, blend_summary,
                ta_labels_state,
            ],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app()
    app.launch(server_port=args.port, share=args.share)
