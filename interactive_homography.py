"""
Interactive homography: user provides the ball yard line,
we anchor detected yard lines to real field coordinates,
compute a homography, and project player positions onto a
bird's-eye field template.
"""

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from field_homography import (
    TEMPLATE_SCALE,
    TEMPLATE_W,
    TEMPLATE_H,
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
)
from field_markings import (
    detect_field_markings,
    FieldMarkings,
    _extend_line,
    _perp_position,
)


# ── Field type hash positions ────────────────────────────────────────────────

# Distance in yards from the near sideline (x=0 in template) to each hash mark
HASH_POSITIONS = {
    "college": (20.0, 33.33),   # 60ft (20 yd) from each sideline per NCAA rules
    "nfl":     (23.58, 29.75),  # 70'9" from each sideline
}


def _hash_template_x(field_type: str = "college") -> tuple[int, int]:
    """Return (near_hash_x, far_hash_x) in template pixels for the given field type."""
    near_yd, far_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    return int(near_yd * TEMPLATE_SCALE), int(far_yd * TEMPLATE_SCALE)


# ── Hash-yards intersection detection ────────────────────────────────────────

_HASH_INTERSECTION_API_URL = "https://detect.roboflow.com/hash-yards-intersection/4"
_HASH_LOCAL_MODEL_PATH = Path("models/hash_intersection.pt")

# Lazy-loaded local YOLO model for hash detection
_hash_yolo_model = None


def _get_hash_model():
    """Load the local hash intersection YOLO model (lazy, singleton)."""
    global _hash_yolo_model
    if _hash_yolo_model is None:
        if _HASH_LOCAL_MODEL_PATH.exists():
            from ultralytics import YOLO
            _hash_yolo_model = YOLO(str(_HASH_LOCAL_MODEL_PATH))
        else:
            return None
    return _hash_yolo_model


def _hash_intersection_predict_local(
    image_bgr: np.ndarray, confidence: int = 40, overlap: int = 40,
) -> list[dict]:
    """Run local YOLO11 hash intersection model.

    Returns predictions in the same format as the Roboflow API:
    [{"x": center_x, "y": center_y, "width": w, "height": h,
      "confidence": conf, "class": "hash"}, ...]
    """
    model = _get_hash_model()
    if model is None:
        raise FileNotFoundError(
            f"Local hash model not found at {_HASH_LOCAL_MODEL_PATH}. "
            f"Train with: python train_hash.py"
        )

    conf_float = confidence / 100.0
    iou_float = 1.0 - (overlap / 100.0)   # Roboflow overlap → YOLO IoU threshold

    results = model.predict(
        image_bgr,
        conf=conf_float,
        iou=iou_float,
        imgsz=640,
        verbose=False,
    )

    predictions = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cx, cy, w, h = boxes.xywh[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = result.names[cls_id]

            predictions.append({
                "x": cx,
                "y": cy,
                "width": w,
                "height": h,
                "confidence": conf,
                "class": cls_name,
            })

    return predictions


def _hash_intersection_predict_api(
    image_bgr: np.ndarray, confidence: int = 40, overlap: int = 40,
) -> list[dict]:
    """Call Roboflow hash-yards-intersection model (v4) — remote API fallback."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set in .env")

    _, buf = cv2.imencode(".jpg", image_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")

    url = (
        f"{_HASH_INTERSECTION_API_URL}"
        f"?api_key={api_key}&confidence={confidence}&overlap={overlap}"
    )
    req = urllib.request.Request(
        url, data=b64.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["predictions"]


# Environment variable to force API fallback: HASH_USE_API=1
_HASH_USE_API = os.getenv("HASH_USE_API", "0").strip().lower() in ("1", "true", "yes")


def _hash_intersection_predict(
    image_bgr: np.ndarray, confidence: int = 40, overlap: int = 40,
) -> list[dict]:
    """Detect hash-yard intersections using local model or Roboflow API.

    Uses local YOLO11 model if available at models/hash_intersection.pt.
    Falls back to Roboflow API if:
      - Local model file doesn't exist, OR
      - HASH_USE_API=1 environment variable is set
    """
    if not _HASH_USE_API and _HASH_LOCAL_MODEL_PATH.exists():
        return _hash_intersection_predict_local(image_bgr, confidence, overlap)
    return _hash_intersection_predict_api(image_bgr, confidence, overlap)


def _recover_grid_candidates(
    anchors: list[dict],
    candidates: list[dict],
    snap_radius: float = 40.0,
    max_steps: int = 2,
) -> list[dict]:
    """Recover low-confidence candidates that align with the grid pattern
    established by high-confidence anchor detections.

    1. Compute nearest-neighbor displacement vectors between anchors
    2. Cluster displacements into two dominant grid directions by angle
    3. Predict expected grid positions up to *max_steps* away from each anchor
    4. Accept candidates within *snap_radius* of an unfilled expected position
    """
    if len(anchors) < 3 or not candidates:
        return []

    anchor_pts = np.array([(a["x"], a["y"]) for a in anchors], dtype=float)

    # ── Step 1: collect nearest-neighbor displacement vectors ──
    nn_disps = []
    for i in range(len(anchor_pts)):
        diffs = anchor_pts - anchor_pts[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:4]
        for j in nearest:
            d = diffs[j]
            length = dists[j]
            if 15 < length < 500:
                # Canonicalize to positive-x half-plane for consistent clustering
                if d[0] < -1 or (abs(d[0]) < 1 and d[1] < 0):
                    d = -d
                nn_disps.append(d)

    if len(nn_disps) < 4:
        return []

    nn_disps = np.array(nn_disps)
    angles = np.arctan2(nn_disps[:, 1], nn_disps[:, 0])

    # ── Step 2: split into two direction clusters via largest angular gap ──
    sorted_order = np.argsort(angles)
    sorted_angles = angles[sorted_order]
    gaps = np.diff(sorted_angles)
    if len(gaps) == 0:
        return []

    split_pos = np.argmax(gaps)
    threshold = (sorted_angles[split_pos] + sorted_angles[split_pos + 1]) / 2

    grid_vectors = []
    for mask in [angles <= threshold, angles > threshold]:
        group = nn_disps[mask]
        if len(group) >= 2:
            vec = np.median(group, axis=0)
            if np.linalg.norm(vec) > 10:
                grid_vectors.append(vec)

    if not grid_vectors:
        return []

    # Adaptive snap radius: cap at 30% of smallest grid spacing
    min_spacing = min(np.linalg.norm(v) for v in grid_vectors)
    effective_radius = min(snap_radius, 0.3 * min_spacing)

    # ── Step 3: predict expected grid positions from each anchor ──
    expected_pts = []
    for pt in anchor_pts:
        for v in grid_vectors:
            for step in range(-max_steps, max_steps + 1):
                if step == 0:
                    continue
                expected_pts.append(pt + step * v)

    if not expected_pts:
        return []
    expected_pts = np.array(expected_pts)

    # Keep only positions that are NOT already near an anchor (i.e. gaps)
    gap_positions = []
    for ep in expected_pts:
        dists_to_anchors = np.linalg.norm(anchor_pts - ep, axis=1)
        if dists_to_anchors.min() > effective_radius:
            gap_positions.append(ep)

    if not gap_positions:
        return []
    gap_positions = np.array(gap_positions)

    # ── Step 4: accept candidates that fill a gap ──
    recovered = []
    seen = set()
    for ci, cand in enumerate(candidates):
        if ci in seen:
            continue
        cp = np.array([cand["x"], cand["y"]])
        dists = np.linalg.norm(gap_positions - cp, axis=1)
        if dists.min() <= effective_radius:
            cand_copy = dict(cand)
            cand_copy["_recovered"] = True
            recovered.append(cand_copy)
            seen.add(ci)

    return recovered


# ── Grid vector estimation ────────────────────────────────────────────────────

def _estimate_grid_vectors(
    points: list[tuple[float, float]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate grid direction vectors from hash mark positions.

    Uses only the **nearest-neighbor** displacement for each point (the
    single shortest vector from each point).  These NN vectors form two
    tight angular clusters corresponding to the two grid directions.  By
    only using NN vectors, we avoid contamination from long diagonal
    displacements.

    Returns (vec_along, vec_across):
      - vec_along: along yard lines (typically the vector with the
        *larger* absolute X-component in a sideline-camera view, since
        yard lines run roughly left↔right across the image)
      - vec_across: across yard lines (5-yard spacing, more vertical)

    Returns (None, None) if fewer than 4 points.
    """
    if len(points) < 4:
        if len(points) >= 2:
            d = np.array(points[1]) - np.array(points[0])
            return d, None
        return None, None

    pts = np.array(points, dtype=float)
    n = len(pts)

    # ── Step 1: Collect 2-NN displacement vectors ────────────────────
    nn_disps = []
    for i in range(n):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = np.inf
        nearest2 = np.argsort(dists)[:2]
        for j in nearest2:
            d = pts[j] - pts[i]
            # Canonicalize to positive-x half-plane
            if d[0] < -1 or (abs(d[0]) < 1 and d[1] < 0):
                d = -d
            nn_disps.append(d)

    nn_disps = np.array(nn_disps)
    angles = np.arctan2(nn_disps[:, 1], nn_disps[:, 0])

    # ── Step 2: Split into two clusters by largest angular gap ──────
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    if len(gaps) == 0:
        return None, None

    split_pos = np.argmax(gaps)
    threshold = (sorted_angles[split_pos] + sorted_angles[split_pos + 1]) / 2

    vecs = []
    for mask in [angles <= threshold, angles > threshold]:
        group = nn_disps[mask]
        if len(group) >= 1:
            vec = np.median(group, axis=0)
            if np.linalg.norm(vec) > 10:
                vecs.append(vec)

    if len(vecs) < 2:
        return (vecs[0] if vecs else None), None

    # ── Step 3: Distinguish along vs across ─────────────────────────
    # In sideline camera views:
    #   - vec_across (5-yard spacing) runs mostly horizontal (low |Y|/|X|)
    #   - vec_along (hash pair, into field depth) is more diagonal (high |Y|/|X|)
    def _yx_ratio(v):
        return abs(v[1]) / max(abs(v[0]), 1e-6)

    if _yx_ratio(vecs[0]) > _yx_ratio(vecs[1]):
        vec_along, vec_across = vecs[0], vecs[1]
    else:
        vec_along, vec_across = vecs[1], vecs[0]

    return vec_along, vec_across


# ── FieldGrid dataclass ──────────────────────────────────────────────────────

@dataclass
class FieldGrid:
    """Unified field grid detection result."""
    detections: list[dict] = field(default_factory=list)
    grid_vec_along: np.ndarray | None = None
    grid_vec_across: np.ndarray | None = None
    yard_line_groups: list[dict] = field(default_factory=list)
    projected_intersections: list[dict] = field(default_factory=list)


# ── Core grid detection ──────────────────────────────────────────────────────

def detect_field_grid(
    image: np.ndarray,
    ball_yard: int,
    ball_image_pos: tuple[int, int] | None = None,
    high_conf: int = 25,
    low_conf: int = 10,
    snap_radius: float = 50.0,
) -> FieldGrid:
    """Detect field grid using the hash-yards-intersection model.

    Key insight: each detected "hash" mark is one end of a hash mark on a
    yard line.  Hash marks come in pairs (near hash + far hash) on the same
    yard line.  The vector connecting a pair runs *along* the yard line
    (``vec_along``).  The vector between adjacent yard lines runs *across*
    the field (``vec_across``).

    Constraints enforced:
      - Each yard line has at most 2 hash marks (near + far).
      - Yard lines are spaced exactly 5 yards apart.
      - The ball position anchors which group gets which yard number.

    Args:
        ball_yard: yard line number (0-100) where the ball is placed.
        ball_image_pos: (x, y) pixel position of the ball in the image.
            If given, the closest yard-line group is anchored to the ball's
            yard number.  Without it, the center group is used.
    """
    all_preds = _hash_intersection_predict(image, confidence=low_conf)
    high_threshold = high_conf / 100.0
    anchors = [p for p in all_preds if p["confidence"] >= high_threshold]
    candidates = [p for p in all_preds if p["confidence"] < high_threshold]

    recovered = _recover_grid_candidates(anchors, candidates, snap_radius)
    for p in anchors:
        p["_recovered"] = False
    final_dets = anchors + recovered

    # Use ALL hash-class detections (the model outputs "hash", not
    # "hash-yard-intersection")
    hash_marks = [p for p in final_dets if p["class"] in ("hash", "hash-yard-intersection")]
    anchor_hashes = [p for p in anchors if p["class"] in ("hash", "hash-yard-intersection")]

    if len(hash_marks) < 2:
        return FieldGrid(detections=final_dets)

    # Estimate grid vectors from HIGH-CONFIDENCE anchors only so that
    # low-confidence recovered points cannot corrupt the geometry.
    anchor_centers = [(p["x"], p["y"]) for p in anchor_hashes]
    vec_along, vec_across = _estimate_grid_vectors(anchor_centers)

    if vec_across is None:
        return FieldGrid(
            detections=final_dets,
            grid_vec_along=vec_along,
            grid_vec_across=vec_across,
        )

    # ── Pair hash marks into yard lines ──────────────────────────────
    # Two hashes on the same yard line are separated along vec_along.
    # We cluster by projecting onto the axis *perpendicular* to
    # vec_along — same-yard-line hashes get the same projection.
    groups = _cluster_into_yard_lines(hash_marks, vec_across, vec_along)

    if not groups:
        return FieldGrid(
            detections=final_dets,
            grid_vec_along=vec_along,
            grid_vec_across=vec_across,
        )

    # ── Anchor yard numbers to ball position ─────────────────────────
    center_yard = round(ball_yard / 5) * 5

    if ball_image_pos is not None:
        # Find the yard-line group whose centroid is closest to the ball
        bx, by = ball_image_pos
        best_idx = 0
        best_dist = float("inf")
        across_axis = vec_across / np.linalg.norm(vec_across)
        ball_proj = bx * across_axis[0] + by * across_axis[1]
        for i, g in enumerate(groups):
            cx, cy = g["centroid"]
            g_proj = cx * across_axis[0] + cy * across_axis[1]
            d = abs(g_proj - ball_proj)
            if d < best_dist:
                best_dist = d
                best_idx = i
        anchor_idx = best_idx
    else:
        anchor_idx = len(groups) // 2

    for i, g in enumerate(groups):
        g["yard"] = center_yard + (i - anchor_idx) * 5

    # Clamp: drop any groups that fall outside 0-100
    groups = [g for g in groups if 0 <= g["yard"] <= 100]
    if not groups:
        return FieldGrid(
            detections=final_dets,
            grid_vec_along=vec_along,
            grid_vec_across=vec_across,
        )

    # Recompute anchor_idx after clamping
    anchor_idx = 0
    for i, g in enumerate(groups):
        if g["yard"] == center_yard:
            anchor_idx = i
            break

    # ── Project full grid ────────────────────────────────────────────
    yard_line_groups, projected = _project_full_grid(
        groups, vec_along, vec_across, image.shape, anchor_idx, center_yard,
    )

    return FieldGrid(
        detections=final_dets,
        grid_vec_along=vec_along,
        grid_vec_across=vec_across,
        yard_line_groups=yard_line_groups,
        projected_intersections=projected,
    )


def _cluster_into_yard_lines(
    hash_marks: list[dict],
    vec_across: np.ndarray,
    vec_along: np.ndarray | None = None,
    tolerance_frac: float = 0.35,
) -> list[dict]:
    """Cluster hash mark detections into yard-line groups.

    Two hash marks on the *same* yard line are separated along ``vec_along``
    (lateral direction).  We cluster by projecting each point onto the axis
    **perpendicular to vec_along** — same-yard-line points get nearly the
    same projection.  If ``vec_along`` is unavailable we fall back to
    ``vec_across``.

    The tolerance is a fraction of ``vec_across`` magnitude (the 5-yard
    spacing), ensuring same-line pairs cluster together while adjacent
    yard lines stay separate.
    """
    if not hash_marks:
        return []

    across_mag = np.linalg.norm(vec_across)
    if across_mag < 1:
        return []

    # Build projection axis: perpendicular to vec_along so that points
    # on the same yard line project to the same value.
    if vec_along is not None and np.linalg.norm(vec_along) > 1:
        # Perpendicular: rotate vec_along by 90° → (-ay, ax)
        perp = np.array([-vec_along[1], vec_along[0]], dtype=float)
        perp /= np.linalg.norm(perp)
    else:
        perp = vec_across / across_mag

    tolerance = tolerance_frac * across_mag

    # Project each hash mark onto the perpendicular axis
    projections = []
    for p in hash_marks:
        proj = p["x"] * perp[0] + p["y"] * perp[1]
        projections.append((proj, p))

    projections.sort(key=lambda t: t[0])

    groups = []
    current_group = [projections[0]]
    for i in range(1, len(projections)):
        if projections[i][0] - current_group[-1][0] < tolerance:
            current_group.append(projections[i])
        else:
            groups.append(current_group)
            current_group = [projections[i]]
    groups.append(current_group)

    result = []
    for g in groups:
        points = [item[1] for item in g]
        centroid_x = np.mean([p["x"] for p in points])
        centroid_y = np.mean([p["y"] for p in points])
        result.append({
            "points": points,
            "centroid": (centroid_x, centroid_y),
            "detected": True,
            "yard": 0,
        })

    return result


def _project_full_grid(
    groups: list[dict],
    vec_along: np.ndarray | None,
    vec_across: np.ndarray,
    img_shape: tuple,
    anchor_idx: int,
    anchor_yard: int,
) -> tuple[list[dict], list[dict]]:
    """Extrapolate the full field grid from detected yard-line groups.

    Returns (yard_line_groups, projected_intersections) where projected
    positions outside detected groups are marked as detected=False.
    """
    h, w = img_shape[:2]
    margin = 50

    all_groups = list(groups)
    all_intersections = []

    # First, collect all detected intersections
    for g in groups:
        for p in g["points"]:
            all_intersections.append({
                "x": p["x"], "y": p["y"],
                "yard": g["yard"],
                "detected": True,
            })

    # Project outward from detected groups
    # Forward (increasing yard)
    for direction in [1, -1]:
        last_known_centroid = None
        last_known_vec = vec_across.copy()
        i = anchor_idx

        while True:
            next_i = i + direction
            yard = anchor_yard + (next_i - anchor_idx) * 5
            if yard < 0 or yard > 100:
                break

            if 0 <= next_i < len(groups):
                # This group exists
                g = groups[next_i]
                last_known_centroid = np.array(g["centroid"])
                # Adapt local vector if we have the previous detected group
                if 0 <= i < len(groups):
                    prev_c = np.array(groups[i]["centroid"])
                    actual_disp = last_known_centroid - prev_c
                    if np.linalg.norm(actual_disp) > 10:
                        last_known_vec = actual_disp * direction
                i = next_i
                continue

            # Need to project
            if last_known_centroid is None:
                if 0 <= i < len(groups):
                    last_known_centroid = np.array(groups[i]["centroid"])
                else:
                    break

            projected_centroid = last_known_centroid + last_known_vec * direction
            px, py = projected_centroid

            # Stop if outside image bounds
            if px < -margin or px > w + margin or py < -margin or py > h + margin:
                break

            new_group = {
                "points": [],
                "centroid": (float(px), float(py)),
                "detected": False,
                "yard": yard,
            }
            all_groups.append(new_group)

            # Project hash intersection positions along this yard line
            if vec_along is not None:
                for step in [-1, 0, 1]:
                    hx = px + step * vec_along[0]
                    hy = py + step * vec_along[1]
                    if -margin <= hx <= w + margin and -margin <= hy <= h + margin:
                        all_intersections.append({
                            "x": float(hx), "y": float(hy),
                            "yard": yard,
                            "detected": False,
                        })

            last_known_centroid = projected_centroid
            i = next_i

    # Also project hash positions along detected yard lines
    if vec_along is not None:
        for g in groups:
            cx, cy = g["centroid"]
            existing_xs = [p["x"] for p in g["points"]]
            for step in [-2, -1, 1, 2]:
                hx = cx + step * vec_along[0]
                hy = cy + step * vec_along[1]
                if -margin <= hx <= w + margin and -margin <= hy <= h + margin:
                    # Don't duplicate if near an existing detection
                    if not any(abs(hx - ex) < 20 for ex in existing_xs):
                        all_intersections.append({
                            "x": float(hx), "y": float(hy),
                            "yard": g["yard"],
                            "detected": False,
                        })

    # Sort groups by yard number
    all_groups.sort(key=lambda g: g["yard"])

    return all_groups, all_intersections


# ── Grid-based homography ────────────────────────────────────────────────────

def compute_grid_homography(
    grid: FieldGrid,
    img_shape: tuple,
    ball_image_pos: tuple[int, int] | None = None,
    ball_template_pos: tuple[int, int] | None = None,
    field_type: str = "college",
) -> np.ndarray | None:
    """Compute homography from detected hash-yard intersections.

    Each detected intersection maps to a known template position based on
    its yard number and lateral hash position (near or far).

    **Near/far determination:**
    - Multi-detection yard lines (2+ detections): highest image-y = near hash.
    - Single-detection yard lines: the correct near/far assignment is found by
      **brute-force enumeration** — all 2^N combinations are tried, and the
      assignment that produces the best-conditioned homography is selected.
      This avoids the perspective-distortion pitfalls of ``vec_along``-based
      or ``vec_across``-based inference.

    Args:
        field_type: "college" or "nfl" — determines hash mark lateral positions.
    """
    detected = [p for p in grid.projected_intersections if p["detected"]]
    if len(detected) < 3:
        return None

    by_yard: dict[int, list[dict]] = {}
    for p in detected:
        by_yard.setdefault(p["yard"], []).append(p)

    near_hash_x, far_hash_x = _hash_template_x(field_type)

    # ── Phase 1: Fixed correspondences from multi-detection yard lines ──
    base_src: list[list[float]] = []
    base_dst: list[list[float]] = []

    for yard, pts in by_yard.items():
        template_y = yard_to_template_y(yard)
        if not (0 <= template_y <= TEMPLATE_H):
            continue
        if len(pts) < 2:
            continue

        # Sort by y descending: highest y = closest to camera = near
        pts.sort(key=lambda p: p["y"], reverse=True)
        base_src.append([float(pts[0]["x"]), float(pts[0]["y"])])
        base_dst.append([near_hash_x, template_y])
        base_src.append([float(pts[-1]["x"]), float(pts[-1]["y"])])
        base_dst.append([far_hash_x, template_y])

        # Middle points interpolate between near and far
        for k in range(1, len(pts) - 1):
            frac = k / (len(pts) - 1)
            tx = near_hash_x + frac * (far_hash_x - near_hash_x)
            base_src.append([pts[k]["x"], pts[k]["y"]])
            base_dst.append([tx, template_y])

    # Ball position as fixed correspondence
    if ball_image_pos is not None and ball_template_pos is not None:
        base_src.append([float(ball_image_pos[0]), float(ball_image_pos[1])])
        base_dst.append([float(ball_template_pos[0]), float(ball_template_pos[1])])

    # ── Phase 2: Collect single-detection yard lines ─────────────────
    singles: list[tuple[int, float, float]] = []   # (yard, px, py)
    for yard, pts in by_yard.items():
        template_y = yard_to_template_y(yard)
        if not (0 <= template_y <= TEMPLATE_H):
            continue
        if len(pts) != 1:
            continue
        singles.append((yard, pts[0]["x"], pts[0]["y"]))

    # ── Phase 3: Enumerate near/far assignments, pick best H ─────────
    # For each single-detection yard line, the detection could be at
    # the near hash or the far hash.  We try all 2^N combinations and
    # select the homography with the lowest condition number.
    n_singles = len(singles)

    if n_singles == 0:
        # No ambiguity — just use base correspondences
        if len(base_src) < 4:
            return None
        src = np.array(base_src, dtype=np.float32)
        dst = np.array(base_dst, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H

    best_H = None
    best_cond = float("inf")

    for combo in range(1 << n_singles):
        src_pts = list(base_src)
        dst_pts = list(base_dst)

        for bit_idx, (yard, px, py) in enumerate(singles):
            template_y = yard_to_template_y(yard)
            is_far = (combo >> bit_idx) & 1
            tx = far_hash_x if is_far else near_hash_x
            src_pts.append([float(px), float(py)])
            dst_pts.append([tx, template_y])

        if len(src_pts) < 4:
            continue

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)

        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            continue

        cond = np.linalg.cond(H)
        if cond < best_cond:
            best_cond = cond
            best_H = H

    return best_H


# ── Field overlay drawing ────────────────────────────────────────────────────

def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple,
    thickness: int = 1,
    dash_len: int = 12,
    gap_len: int = 8,
):
    """Draw a dashed line from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    dx /= dist
    dy /= dist

    pos = 0.0
    while pos < dist:
        seg_end = min(pos + dash_len, dist)
        sx = int(x1 + pos * dx)
        sy = int(y1 + pos * dy)
        ex = int(x1 + seg_end * dx)
        ey = int(y1 + seg_end * dy)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos = seg_end + gap_len


def draw_field_overlay(
    image: np.ndarray,
    grid: FieldGrid,
    H: np.ndarray | None = None,
    field_type: str = "college",
) -> np.ndarray:
    """Draw the field grid overlay on the image using homography back-projection.

    When a valid homography is available, template yard lines and hash marks
    are projected back onto the image using H⁻¹ for pixel-perfect perspective.
    Falls back to grid-vector-based drawing if H is not available.

    - Detected yard lines: solid white lines (thick)
    - Other 5-yard lines visible in the image: solid lighter lines
    - Hash marks: small cyan ticks
    - Detected intersections: filled cyan circles
    - Projected intersections: hollow cyan circles
    - Yard number labels
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    if not grid.yard_line_groups and not grid.projected_intersections:
        cv2.putText(overlay, "No grid detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return overlay

    detected_yards = {g["yard"] for g in grid.yard_line_groups if g["detected"]}

    # ── Homography-based overlay (preferred) ─────────────────────────
    if H is not None:
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = None
    else:
        H_inv = None

    if H_inv is not None:
        near_hash_x, far_hash_x = _hash_template_x(field_type)

        # Scale line thickness and font size to image resolution
        scale = max(1, int(round(max(w, h) / 1000)))
        font_scale = 0.6 * scale
        thick_line = max(2, 2 * scale)
        thin_line = max(1, scale)

        # Project template yard lines back to image
        for yard in range(0, 101, 5):
            ty = yard_to_template_y(yard)

            # Yard line endpoints: sideline to sideline
            tmpl_pts = np.array([
                [[0.0, float(ty)]],
                [[float(TEMPLATE_W), float(ty)]],
            ], dtype=np.float32)
            img_pts = cv2.perspectiveTransform(tmpl_pts, H_inv)
            pt1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
            pt2 = (int(img_pts[1][0][0]), int(img_pts[1][0][1]))

            # Skip lines that are entirely outside the image (with margin)
            margin = 200
            if (pt1[0] < -margin and pt2[0] < -margin) or \
               (pt1[0] > w + margin and pt2[0] > w + margin) or \
               (pt1[1] < -margin and pt2[1] < -margin) or \
               (pt1[1] > h + margin and pt2[1] > h + margin):
                continue

            if yard in detected_yards:
                # Bright yellow for detected yard lines
                cv2.line(overlay, pt1, pt2, (0, 255, 255), thick_line + 1)
            elif yard % 10 == 0:
                cv2.line(overlay, pt1, pt2, (255, 255, 255), thick_line)
            else:
                cv2.line(overlay, pt1, pt2, (200, 200, 200), thin_line)

            # Yard number labels on both sides of the hash marks
            if yard % 10 == 0 and 10 <= yard <= 90:
                label_yard = yard if yard <= 50 else 100 - yard
                label = str(label_yard)

                for label_x_tmpl in [near_hash_x - 40, far_hash_x + 15]:
                    label_tmpl = np.array([
                        [[float(label_x_tmpl), float(ty)]],
                    ], dtype=np.float32)
                    label_img = cv2.perspectiveTransform(label_tmpl, H_inv)
                    lx = int(label_img[0][0][0])
                    ly = int(label_img[0][0][1])
                    if -50 <= lx <= w + 50 and -50 <= ly <= h + 50:
                        cv2.putText(overlay, label, (lx, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 0, 0), int(font_scale * 6))
                        cv2.putText(overlay, label, (lx, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 255, 255), int(font_scale * 2.5))

        # Draw hash marks as cyan ticks
        for yard in range(0, 101):
            if yard % 5 == 0:
                continue  # Don't draw on yard lines
            ty = yard_to_template_y(yard)
            for hx in [near_hash_x, far_hash_x]:
                tmpl_pts = np.array([
                    [[float(hx - 8), float(ty)]],
                    [[float(hx + 8), float(ty)]],
                ], dtype=np.float32)
                img_pts = cv2.perspectiveTransform(tmpl_pts, H_inv)
                hp1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
                hp2 = (int(img_pts[1][0][0]), int(img_pts[1][0][1]))
                if (-50 <= hp1[0] <= w + 50 and -50 <= hp1[1] <= h + 50):
                    cv2.line(overlay, hp1, hp2, (255, 255, 0), thin_line)

    else:
        # ── Fallback: grid-vector-based drawing ──────────────────────
        vec_along = grid.grid_vec_along
        for g in grid.yard_line_groups:
            cx, cy = g["centroid"]
            yard = g["yard"]
            if vec_along is not None:
                along_norm = vec_along / np.linalg.norm(vec_along)
                extent = max(w, h)
                pt1 = (int(cx - extent * along_norm[0]),
                        int(cy - extent * along_norm[1]))
                pt2 = (int(cx + extent * along_norm[0]),
                        int(cy + extent * along_norm[1]))
            else:
                pt1 = (0, int(cy))
                pt2 = (w - 1, int(cy))

            if g["detected"]:
                cv2.line(overlay, pt1, pt2, (255, 255, 255), 2)
            else:
                _draw_dashed_line(overlay, pt1, pt2, (200, 200, 200), 1)

            if 0 <= yard <= 100:
                label_yard = yard if yard <= 50 else 100 - yard
                label = str(label_yard)
                lx, ly = int(cx), int(cy)
                if 0 <= lx < w and 0 <= ly < h:
                    cv2.putText(overlay, label, (lx - 8, ly - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(overlay, label, (lx - 8, ly - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ── Draw detected intersections (always) ─────────────────────────
    r = max(5, int(3 * max(1, max(w, h) / 1000)))
    for p in grid.projected_intersections:
        px, py = int(p["x"]), int(p["y"])
        if not (-20 <= px <= w + 20 and -20 <= py <= h + 20):
            continue
        if p["detected"]:
            cv2.circle(overlay, (px, py), r + 2, (0, 0, 0), 2)
            cv2.circle(overlay, (px, py), r, (0, 255, 255), -1)
        else:
            cv2.circle(overlay, (px, py), r, (0, 255, 255), 1)

    return overlay


# ── Field template drawing ───────────────────────────────────────────────────

def draw_field_template(
    orientation: str = "vertical",
    field_type: str = "college",
    direction: str = "right",
) -> np.ndarray:
    """Draw an overhead football field template.

    Template coordinates (portrait / internal):
      - Y axis = along the field (0 = far end zone, 1200 = near end zone)
      - X axis = across the field (0 = left sideline, 533 = right sideline)
      - Yard N from the left end zone: y = (N + 10) * TEMPLATE_SCALE

    Args:
        orientation: "vertical" (tall, default) or "horizontal" (wide, landscape)
        field_type: "college" or "nfl" — determines hash mark lateral positions.
        direction: "right" (offense goes left→right) or "left" (offense goes right→left).
                   Only applies when orientation="horizontal".
    """
    template = np.zeros((TEMPLATE_H, TEMPLATE_W, 3), dtype=np.uint8)
    template[:] = (34, 139, 34)  # Forest green

    # Yard lines every 5 yards
    for yd in range(0, FIELD_LENGTH_YD + 1, 5):
        y = int(yd * TEMPLATE_SCALE)
        thickness = 2 if yd % 10 == 0 else 1
        cv2.line(template, (0, y), (TEMPLATE_W - 1, y), (255, 255, 255), thickness)

    # Sidelines
    cv2.line(template, (0, 0), (0, TEMPLATE_H - 1), (255, 255, 255), 2)
    cv2.line(template, (TEMPLATE_W - 1, 0), (TEMPLATE_W - 1, TEMPLATE_H - 1), (255, 255, 255), 2)

    # End zone lines
    ez = int(10 * TEMPLATE_SCALE)
    cv2.line(template, (0, ez), (TEMPLATE_W - 1, ez), (255, 255, 255), 3)
    cv2.line(template, (0, TEMPLATE_H - ez), (TEMPLATE_W - 1, TEMPLATE_H - ez), (255, 255, 255), 3)

    # Yard numbers on both sides
    for yd in range(10, 51, 10):
        y_top = int((yd + 10) * TEMPLATE_SCALE)
        y_bot = int((FIELD_LENGTH_YD - yd - 10) * TEMPLATE_SCALE)
        num = str(yd)
        # Left side numbers
        cv2.putText(template, num, (TEMPLATE_W // 6 - 10, y_top + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Right side numbers
        cv2.putText(template, num, (TEMPLATE_W * 5 // 6 - 10, y_top + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if yd < 50:
            cv2.putText(template, num, (TEMPLATE_W // 6 - 10, y_bot + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(template, num, (TEMPLATE_W * 5 // 6 - 10, y_bot + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hash marks at field-type-specific positions
    hash_left_x, hash_right_x = _hash_template_x(field_type)
    for yd in range(10, 111):
        y = int(yd * TEMPLATE_SCALE)
        if yd % 5 != 0:  # Don't overlap yard lines
            cv2.line(template, (hash_left_x - 5, y), (hash_left_x + 5, y), (255, 255, 255), 1)
            cv2.line(template, (hash_right_x - 5, y), (hash_right_x + 5, y), (255, 255, 255), 1)

    if orientation == "horizontal":
        # CCW rotation: portrait (px, py) → landscape (py, W-1-px)
        # This puts yard 0 (low template Y) on the LEFT side of the image.
        template = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if direction == "left":
            # Flip horizontally so yard 0 is on the RIGHT (offense going left)
            template = cv2.flip(template, 1)

    return template


def yard_to_template_y(yard_number: int) -> int:
    """Convert a yard line number (0-100) to template Y coordinate.

    The template has 10-yard end zones on each side, so:
      yard 0 (goal line) → y = 10 * SCALE = 100
      yard 50 → y = 60 * SCALE = 600
      yard 100 (far goal line) → y = 110 * SCALE = 1100
    """
    return int((yard_number + 10) * TEMPLATE_SCALE)


def template_y_to_yard(y: int) -> int:
    """Convert template Y coordinate to yard line number."""
    yard = (y / TEMPLATE_SCALE) - 10
    return max(0, min(100, int(round(yard))))


# ── Anchor yard lines to real yard numbers ───────────────────────────────────

def assign_yard_numbers(
    markings: FieldMarkings,
    ball_yard: int,
    img_shape: tuple,
) -> list[tuple[np.ndarray, int]]:
    """Assign real yard numbers to detected yard lines.

    Strategy:
      1. Find which detected yard line is closest to the ball position
         (approximated as the center of the image).
      2. Snap that line to the nearest 5-yard mark near ball_yard.
      3. Assign adjacent lines at +/- 5 yard intervals.

    Returns list of (extended_line_endpoints, yard_number).
    """
    if not markings.yard_lines:
        return []

    h, w = img_shape[:2]

    # Sort yard lines by their perpendicular position
    yard_lines = sorted(markings.yard_lines, key=lambda yl: yl.position)

    # The ball is approximately at the center of the image.
    # Find which yard line position is closest to the image center.
    center_pos = _perp_position_from_point(w / 2, h / 2, markings.dominant_angle)
    positions = [yl.position for yl in yard_lines]
    center_idx = int(np.argmin([abs(p - center_pos) for p in positions]))

    # Snap ball_yard to nearest 5-yard mark
    center_yard = round(ball_yard / 5) * 5

    # Assign yard numbers to all lines
    assignments = []
    for i, yl in enumerate(yard_lines):
        offset = i - center_idx
        yard = center_yard + offset * 5

        # Validate: yard numbers must be 0-100
        if 0 <= yard <= 100:
            extended = yl.extended if len(yl.extended) == 4 else _extend_line(yl.representative, img_shape)
            assignments.append((extended, yard))

    return assignments


def _perp_position_from_point(x: float, y: float, ref_angle: float) -> float:
    """Project a point onto the axis perpendicular to ref_angle."""
    perp_rad = np.deg2rad(ref_angle + 90)
    return x * np.cos(perp_rad) + y * np.sin(perp_rad)


# ── Homography from anchored yard lines ──────────────────────────────────────

def _line_intersect(l1, l2):
    """Find intersection of two infinite lines given as (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = [float(v) for v in l1]
    x3, y3, x4, y4 = [float(v) for v in l2]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def _deduplicate_correspondences(
    src: list[list[float]], dst: list[list[float]], min_img_dist: float = 15.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Remove near-duplicate correspondence points.

    When two source points are closer than min_img_dist pixels, keep only
    the first one. This handles duplicate yard-line detections that place
    nearly-identical image points at very different template positions.
    """
    if not src:
        return src, dst
    keep_src, keep_dst = [src[0]], [dst[0]]
    for s, d in zip(src[1:], dst[1:]):
        too_close = False
        for ks in keep_src:
            dx = s[0] - ks[0]
            dy = s[1] - ks[1]
            if (dx * dx + dy * dy) < min_img_dist * min_img_dist:
                too_close = True
                break
        if not too_close:
            keep_src.append(s)
            keep_dst.append(d)
    return keep_src, keep_dst


def compute_anchored_homography(
    assignments: list[tuple[np.ndarray, int]],
    dominant_angle: float,
    img_shape: tuple,
    markings=None,
    ball_image_pos: tuple[int, int] | None = None,
    ball_template_pos: tuple[int, int] | None = None,
    field_type: str = "college",
) -> np.ndarray | None:
    """Compute homography from hash-mark intersections, yard line endpoints,
    and an optional ball anchor.

    Correspondence point sources:
      1. Hash mark × yard line intersections — each hash row at a known
         lateral yard position crossed with each assigned yard line gives
         a precise 2D correspondence.
      2. Yard line endpoints — each extended yard line gives two points.
         In a sideline camera the bottom of the image is approximately the
         near sideline (camera is ON the sideline), and the top of the
         image shows roughly to the far hash mark area.  Endpoints are
         mapped to conservative lateral estimates:
           - bottom endpoint → near sideline (template x ≈ 0)
           - top endpoint → far hash position (template x ≈ far_hash)
      3. Ball position anchor — user-clicked ball in both image and
         template gives one high-quality 2D correspondence.

    Near-duplicate image points (from double-detected yard lines) are
    filtered before computing the homography.

    Returns 3x3 homography matrix or None.
    """
    if len(assignments) < 2:
        return None

    h, w = img_shape[:2]
    margin = max(w, h) * 0.1

    near_hash_yd, far_hash_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    near_hash_x = int(near_hash_yd * TEMPLATE_SCALE)
    far_hash_x = int(far_hash_yd * TEMPLATE_SCALE)

    src_points = []
    dst_points = []

    # ── Hash mark × yard line intersections (primary) ───────────────────
    # Determine near vs far hash by IMAGE position, not by the label from
    # field_markings.py (which uses abstract geometry, not camera perspective).
    # In a sideline camera: low y = top of image = far side of field,
    # high y = bottom of image = near side (camera side).
    hash_lines = []  # store fitted hash lines for later use
    if markings is not None and hasattr(markings, "hash_rows") and markings.hash_rows:
        # Sort rows by average y-position
        rows_with_y = []
        for row in markings.hash_rows:
            avg_y = float(np.mean([hm.midpoint[1] for hm in row.marks]))
            rows_with_y.append((avg_y, row))
        rows_with_y.sort(key=lambda t: t[0])  # lowest y first = far side

        for i, (avg_y, row) in enumerate(rows_with_y):
            if len(rows_with_y) == 2:
                # Two rows: lower-y = far hash, higher-y = near hash
                tmpl_x = far_hash_x if i == 0 else near_hash_x
            else:
                # Single row: use image position
                tmpl_x = far_hash_x if avg_y < h * 0.5 else near_hash_x

            pts = [list(hm.midpoint) for hm in row.marks]
            if len(pts) < 2:
                continue
            pts_arr = np.array(pts, dtype=np.float32)
            [vx, vy, x0, y0] = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
            hash_line = np.array([
                x0 - vx * 5000, y0 - vy * 5000,
                x0 + vx * 5000, y0 + vy * 5000,
            ])
            hash_lines.append((hash_line, tmpl_x))

            for yl_line, yard in assignments:
                pt = _line_intersect(yl_line, hash_line)
                if pt is None:
                    continue
                if (-margin <= pt[0] <= w + margin and
                        -margin <= pt[1] <= h + margin):
                    template_y = yard_to_template_y(yard)
                    src_points.append([pt[0], pt[1]])
                    dst_points.append([tmpl_x, template_y])

    # ── Yard line endpoints ─────────────────────────────────────────────
    # Bottom endpoint → near sideline (x=0 on template).
    #   The camera sits on the near sideline, so the bottom of the frame
    #   is approximately at the sideline.
    # Top endpoint → far hash row position.
    #   Sideline cameras typically show up to the far hash area but rarely
    #   all the way to the far sideline.  Using far_hash_x is conservative.
    for line, yard in assignments:
        template_y = yard_to_template_y(yard)
        x1, y1, x2, y2 = [float(v) for v in line]

        if y1 <= y2:
            pt_top, pt_bot = [x1, y1], [x2, y2]
        else:
            pt_top, pt_bot = [x2, y2], [x1, y1]

        top_ok = (-margin <= pt_top[0] <= w + margin and
                  -margin <= pt_top[1] <= h + margin)
        bot_ok = (-margin <= pt_bot[0] <= w + margin and
                  -margin <= pt_bot[1] <= h + margin)

        if top_ok:
            src_points.append(pt_top)
            dst_points.append([far_hash_x, template_y])
        if bot_ok:
            src_points.append(pt_bot)
            dst_points.append([0, template_y])

    # ── Ball position anchor ────────────────────────────────────────────
    if ball_image_pos is not None and ball_template_pos is not None:
        src_points.append([float(ball_image_pos[0]), float(ball_image_pos[1])])
        dst_points.append([float(ball_template_pos[0]), float(ball_template_pos[1])])

    # ── Deduplicate ─────────────────────────────────────────────────────
    src_points, dst_points = _deduplicate_correspondences(src_points, dst_points)

    if len(src_points) < 4:
        return None

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


# ── Player detection (blended YOLO + Roboflow) ──────────────────────────────

# Roboflow role colors (BGR) for drawing on field template
ROLE_COLORS_BGR = {
    "qb":      (0, 0, 255),       # red
    "oline":   (0, 180, 0),       # green
    "skill":   (255, 80, 0),      # blue
    "defense": (0, 165, 255),     # orange
    "ref":     (0, 255, 255),     # yellow
}

# Roles that are actual players (not refs/coaches)
PLAYER_ROLES = {"qb", "oline", "skill", "defense"}


def _bbox_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _roboflow_detect(image_bgr: np.ndarray, confidence: int = 30) -> list[dict]:
    """Call Roboflow football-presnap-tracker API.

    Returns list of {"bbox": (x1,y1,x2,y2), "role": str, "conf": float}.
    Returns empty list if API key is not set or request fails.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        return []

    try:
        _, buf = cv2.imencode(".jpg", image_bgr)
        b64 = base64.b64encode(buf).decode("utf-8")
        url = (
            f"https://detect.roboflow.com/football-presnap-tracker/6"
            f"?api_key={api_key}&confidence={confidence}&overlap=40"
        )
        req = urllib.request.Request(
            url, data=b64.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        preds = json.loads(resp.read())["predictions"]
    except Exception:
        return []

    results = []
    for p in preds:
        x1 = int(p["x"] - p["width"] / 2)
        y1 = int(p["y"] - p["height"] / 2)
        x2 = int(p["x"] + p["width"] / 2)
        y2 = int(p["y"] + p["height"] / 2)
        results.append({
            "bbox": (x1, y1, x2, y2),
            "role": p["class"],
            "conf": p["confidence"],
        })
    return results


def _segmentation_detect(image_bgr: np.ndarray) -> list[dict]:
    """Call Roboflow segmentation workflow for player detection.

    Returns list of dicts with keys:
        bbox: (x1, y1, x2, y2)
        conf: float
        class: str ("player" or "ref")
        mask: np.ndarray (binary mask for this player, same size as image)
    Returns empty list on failure.
    """
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        return []

    api_key = os.getenv("ROBOFLOW_SEGMENTATION_API_KEY", "")
    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        return []

    try:
        # Save image to temp file for the SDK
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, image_bgr)

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        result = client.run_workflow(
            workspace_name="isaiahs-workspace",
            workflow_id="general-segmentation-api-3",
            images={"image": tmp_path},
            parameters={"classes": "player, ref"},
            use_cache=True,
        )
        os.unlink(tmp_path)

        preds = result[0]["predictions"]["predictions"]
    except Exception:
        return []

    h, w = image_bgr.shape[:2]
    results = []
    for p in preds:
        x1 = int(p["x"] - p["width"] / 2)
        y1 = int(p["y"] - p["height"] / 2)
        x2 = int(p["x"] + p["width"] / 2)
        y2 = int(p["y"] + p["height"] / 2)

        # Build binary mask from polygon points
        mask = None
        points = p.get("points", [])
        if points:
            pts = np.array([(int(pt["x"]), int(pt["y"])) for pt in points])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

        results.append({
            "bbox": (x1, y1, x2, y2),
            "conf": p["confidence"],
            "class": p["class"],
            "mask": mask,
        })
    return results


def _yolo_detect(image_bgr: np.ndarray, conf: float = 0.3) -> list[dict]:
    """Run YOLOv8m person detection.

    Returns list of {"bbox": (x1,y1,x2,y2), "conf": float}.
    """
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    results = model(image_bgr, conf=conf, classes=[0], verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": box.conf.item(),
        })
    return detections


def _match_by_iou(
    dets_a: list[dict],
    dets_b: list[dict],
    iou_thresh: float = 0.3,
) -> tuple[list[tuple], set, set]:
    """Match detections from two sources by IoU.

    Returns (matches, matched_a_indices, matched_b_indices).
    Each match is (a_idx, b_idx, iou).
    """
    matched_a = set()
    matched_b = set()
    matches = []

    for bi, bd in enumerate(dets_b):
        best_iou = 0
        best_ai = -1
        for ai, ad in enumerate(dets_a):
            if ai in matched_a:
                continue
            iou = _bbox_iou(ad["bbox"], bd["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ai = ai
        if best_iou >= iou_thresh and best_ai >= 0:
            matches.append((best_ai, bi, best_iou))
            matched_a.add(best_ai)
            matched_b.add(bi)

    return matches, matched_a, matched_b


def detect_player_positions(
    image: np.ndarray,
    conf: float = 0.3,
    iou_thresh: float = 0.3,
) -> list[dict]:
    """Blended player detection: segmentation for recall, presnap for filtering.

    Pipeline:
      1. Segmentation model (primary) — finds all people with polygon masks
      2. Presnap model — provides role labels and sideline filtering
      3. Match seg ↔ presnap by IoU
      4. Seg + presnap player role (qb/oline/skill/defense) → KEEP
      5. Seg + presnap ref → DROP
      6. Seg with no presnap match → use field mask: on grass = KEEP, off = DROP
      7. Unmatched presnap player roles → KEEP (presnap found someone seg missed)

    Returns list of dicts with keys:
        center, foot, bbox, conf, role, source, mask (optional)
    """
    from detect_players import build_field_mask, is_on_field

    # Run the two models that matter
    seg_dets = _segmentation_detect(image)
    rf_dets = _roboflow_detect(image, confidence=int(conf * 100))

    # Build field mask for filtering unmatched detections
    field_mask = build_field_mask(image)

    def _make_player(bbox, conf_val, role, source, mask=None):
        x1, y1, x2, y2 = bbox
        p = {
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "foot": ((x1 + x2) // 2, y2),
            "bbox": bbox,
            "conf": conf_val,
            "role": role,
            "source": source,
        }
        if mask is not None:
            p["mask"] = mask
        return p

    # If segmentation failed, fall back to presnap + field mask
    if not seg_dets:
        players = []
        for rd in rf_dets:
            if rd["role"] == "ref":
                continue
            players.append(_make_player(
                rd["bbox"], rd["conf"], rd["role"], "presnap",
            ))
        if not players:
            # Last resort: YOLO
            yolo_dets = _yolo_detect(image, conf)
            for yd in yolo_dets:
                if is_on_field(list(yd["bbox"]), field_mask):
                    players.append(_make_player(
                        yd["bbox"], yd["conf"], "unknown", "yolo",
                    ))
        return players

    # Match segmentation ↔ presnap by IoU
    seg_to_rf = {}  # seg_idx → rf_idx
    rf_matched = set()

    for si, sd in enumerate(seg_dets):
        best_iou = 0
        best_ri = -1
        for ri, rd in enumerate(rf_dets):
            if ri in rf_matched:
                continue
            iou = _bbox_iou(sd["bbox"], rd["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ri = ri
        if best_iou >= iou_thresh and best_ri >= 0:
            seg_to_rf[si] = best_ri
            rf_matched.add(best_ri)

    players = []

    # Process all segmentation detections
    for si, sd in enumerate(seg_dets):
        mask = sd.get("mask")
        bbox = sd["bbox"]
        c = sd["conf"]

        if si in seg_to_rf:
            # Matched with presnap — use presnap's role to decide
            rd = rf_dets[seg_to_rf[si]]
            role = rd["role"]
            c = max(c, rd["conf"])

            if role == "ref":
                continue  # Drop refs

            players.append(_make_player(bbox, c, role, "seg+presnap", mask))

        else:
            # No presnap match — use field mask to decide
            # If player's feet are on green grass, they're on the field
            if is_on_field(list(bbox), field_mask):
                players.append(_make_player(
                    bbox, c, "unknown", "seg+field", mask,
                ))
            # else: sideline person, drop

    # Unmatched presnap detections (presnap found someone segmentation missed)
    # BUT: skip if they overlap significantly with an already-kept player
    # (the presnap model sometimes double-detects a player with different roles)
    for ri, rd in enumerate(rf_dets):
        if ri in rf_matched:
            continue
        if rd["role"] == "ref":
            continue

        # Check for overlap with already-kept players
        is_duplicate = False
        for existing in players:
            if _bbox_iou(rd["bbox"], existing["bbox"]) > 0.2:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        players.append(_make_player(
            rd["bbox"], rd["conf"], rd["role"], "presnap",
        ))

    return players


def project_players_to_field(
    players: list[dict],
    H: np.ndarray,
    img_shape: tuple,
) -> list[dict]:
    """Project player foot positions through homography to field coordinates.

    Returns list of {"field_pos": (x, y), "yard": float, "lateral": float, ...}
    """
    if H is None or not players:
        return []

    # Use foot positions for projection (most accurate ground-plane point)
    feet = np.array([[p["foot"][0], p["foot"][1]] for p in players], dtype=np.float32)
    feet = feet.reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(feet, H)
    projected = projected.reshape(-1, 2)

    result = []
    for i, p in enumerate(players):
        fx, fy = projected[i]
        # Convert template coords to yard numbers
        yard = template_y_to_yard(int(fy))
        lateral = fx / TEMPLATE_W * FIELD_WIDTH_YD

        # Only keep players that project onto the field
        if 0 <= fx <= TEMPLATE_W and 0 <= fy <= TEMPLATE_H:
            result.append({
                **p,
                "field_pos": (float(fx), float(fy)),
                "yard": yard,
                "lateral": lateral,
            })

    return result


# ── Visualization ────────────────────────────────────────────────────────────

TEAM_COLORS_BGR = {
    0: (0, 165, 255),     # orange
    1: (255, 50, 50),     # blue
    -1: (180, 180, 180),  # gray
}


def _player_color_bgr(team_label: int | None = None) -> tuple:
    """Pick a color for a player dot based on team label only.

    0 = offense (orange), 1 = defense (blue), -1/None = gray.
    """
    if team_label is not None and team_label >= 0:
        return TEAM_COLORS_BGR.get(team_label, (180, 180, 180))
    return (180, 180, 180)


def _portrait_to_landscape(
    fx: float, fy: float, direction: str = "right",
) -> tuple[int, int]:
    """Convert portrait template coords to landscape pixel coords.

    After 90° CCW rotation: portrait (px, py) → landscape (py, W-1-px).
    With direction="left", an additional horizontal flip is applied.
    """
    lx = int(fy)
    ly = TEMPLATE_W - 1 - int(fx)
    if direction == "left":
        lx = TEMPLATE_H - 1 - lx
    return lx, ly


def draw_players_on_field(
    field_players: list[dict],
    ball_pos: tuple[int, int] | None = None,
    team_labels: list[int] | None = None,
    field_type: str = "college",
    orientation: str = "horizontal",
    direction: str = "right",
) -> np.ndarray:
    """Draw projected player positions on a field template.

    Colors by team label: orange = offense, blue = defense, gray = unknown.

    Args:
        ball_pos: (template_x, template_y) position of the ball on the template
                  (portrait coords), or None to skip drawing the ball.
        field_type: "college" or "nfl" — determines hash mark positions on template.
        orientation: "vertical" (portrait) or "horizontal" (landscape).
        direction: "right" or "left" — offense direction (landscape only).

    Returns BGR image of the field with player dots.
    """
    template = draw_field_template(
        orientation=orientation, field_type=field_type, direction=direction,
    )
    is_landscape = orientation == "horizontal"

    def _to_px(fx: float, fy: float) -> tuple[int, int]:
        """Map portrait template coords to output pixel coords."""
        if is_landscape:
            return _portrait_to_landscape(fx, fy, direction)
        return int(fx), int(fy)

    # Draw ball position
    if ball_pos is not None:
        bx, by = _to_px(ball_pos[0], ball_pos[1])
        cv2.circle(template, (bx, by), 8, (0, 200, 255), -1)
        yard = template_y_to_yard(ball_pos[1])
        cv2.putText(template, f"BALL {yard}yd", (bx + 12, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Draw players
    for i, p in enumerate(field_players):
        fx, fy = p["field_pos"]
        x, y = _to_px(fx, fy)

        team = team_labels[i] if team_labels and i < len(team_labels) else None
        color = _player_color_bgr(team)

        cv2.circle(template, (x, y), 7, color, -1)
        cv2.circle(template, (x, y), 7, (0, 0, 0), 1)  # outline

        # Label: O (offense), D (defense), ? (unknown)
        label = "O" if team == 0 else ("D" if team == 1 else "?")
        cv2.putText(template, label, (x - 4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Draw legend
    legend_y = 20
    for label, team_id in [("offense", 0), ("defense", 1), ("unknown", -1)]:
        color = TEAM_COLORS_BGR.get(team_id, (180, 180, 180))
        cv2.circle(template, (15, legend_y), 6, color, -1)
        cv2.putText(template, label, (27, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 18

    return template


def draw_correspondences_debug(
    image: np.ndarray,
    assignments: list[tuple[np.ndarray, int]],
    players: list[dict] | None = None,
    ball_image_pos: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw yard line assignments, player positions, and ball anchor on the image."""
    debug = image.copy()

    colors = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
    ]

    for i, (line, yard) in enumerate(assignments):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = line
        cv2.line(debug, (x1, y1), (x2, y2), color, 3)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(debug, f"{yard}yd", (mx - 20, my - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Mark correspondence points
        cv2.circle(debug, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(debug, (x2, y2), 8, (0, 255, 0), -1)

    # Draw ball anchor point
    if ball_image_pos is not None:
        bx, by = ball_image_pos
        cv2.circle(debug, (bx, by), 14, (0, 200, 255), 3)
        cv2.circle(debug, (bx, by), 5, (0, 200, 255), -1)
        cv2.putText(debug, "BALL ANCHOR", (bx + 18, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Draw player foot positions
    if players:
        for p in players:
            fx, fy = p["foot"]
            cv2.circle(debug, (fx, fy), 5, (0, 200, 255), -1)

    return debug


# ── Full pipeline ────────────────────────────────────────────────────────────

def _match_precomputed_labels(
    field_players: list[dict],
    precomputed: list[tuple[tuple[int, int, int, int], int]],
    iou_threshold: float = 0.3,
) -> list[int | None]:
    """Match pre-computed team labels to field_players by bbox IoU.

    Args:
        field_players: player dicts with 'bbox' or 'x','y','w','h' keys.
        precomputed: list of (bbox_xyxy, team_label) from the Team Assignment tab.
        iou_threshold: minimum IoU to accept a match.

    Returns:
        list of matched labels (int) or None where no match found.
    """
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        return inter / (a_area + b_area - inter)

    matched = []
    for p in field_players:
        bbox = p.get("bbox")
        if bbox is None:
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            bbox = (x, y, x + w, y + h)

        best_iou = 0.0
        best_label = None
        for pre_bbox, pre_label in precomputed:
            score = _iou(bbox, pre_bbox)
            if score > best_iou:
                best_iou = score
                best_label = pre_label
        matched.append(best_label if best_iou >= iou_threshold else None)
    return matched


def run_interactive_homography(
    image: np.ndarray,
    ball_yard: int,
    ball_template_pos: tuple[int, int] | None = None,
    ball_image_pos: tuple[int, int] | None = None,
    conf: float = 0.3,
    field_type: str = "college",
    offense_direction: str | None = None,
    precomputed_team_labels: list[tuple[tuple[int, int, int, int], int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Full interactive homography pipeline.

    Uses the hash-yards-intersection model as primary source for field
    geometry, with Hough-based detection as fallback.

    Args:
        image: BGR input image
        ball_yard: yard line number where the ball is (0-100)
        ball_template_pos: (x, y) on the template where user clicked, or None
                           to place at center of the yard line
        ball_image_pos: (x, y) pixel position of ball in the image, or None.
        conf: YOLO confidence threshold
        field_type: "college" or "nfl" — determines hash mark positions.
        precomputed_team_labels: optional list of (bbox_xyxy, label) from the
            Team Assignment tab.  When provided, labels are transferred by
            bbox IoU match and classify_teams_multi is NOT re-run.

    Returns:
        (field_with_players, overlay, correspondences_debug, warped_image, summary_text)
    """
    h, w = image.shape[:2]
    ph = np.full((TEMPLATE_H, TEMPLATE_W, 3), 40, dtype=np.uint8)
    homography_source = "none"

    # Step 1: Try grid-based detection (primary)
    grid = None
    H = None
    assignments = []
    try:
        grid = detect_field_grid(image, ball_yard, ball_image_pos=ball_image_pos)
        detected_count = sum(1 for p in grid.projected_intersections if p["detected"])
        if detected_count >= 2:
            H = compute_grid_homography(
                grid, image.shape,
                ball_image_pos=ball_image_pos,
                ball_template_pos=ball_template_pos,
                field_type=field_type,
            )
            if H is not None:
                homography_source = "grid"
    except Exception:
        pass

    # Step 2: Fallback to Hough-based detection
    if H is None:
        markings = detect_field_markings(image)
        if markings.yard_lines:
            assignments = assign_yard_numbers(markings, ball_yard, image.shape)
            if len(assignments) >= 2:
                H = compute_anchored_homography(
                    assignments, markings.dominant_angle, image.shape,
                    markings=markings,
                    ball_image_pos=ball_image_pos,
                    ball_template_pos=ball_template_pos,
                    field_type=field_type,
                )
                if H is not None:
                    homography_source = "hough"

    # Step 3: Draw field overlay (always, even if homography fails)
    if grid is not None:
        overlay = draw_field_overlay(image, grid, H=H, field_type=field_type)
    else:
        overlay = image.copy()

    if H is None:
        msg = "Homography failed (both grid and Hough)"
        return ph, overlay, image.copy(), ph, msg

    # Step 4: Blended player detection
    players = detect_player_positions(image, conf)

    # Step 5: Project players to field
    field_players = project_players_to_field(players, H, image.shape)

    # Step 6: Team classification
    #   If pre-computed labels were supplied (from the Team Assignment tab),
    #   transfer them by bbox IoU instead of re-running classification.
    from team_classifier import classify_teams_multi

    if precomputed_team_labels:
        matched = _match_precomputed_labels(field_players, precomputed_team_labels)

        # The TA tab uses unsupervised K-Means, so cluster IDs (0/1) are
        # arbitrary.  We need to align them with the FM tab's semantic
        # labels (0=offense, 1=defense) using player positions relative
        # to the line of scrimmage.
        has_yards = all("yard" in p for p in field_players)
        if has_yards and ball_yard is not None and offense_direction is not None:
            # Count how many cluster-0 and cluster-1 players are on each
            # side of the ball.
            off_side_0, off_side_1 = 0, 0
            for p, m in zip(field_players, matched):
                if m is None or m == -1:
                    continue
                yd = p["yard"]
                if offense_direction == "right":
                    on_off_side = yd > ball_yard
                else:
                    on_off_side = yd < ball_yard
                if on_off_side:
                    if m == 0:
                        off_side_0 += 1
                    else:
                        off_side_1 += 1
            # If cluster-1 has more players on the offense side than
            # cluster-0, the labels are swapped → flip them.
            need_flip = off_side_1 > off_side_0
            if need_flip:
                matched = [
                    (1 - m if m is not None and m >= 0 else m)
                    for m in matched
                ]

        team_labels = [m if m is not None else -1 for m in matched]
        team_diags = [
            {"position_signal": None, "position_zone": None,
             "color_signal": m, "color_hsv": None,
             "confidence": 0.8 if m is not None else 0.0,
             "conflict": False, "source": "precomputed"}
            for m in matched
        ]
    else:
        team_labels, team_diags = classify_teams_multi(
            field_players, image, ball_yard, offense_direction,
        )
    # ── Post-process: flip obvious outliers on wrong side of ball ──────
    #   After team labels are assigned (either precomputed or fresh), a
    #   single player may still be mislabelled.  If one side of the LOS
    #   has an overwhelming majority of one label, flip the minority.
    _has_yards = all("yard" in p for p in field_players)
    if _has_yards and ball_yard is not None and offense_direction is not None:
        _OUTLIER_MARGIN = 0.5  # yards from LOS to ignore (right at the line)
        off_side: list[int] = []   # indices on offense side
        def_side: list[int] = []   # indices on defense side
        for i, p in enumerate(field_players):
            if team_labels[i] == -1:
                continue
            yd = p["yard"]
            if abs(yd - ball_yard) < _OUTLIER_MARGIN:
                continue  # too close to LOS, ambiguous
            if offense_direction == "right":
                on_off = yd > ball_yard
            else:
                on_off = yd < ball_yard
            if on_off:
                off_side.append(i)
            else:
                def_side.append(i)

        def _flip_outliers(indices: list[int], expected_label: int) -> None:
            """Flip minority labels on one side of the ball."""
            if len(indices) < 3:
                return
            n_expected = sum(1 for i in indices if team_labels[i] == expected_label)
            n_other = len(indices) - n_expected
            # Only flip when the majority is clear (≥3 correct) and
            # the minority is small (≤25% of the majority, min 1).
            if n_expected >= 3 and 0 < n_other <= max(1, n_expected // 4):
                for i in indices:
                    if team_labels[i] != expected_label:
                        team_labels[i] = expected_label

        _flip_outliers(off_side, 0)   # offense side → expect label 0
        _flip_outliers(def_side, 1)   # defense side → expect label 1

    for i, diag in enumerate(team_diags):
        field_players[i]["team_diag"] = diag

    # Step 7: Ball position on template
    if ball_template_pos is None:
        ball_template_pos = (TEMPLATE_W // 2, yard_to_template_y(ball_yard))

    # Step 8: Draw outputs
    field_img = draw_players_on_field(
        field_players, ball_template_pos, team_labels,
        field_type=field_type,
        orientation="horizontal",
        direction=offense_direction or "right",
    )
    corr_debug = draw_correspondences_debug(image, assignments, players, ball_image_pos)
    warped = cv2.warpPerspective(image, H, (TEMPLATE_W, TEMPLATE_H))

    # Summary
    source_counts = {}
    for p in field_players:
        src = p.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    grid_info = ""
    if grid is not None:
        det_count = sum(1 for p in grid.projected_intersections if p["detected"])
        proj_count = sum(1 for p in grid.projected_intersections if not p["detected"])
        yl_det = sum(1 for g in grid.yard_line_groups if g["detected"])
        yl_proj = sum(1 for g in grid.yard_line_groups if not g["detected"])
        grid_info = (
            f"\nGrid: {det_count} detected + {proj_count} projected intersections"
            f"\nYard lines: {yl_det} detected + {yl_proj} projected"
        )

    # Team classification stats
    off_count = sum(1 for l in team_labels if l == 0)
    def_count = sum(1 for l in team_labels if l == 1)
    unk_count = sum(1 for l in team_labels if l == -1)
    high_conf = sum(1 for d in team_diags if d["confidence"] >= 0.7)
    conflicts = sum(1 for d in team_diags if d["conflict"])
    zone_counts = {}
    for d in team_diags:
        z = d.get("position_zone") or "no_pos"
        zone_counts[z] = zone_counts.get(z, 0) + 1

    dir_label = offense_direction or "not set"
    ball_anchor = "yes" if ball_image_pos else "no"
    parts = [
        f"Field type: {field_type}",
        f"Ball at {ball_yard}-yard line (image anchor: {ball_anchor})",
        f"Offense direction: {dir_label}",
        f"Homography source: {homography_source}",
        f"Players detected: {len(players)} (refs filtered)",
        f"Players on field: {len(field_players)}",
        f"Sources: {', '.join(f'{k}={v}' for k, v in sorted(source_counts.items()))}",
        f"Teams: offense={off_count}, defense={def_count}, unknown={unk_count}",
        f"Zones: {', '.join(f'{k}={v}' for k, v in sorted(zone_counts.items()))}",
        f"Classification: {high_conf}/{len(team_labels)} high-confidence, {conflicts} conflicts",
    ]
    if grid_info:
        parts.append(grid_info)

    return field_img, overlay, corr_debug, warped, "\n".join(parts), field_players, team_labels
