"""
Detect painted yard numbers and directional arrows on the football field.

Strategy (no OCR — painted field numbers are too stylized for standard OCR):
1. Build a white-marking mask and subtract detected yard lines
2. Morphological cleanup to connect broken digit strokes
3. Find "0" digits as anchors via hollowness (ring-shaped = high edge fill,
   low center fill)
4. Pair each "0" with the nearest tens-digit blob to its left
5. Detect triangular directional arrows near number groups
6. Infer yard number from relative spacing and arrow direction

Arrow direction + yard number -> exact field position, which directly feeds
the homography estimation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from field_homography import build_field_mask, detect_white_lines_mask, detect_field_lines


# Hollowness threshold: above this, a blob is classified as "0".
# Empirically calibrated: real "0" shapes score ~0.35+; other digits < 0.20.
_HOLLOWNESS_THRESH = 0.25


# ── Low-level helpers ────────────────────────────────────────────────────────

def _remove_yard_lines(
    white_mask: np.ndarray,
    image: np.ndarray,
    field_mask: np.ndarray,
) -> np.ndarray:
    """Subtract detected yard-line pixels from the white mask.

    Uses a thin brush (15 px) so that digits crossed by a yard line
    lose only a narrow stripe and can be reconnected by morphological closing.
    """
    lines, _ = detect_field_lines(image, field_mask)
    if lines is None:
        return white_mask.copy()

    line_mask = np.zeros_like(white_mask)
    for x1, y1, x2, y2 in lines:
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 15)

    return cv2.subtract(white_mask, line_mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Aggressive morphological cleanup to reconnect broken digit strokes."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open)
    return out


def _compute_hollowness(
    mask: np.ndarray, x: int, y: int, w: int, h: int,
) -> float:
    """Measure how hollow a blob is (edge fill minus center fill).

    A "0" digit is a ring: high edge fill, low center fill -> positive.
    Other digits and noise are not hollow -> near zero or negative.
    """
    region = mask[y : y + h, x : x + w]
    if region.size == 0:
        return 0.0

    mx = max(1, w // 4)
    my = max(1, h // 4)

    center = region[my : h - my, mx : w - mx]
    center_fill = np.count_nonzero(center) / center.size if center.size else 0

    edge_mask = np.ones_like(region, dtype=bool)
    edge_mask[my : h - my, mx : w - mx] = False
    edge_px = region[edge_mask]
    edge_fill = np.count_nonzero(edge_px) / edge_px.size if edge_px.size else 0

    return edge_fill - center_fill


def _find_blobs(
    mask: np.ndarray,
    image_shape: tuple,
    hollow_mask: np.ndarray | None = None,
    min_area: int = 600,
    max_area: int = 20000,
    max_aspect: float = 5.0,
    bottom_frac: float = 0.50,
) -> list[dict]:
    """Find external contour blobs in the bottom portion of the mask."""
    h = image_shape[0]
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    blobs = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y < h * (1.0 - bottom_frac):
            continue
        aspect = bw / max(bh, 1)
        if aspect > max_aspect:
            continue

        hmask = hollow_mask if hollow_mask is not None else mask
        hollowness = _compute_hollowness(hmask, x, y, bw, bh)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        blobs.append({
            "cnt": cnt,
            "x": x, "y": y, "w": bw, "h": bh,
            "area": area, "hollowness": hollowness,
            "solidity": solidity, "aspect": aspect,
        })

    return blobs


def _merge_nearby_blobs(
    blobs: list[dict],
    x_gap: int = 15,
    y_tolerance: int = 30,
) -> list[dict]:
    """Merge horizontally adjacent blobs (same digit split by line removal).

    Only merges blobs with similar heights (within 2x).
    """
    if len(blobs) <= 1:
        return blobs

    blobs = sorted(blobs, key=lambda b: b["x"])
    merged = [blobs[0]]

    for b in blobs[1:]:
        prev = merged[-1]
        gap = b["x"] - (prev["x"] + prev["w"])
        y_diff = abs(b["y"] - prev["y"])
        h_ratio = max(prev["h"], b["h"]) / max(min(prev["h"], b["h"]), 1)

        if gap < x_gap and y_diff < y_tolerance and h_ratio < 2.0:
            nx = min(prev["x"], b["x"])
            ny = min(prev["y"], b["y"])
            nw = max(prev["x"] + prev["w"], b["x"] + b["w"]) - nx
            nh = max(prev["y"] + prev["h"], b["y"] + b["h"]) - ny
            prev["x"], prev["y"], prev["w"], prev["h"] = nx, ny, nw, nh
            prev["area"] += b["area"]
            prev["hollowness"] = max(prev["hollowness"], b["hollowness"])
            if b["area"] > cv2.contourArea(prev["cnt"]):
                prev["cnt"] = b["cnt"]
        else:
            merged.append(b)

    return merged


# ── Arrow detection ──────────────────────────────────────────────────────────

def _detect_arrows(
    blobs: list[dict],
    image_shape: tuple,
) -> list[dict]:
    """Identify directional arrow blobs and determine their pointing direction.

    Arrows are thin, wide, small shapes.  Direction is inferred by comparing
    the centroid position to the bounding-box center (the heavy tail pulls
    the centroid away from the pointed tip).
    """
    arrows = []
    for b in blobs:
        if b["aspect"] < 2.5 or b["h"] > 30 or b["area"] > 2500:
            continue

        cnt = b["cnt"]
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        hull = cv2.convexHull(cnt).squeeze()
        if hull.ndim != 2:
            continue

        # The tip is farther from the centroid than the tail.
        dist_left = cx - hull[:, 0].min()
        dist_right = hull[:, 0].max() - cx
        direction = "left" if dist_left > dist_right else "right"

        arrows.append({
            "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
            "center": (int(cx), int(cy)),
            "direction": direction,
        })

    return arrows


# ── Grouping into yard markers ───────────────────────────────────────────────

def _group_markers(
    blobs: list[dict],
    arrows: list[dict],
    x_proximity: int = 250,
) -> list[dict]:
    """Group "0" anchors with nearby tens-digit blobs and arrows."""
    zeros = [b for b in blobs if b["hollowness"] > _HOLLOWNESS_THRESH]
    tens_candidates = [
        b for b in blobs
        if b["hollowness"] <= _HOLLOWNESS_THRESH
        and 800 < b["area"] < 8000
        and 0.2 < b["aspect"] < 3.0
    ]

    markers: list[dict] = []

    for z in zeros:
        z_left = z["x"]
        z_cy = z["y"] + z["h"] / 2

        # Find nearest tens-digit to the LEFT of this "0"
        best_ten = None
        best_dist = float("inf")
        for t in tens_candidates:
            dx = z_left - t["x"]
            dy = abs(z_cy - (t["y"] + t["h"] / 2))
            if 0 < dx < x_proximity and dy < 80 and dx < best_dist:
                best_dist = dx
                best_ten = t

        if best_ten is None:
            continue

        # Bounding box spanning both digits
        mx = min(best_ten["x"], z["x"])
        my = min(best_ten["y"], z["y"])
        mw = max(best_ten["x"] + best_ten["w"], z["x"] + z["w"]) - mx
        mh = max(best_ten["y"] + best_ten["h"], z["y"] + z["h"]) - my

        # Collect nearby arrows
        marker_arrows = []
        for a in arrows:
            a_cx, a_cy = a["center"]
            if abs(a_cy - z_cy) < 80 and abs(a_cx - (mx + mw / 2)) < x_proximity * 1.5:
                marker_arrows.append(a)

        direction = marker_arrows[0]["direction"] if marker_arrows else None

        markers.append({
            "zero": z,
            "tens": best_ten,
            "bbox": (mx, my, mw, mh),
            "center": (mx + mw // 2, my + mh // 2),
            "direction": direction,
            "arrows": marker_arrows,
        })

    _infer_yard_numbers(markers)
    return markers


def _infer_yard_numbers(markers: list[dict]) -> None:
    """Assign the actual yard number to each detected marker.

    With a single marker, defaults to 20 (most common broadcast view).
    With multiple markers, uses relative spacing (always 10 yards apart)
    and arrow direction to assign numbers.
    """
    if not markers:
        return

    if len(markers) == 1:
        m = markers[0]
        m["number"] = 20
        m["confidence"] = 0.4
        return

    markers.sort(key=lambda m: m["center"][0])

    arrow_dir = None
    for m in markers:
        if m["direction"]:
            arrow_dir = m["direction"]
            break

    # Arrows point toward the nearer end zone.
    # LEFT arrow -> left end zone is nearer -> numbers increase rightward.
    increasing = arrow_dir == "left" if arrow_dir else True

    base = 20
    for i, m in enumerate(markers):
        offset = i * 10 if increasing else -i * 10
        m["number"] = base + offset
        m["confidence"] = 0.6 if arrow_dir else 0.3


# ── Main API ─────────────────────────────────────────────────────────────────

def detect_field_numbers(
    image: np.ndarray,
    white_mask: np.ndarray | None = None,
    field_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Detect painted yard numbers and directional arrows on the field.

    Returns a list of dicts (or a (markers, debug_info) tuple when
    *return_debug* is True):
        {
            "number": int (10, 20, 30, 40, or 50),
            "confidence": float (0-1),
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "direction": "left" | "right" | None,
            "arrows": [{"direction": ..., "center": ..., ...}, ...],
        }
    """
    if field_mask is None:
        field_mask = build_field_mask(image)
    if white_mask is None:
        white_mask = detect_white_lines_mask(image, field_mask)

    # 1. Remove yard lines and clean the mask
    no_lines = _remove_yard_lines(white_mask, image, field_mask)
    cleaned = _clean_mask(no_lines)

    # 2. Find blobs (use pre-closing mask for hollowness)
    blobs = _find_blobs(cleaned, image.shape, hollow_mask=no_lines)

    # 3. Merge blobs split by line removal, then recompute hollowness
    blobs = _merge_nearby_blobs(blobs)
    for b in blobs:
        b["hollowness"] = _compute_hollowness(
            no_lines, b["x"], b["y"], b["w"], b["h"],
        )

    # 4. Detect arrows (pre-merge blobs, looser thresholds)
    arrow_blobs = _find_blobs(
        cleaned, image.shape, min_area=300, max_area=5000, max_aspect=8.0,
    )
    arrows = _detect_arrows(arrow_blobs, image.shape)

    # 5. Group and infer yard numbers
    markers = _group_markers(blobs, arrows)

    if return_debug:
        debug_info = {
            "white_mask": white_mask,
            "no_lines": no_lines,
            "cleaned": cleaned,
            "blobs": blobs,
            "arrows": arrows,
        }
        return markers, debug_info
    return markers


# ── Debug visualisation ──────────────────────────────────────────────────────

def _draw_debug_panel(
    image: np.ndarray,
    markers: list[dict],
    debug_info: dict,
) -> np.ndarray:
    """Build a multi-panel debug image showing every pipeline stage.

    Layout (2x2 grid, each panel is half the original size):
        ┌──────────────┬──────────────┐
        │ white mask   │ cleaned mask │
        │  (no lines)  │  + blobs     │
        ├──────────────┼──────────────┤
        │ blob classif │ final result │
        │  colour-coded│              │
        └──────────────┴──────────────┘
    """
    h, w = image.shape[:2]
    ph, pw = h // 2, w // 2  # panel size

    def _resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

    def _to_bgr(mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    blobs = debug_info["blobs"]
    arrows = debug_info["arrows"]

    # ── Panel 1: no-lines mask ───────────────────────────────────────────
    p1 = _resize(_to_bgr(debug_info["no_lines"]))
    cv2.putText(p1, "1. White mask (yard lines removed)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # ── Panel 2: cleaned mask with all blob bboxes ───────────────────────
    p2 = _resize(_to_bgr(debug_info["cleaned"]))
    scale = pw / w  # coord scale factor for half-size panels
    for b in blobs:
        x1 = int(b["x"] * scale)
        y1 = int(b["y"] * scale)
        x2 = int((b["x"] + b["w"]) * scale)
        y2 = int((b["y"] + b["h"]) * scale)
        cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 255, 0), 1)
        lbl = f"h={b['hollowness']:.2f}"
        cv2.putText(p2, lbl, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(p2, "2. Cleaned mask + blobs (hollowness)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # ── Panel 3: original image with colour-coded blob classification ────
    p3 = _resize(image.copy())
    # Colours: green = "0" anchor, blue = tens candidate, cyan = arrow, gray = rejected
    COLOR_ZERO = (0, 255, 0)
    COLOR_TENS = (255, 100, 0)
    COLOR_ARROW = (0, 200, 255)
    COLOR_REJECT = (120, 120, 120)

    # Identify which blobs are used in markers
    used_zeros = {id(m["zero"]) for m in markers}
    used_tens = {id(m["tens"]) for m in markers}

    for b in blobs:
        x1 = int(b["x"] * scale)
        y1 = int(b["y"] * scale)
        x2 = int((b["x"] + b["w"]) * scale)
        y2 = int((b["y"] + b["h"]) * scale)

        if b["hollowness"] > _HOLLOWNESS_THRESH:
            color = COLOR_ZERO
            tag = '"0"'
        elif 800 < b["area"] < 8000 and 0.2 < b["aspect"] < 3.0:
            color = COLOR_TENS if id(b) in used_tens else (200, 150, 50)
            tag = "tens"
        else:
            color = COLOR_REJECT
            tag = "rej"

        cv2.rectangle(p3, (x1, y1), (x2, y2), color, 2)
        cv2.putText(p3, tag, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for a in arrows:
        acx = int(a["center"][0] * scale)
        acy = int(a["center"][1] * scale)
        tip_dx = -15 if a["direction"] == "left" else 15
        cv2.arrowedLine(p3, (acx - tip_dx, acy), (acx + tip_dx, acy),
                        COLOR_ARROW, 2, tipLength=0.5)
        cv2.putText(p3, "arrow", (acx - 20, acy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_ARROW, 1)

    cv2.putText(p3, "3. Blob classification", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    # Legend
    ly = ph - 60
    for label, color in [("0-anchor", COLOR_ZERO), ("tens-digit", COLOR_TENS),
                         ("arrow", COLOR_ARROW), ("rejected", COLOR_REJECT)]:
        cv2.rectangle(p3, (10, ly), (25, ly + 12), color, -1)
        cv2.putText(p3, label, (30, ly + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        ly += 16

    # ── Panel 4: final result ────────────────────────────────────────────
    p4 = _resize(image.copy())
    for det in markers:
        x, y, bw, bh = det["bbox"]
        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + bw) * scale)
        y2 = int((y + bh) * scale)
        cv2.rectangle(p4, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label = f"{det['number']}"
        if det["direction"]:
            ch = "<" if det["direction"] == "left" else ">"
            label = f"{ch} {label} {ch}"
        label += f"  conf={det['confidence']:.2f}"
        cv2.putText(p4, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for arrow in det["arrows"]:
            acx = int(arrow["center"][0] * scale)
            acy = int(arrow["center"][1] * scale)
            tip_dx = -20 if arrow["direction"] == "left" else 20
            cv2.arrowedLine(p4, (acx - tip_dx, acy), (acx + tip_dx, acy),
                            (0, 200, 255), 2, tipLength=0.4)

    if not markers:
        cv2.putText(p4, "No yard numbers detected", (pw // 4, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(p4, "4. Final detections", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # ── Assemble 2x2 grid ────────────────────────────────────────────────
    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    return np.vstack([top, bot])


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect yard numbers on football field",
    )
    parser.add_argument(
        "images", nargs="*", default=["steelers.jpg", "texas.jpg"],
        help="Image paths to process",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate debug images",
    )
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for image_path in args.images:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {image_path}: file not found")
            continue

        print(f"\nProcessing {path.name}...")

        field_mask = build_field_mask(image)
        white_mask = detect_white_lines_mask(image, field_mask)

        detections, debug_info = detect_field_numbers(
            image, white_mask, field_mask, return_debug=True,
        )

        if not detections:
            print("  No yard numbers detected")
        else:
            for det in detections:
                arrow_str = ""
                if det["direction"]:
                    arrow_str = f"  arrow={det['direction']}"
                print(
                    f"  Found {det['number']}-yard line"
                    f"  conf={det['confidence']:.2f}"
                    f"  at ({det['center'][0]}, {det['center'][1]})"
                    f"{arrow_str}"
                )

        # Draw final annotated image
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 3)

            label = f"{det['number']}"
            if det["direction"]:
                arrow_char = "<" if det["direction"] == "left" else ">"
                label = f"{arrow_char} {label} {arrow_char}"
            label += f" ({det['confidence']:.2f})"
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            for arrow in det["arrows"]:
                acx, acy = arrow["center"]
                tip_dx = -30 if arrow["direction"] == "left" else 30
                cv2.arrowedLine(
                    annotated,
                    (acx - tip_dx, acy), (acx + tip_dx, acy),
                    (0, 200, 255), 3, tipLength=0.4,
                )

        out_path = output_dir / f"{path.stem}_numbers{path.suffix}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved -> {out_path}")

        if args.debug:
            # Multi-panel pipeline debug image
            panel = _draw_debug_panel(image, detections, debug_info)
            debug_path = output_dir / f"{path.stem}_numbers_debug{path.suffix}"
            cv2.imwrite(str(debug_path), panel)
            print(f"  Saved debug panel -> {debug_path}")

            # Also save raw intermediate masks
            cv2.imwrite(
                str(output_dir / f"{path.stem}_no_lines.jpg"),
                debug_info["no_lines"],
            )
            cv2.imwrite(
                str(output_dir / f"{path.stem}_cleaned.jpg"),
                debug_info["cleaned"],
            )


if __name__ == "__main__":
    main()
