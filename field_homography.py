"""
Field homography estimation using classical CV.

Detects yard lines via Canny + Hough transform on the green field region,
clusters them into line groups, identifies intersection points, and computes
a homography to project the image onto a standard overhead field template.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Field dimensions (in yards, converted to a pixel template) ──────────────
# Standard American football field: 100 yards + two 10-yard end zones = 120 yards
# Width: 53⅓ yards
FIELD_LENGTH_YD = 120
FIELD_WIDTH_YD = 53.33

# Template scale: pixels per yard
TEMPLATE_SCALE = 10
TEMPLATE_W = int(FIELD_WIDTH_YD * TEMPLATE_SCALE)   # 533 px
TEMPLATE_H = int(FIELD_LENGTH_YD * TEMPLATE_SCALE)   # 1200 px

# ── Green field detection (same as detect_players.py) ───────────────────────
GREEN_LOW = np.array([30, 40, 40])
GREEN_HIGH = np.array([80, 255, 255])


def build_field_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of the green playing field."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def detect_white_lines_mask(image: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    """Isolate white field markings (yard lines, hash marks) on the green field.

    Returns a binary mask of white-on-green pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White pixels: low saturation, high value (brightness)
    # Use adaptive thresholding on the V channel to handle varying lighting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    field_gray = cv2.bitwise_and(gray, gray, mask=field_mask)

    # Compute local mean brightness on the field to set adaptive threshold
    field_pixels = gray[field_mask > 0]
    if len(field_pixels) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    mean_val = np.mean(field_pixels)

    # White lines are brighter than surrounding grass — threshold relative to mean
    v_thresh = max(120, int(mean_val + 40))
    s_thresh = 80  # Allow slightly more saturation for night games

    white_mask = cv2.inRange(hsv, np.array([0, 0, v_thresh]), np.array([180, s_thresh, 255]))

    # Only keep white pixels that are on the field
    white_on_field = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)

    # Morphological cleanup: close small gaps in lines, remove noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_on_field = cv2.morphologyEx(white_on_field, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_on_field = cv2.morphologyEx(white_on_field, cv2.MORPH_OPEN, kernel_open)

    return white_on_field


def detect_field_lines(image: np.ndarray, field_mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Detect lines on the field using white pixel isolation + Hough transform.

    Returns (lines as [[x1, y1, x2, y2], ...] or None, white_mask for debug).
    """
    white_mask = detect_white_lines_mask(image, field_mask)

    # Edge detection on the white mask only
    edges = cv2.Canny(white_mask, 50, 150)

    # Scale-adaptive line length: longer minimum for larger images
    h = image.shape[0]
    min_line_length = max(60, h // 10)

    # Hough line detection with tighter thresholds
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        return None, white_mask

    return lines.reshape(-1, 4), white_mask


def line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle of a line segment in degrees (0-180)."""
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180


def cluster_lines_by_angle(
    lines: np.ndarray, angle_threshold: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Split lines into two dominant angle groups (yard lines vs sidelines).

    Returns (group_a, group_b) where group_a is the larger group.
    """
    angles = np.array([line_angle(*line) for line in lines])

    # Use the median angle as the primary direction
    median_angle = np.median(angles)

    # Lines close to median = group A, far from median = group B
    diff = np.abs(angles - median_angle)
    # Handle wraparound at 0/180
    diff = np.minimum(diff, 180 - diff)

    mask_a = diff < angle_threshold
    mask_b = ~mask_a

    return lines[mask_a], lines[mask_b]


def line_midpoint_along_axis(line: np.ndarray, axis: str = "x") -> float:
    """Get the midpoint of a line projected onto an axis."""
    x1, y1, x2, y2 = line
    if axis == "x":
        return (x1 + x2) / 2
    return (y1 + y2) / 2


def cluster_parallel_lines(
    lines: np.ndarray, min_gap: float = 20
) -> list[np.ndarray]:
    """Cluster parallel lines by their position (perpendicular distance).

    Groups lines that are close together (same yard line detected multiple times)
    and returns one representative line per cluster.
    """
    if len(lines) == 0:
        return []

    # Determine the dominant direction
    angles = [line_angle(*line) for line in lines]
    median_angle = np.median(angles)

    # Project midpoints perpendicular to the dominant direction
    perp_axis = "y" if abs(median_angle) < 45 or abs(median_angle - 180) < 45 else "x"
    positions = np.array([line_midpoint_along_axis(l, perp_axis) for l in lines])

    # Sort by position
    sorted_idx = np.argsort(positions)
    sorted_lines = lines[sorted_idx]
    sorted_positions = positions[sorted_idx]

    # Cluster by proximity
    clusters = []
    current_cluster = [sorted_lines[0]]
    current_pos = sorted_positions[0]

    for i in range(1, len(sorted_lines)):
        if sorted_positions[i] - current_pos < min_gap:
            current_cluster.append(sorted_lines[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_lines[i]]
            current_pos = sorted_positions[i]
    clusters.append(current_cluster)

    # Take the longest line from each cluster as representative
    representatives = []
    for cluster in clusters:
        lengths = [np.hypot(l[2] - l[0], l[3] - l[1]) for l in cluster]
        representatives.append(cluster[np.argmax(lengths)])

    return representatives


def extend_line(line: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Extend a line segment to span the full image."""
    x1, y1, x2, y2 = line.astype(float)
    h, w = img_shape[:2]

    if abs(x2 - x1) < 1:  # Near-vertical
        return np.array([int(x1), 0, int(x2), h - 1])

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Extend to image boundaries
    pts = []
    # Left edge
    y_at_0 = intercept
    if 0 <= y_at_0 < h:
        pts.append((0, y_at_0))
    # Right edge
    y_at_w = slope * (w - 1) + intercept
    if 0 <= y_at_w < h:
        pts.append((w - 1, y_at_w))
    # Top edge
    if abs(slope) > 1e-6:
        x_at_0 = -intercept / slope
        if 0 <= x_at_0 < w:
            pts.append((x_at_0, 0))
    # Bottom edge
    if abs(slope) > 1e-6:
        x_at_h = (h - 1 - intercept) / slope
        if 0 <= x_at_h < w:
            pts.append((x_at_h, h - 1))

    if len(pts) < 2:
        return line

    # Take the two points farthest apart
    pts = np.array(pts)
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return np.array([int(pts[i][0]), int(pts[i][1]), int(pts[j][0]), int(pts[j][1])])


def find_intersections(
    lines_a: list[np.ndarray], lines_b: list[np.ndarray], img_shape: tuple
) -> list[tuple[float, float]]:
    """Find intersection points between two groups of lines.

    Only returns intersections that fall within the image bounds.
    """
    h, w = img_shape[:2]
    intersections = []

    for la in lines_a:
        for lb in lines_b:
            pt = _line_intersection(la, lb)
            if pt is not None:
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    intersections.append((x, y))

    return intersections


def _line_intersection(
    line1: np.ndarray, line2: np.ndarray
) -> tuple[float, float] | None:
    """Compute intersection of two line segments (extended to infinite lines)."""
    x1, y1, x2, y2 = line1.astype(float)
    x3, y3, x4, y4 = line2.astype(float)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return (ix, iy)


def draw_field_template() -> np.ndarray:
    """Draw a simple overhead football field template."""
    template = np.zeros((TEMPLATE_H, TEMPLATE_W, 3), dtype=np.uint8)
    template[:] = (34, 139, 34)  # Forest green

    # Draw yard lines every 10 yards (every 5 for minor)
    for yd in range(0, FIELD_LENGTH_YD + 1, 5):
        y = int(yd * TEMPLATE_SCALE)
        thickness = 2 if yd % 10 == 0 else 1
        cv2.line(template, (0, y), (TEMPLATE_W - 1, y), (255, 255, 255), thickness)

    # Draw sidelines
    cv2.line(template, (0, 0), (0, TEMPLATE_H - 1), (255, 255, 255), 2)
    cv2.line(template, (TEMPLATE_W - 1, 0), (TEMPLATE_W - 1, TEMPLATE_H - 1), (255, 255, 255), 2)

    # Draw end zone lines
    ez = int(10 * TEMPLATE_SCALE)
    cv2.line(template, (0, ez), (TEMPLATE_W - 1, ez), (255, 255, 255), 3)
    cv2.line(template, (0, TEMPLATE_H - ez), (TEMPLATE_W - 1, TEMPLATE_H - ez), (255, 255, 255), 3)

    # Yard numbers
    for yd in range(10, 51, 10):
        y_top = int((yd + 10) * TEMPLATE_SCALE)
        y_bot = int((FIELD_LENGTH_YD - yd - 10) * TEMPLATE_SCALE)
        num = str(yd) if yd < 50 else "50"
        cv2.putText(template, num, (TEMPLATE_W // 4, y_top + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if yd < 50:
            cv2.putText(template, num, (TEMPLATE_W // 4, y_bot + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return template


def estimate_homography(image_path: str, output_dir: Path) -> np.ndarray | None:
    """Full pipeline: detect lines, find intersections, compute homography.

    Returns the 3x3 homography matrix or None if estimation fails.
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"\nProcessing {path.name} ({w}x{h})...")

    # Step 1: Field mask
    field_mask = build_field_mask(image)

    # Step 2: Detect white field markings and lines
    lines, white_mask = detect_field_lines(image, field_mask)

    white_path = output_dir / f"{path.stem}_white_mask{path.suffix}"
    cv2.imwrite(str(white_path), white_mask)
    print(f"  Saved white mask → {white_path}")

    if lines is None or len(lines) < 4:
        print("  Not enough lines detected")
        return None
    print(f"  Detected {len(lines)} raw line segments")

    # Step 3: Keep only lines near the dominant angle (yard lines)
    angles = np.array([line_angle(*line) for line in lines])
    median_angle = np.median(angles)
    angle_diff = np.abs(angles - median_angle)
    angle_diff = np.minimum(angle_diff, 180 - angle_diff)
    yard_line_mask = angle_diff < 15.0
    yard_lines_raw = lines[yard_line_mask]
    print(f"  {len(yard_lines_raw)} lines near dominant angle ({median_angle:.1f}°)")

    # Step 4: Cluster into distinct yard lines
    min_gap = max(20, h // 40)
    yard_lines = cluster_parallel_lines(yard_lines_raw, min_gap=min_gap)
    print(f"  Clustered into {len(yard_lines)} distinct yard lines")

    if len(yard_lines) < 2:
        print("  Need at least 2 yard lines for homography")
        return None

    # Extend yard lines to full length and sort by position
    extended = [extend_line(l, image.shape) for l in yard_lines]

    # Sort lines left-to-right (or top-to-bottom depending on orientation)
    # Use perpendicular position to the dominant line direction
    perp_axis = "y" if abs(median_angle) < 45 or abs(median_angle - 180) < 45 else "x"
    positions = [line_midpoint_along_axis(np.array(l), perp_axis) for l in extended]
    sort_order = np.argsort(positions)
    extended = [extended[i] for i in sort_order]

    # Draw debug visualization
    debug = image.copy()
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255)]
    for i, line in enumerate(extended):
        color = colors[i % len(colors)]
        cv2.line(debug, (line[0], line[1]), (line[2], line[3]), color, 3)
        mid_x = (line[0] + line[2]) // 2
        mid_y = (line[1] + line[3]) // 2
        cv2.putText(debug, f"L{i}", (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    debug_path = output_dir / f"{path.stem}_lines_debug{path.suffix}"
    cv2.imwrite(str(debug_path), debug)
    print(f"  Saved line debug → {debug_path}")

    # Step 5: Build point correspondences from yard lines
    # Each yard line gives us 2 points (its two endpoints after extension).
    # We assume consecutive detected lines are 5 yards apart.
    # Place them centered in the template.
    #
    # Key insight: yard lines run sideline-to-sideline. We need to figure out
    # which endpoint is the "left sideline" and which is the "right sideline"
    # in field coordinates. The template is oriented with:
    #   - Y axis = along the field (yard lines are horizontal)
    #   - X axis = across the field (sideline to sideline)
    #
    # For each extended line, we need to consistently assign endpoints to
    # the "near sideline" (x=0) and "far sideline" (x=TEMPLATE_W).
    # We sort each line's endpoints so pt_a is always the one with smaller
    # (x+y) sum — this gives consistent left/top assignment across all lines.
    n_lines = len(extended)
    center_yd = 50
    start_yd = center_yd - ((n_lines - 1) * 5) / 2

    src_points = []
    dst_points = []

    for i, line in enumerate(extended):
        x1, y1, x2, y2 = line
        yd = start_yd + i * 5
        template_y = int((yd + 10) * TEMPLATE_SCALE)  # +10 for end zone offset

        # Consistently order endpoints: pt_a = smaller sum, pt_b = larger sum
        if (x1 + y1) <= (x2 + y2):
            pt_a, pt_b = [x1, y1], [x2, y2]
        else:
            pt_a, pt_b = [x2, y2], [x1, y1]

        # pt_a → near sideline (x=0), pt_b → far sideline (x=TEMPLATE_W)
        src_points.append(pt_a)
        dst_points.append([0, template_y])

        src_points.append(pt_b)
        dst_points.append([TEMPLATE_W - 1, template_y])

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    print(f"  Using {len(src_points)} point correspondences from {n_lines} yard lines")

    # Draw correspondence points
    for pt in src_points:
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
    corr_path = output_dir / f"{path.stem}_correspondences{path.suffix}"
    cv2.imwrite(str(corr_path), debug)
    print(f"  Saved correspondences → {corr_path}")

    # Step 6: Compute homography
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None:
        print("  Homography computation failed")
        return None

    inliers = int(status.sum()) if status is not None else len(src_points)
    print(f"  Homography computed ({inliers}/{len(src_points)} inliers)")

    # Warp the image to bird's eye view
    template = draw_field_template()
    warped = cv2.warpPerspective(image, H, (TEMPLATE_W, TEMPLATE_H))

    # Blend warped image with template
    blend = cv2.addWeighted(template, 0.3, warped, 0.7, 0)

    warp_path = output_dir / f"{path.stem}_birdseye{path.suffix}"
    cv2.imwrite(str(warp_path), warped)
    print(f"  Saved bird's-eye view → {warp_path}")

    blend_path = output_dir / f"{path.stem}_blend{path.suffix}"
    cv2.imwrite(str(blend_path), blend)
    print(f"  Saved blended overlay → {blend_path}")

    return H


def order_points(pts: np.ndarray) -> np.ndarray | None:
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    if len(pts) < 4:
        return None

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Estimate field homography")
    parser.add_argument(
        "images",
        nargs="*",
        default=["steelers.jpg", "texas.jpg"],
        help="Image paths to process",
    )
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for image_path in args.images:
        estimate_homography(image_path, output_dir)


if __name__ == "__main__":
    main()
