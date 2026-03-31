"""
Detect and classify football field markings: yard lines, hash marks, sidelines.

Uses player-masked white pixel isolation, Hough line detection with two passes
(long lines + short segments), angle-based clustering, and geometric validation
to reliably label every visible marking on the field.

Returns structured results that can feed into homography estimation.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO

from field_homography import build_field_mask


# ── Lazy YOLO for player masking ─────────────────────────────────────────────

_yolo_model = None


def _get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8m.pt")
    return _yolo_model


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    """A detected line segment with classification."""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float         # degrees 0-180
    length: float        # pixels
    label: str = ""      # "yard_line", "hash_mark", "sideline", "unknown"
    confidence: float = 0.0
    group: int = -1      # angle group index (0 = dominant, 1 = perpendicular)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])


@dataclass
class LineCluster:
    """A group of segments representing the same physical marking."""
    segments: list[Segment]
    label: str              # "yard_line", "sideline"
    position: float         # perpendicular distance (for sorting)
    representative: np.ndarray = field(default_factory=lambda: np.array([]))
    extended: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class HashMarkRow:
    """A row of hash marks at a consistent lateral position."""
    marks: list[Segment]
    lateral_position: float   # average lateral position of the row
    label: str = ""           # "near_inbound", "far_inbound", "near_sideline", "far_sideline"


@dataclass
class FieldMarkings:
    """Complete set of detected and classified field markings."""
    yard_lines: list[LineCluster]
    sidelines: list[LineCluster]
    hash_marks: list[Segment]
    hash_rows: list[HashMarkRow]  # hash marks organized by lateral row
    all_segments: list[Segment]
    white_mask: np.ndarray
    dominant_angle: float   # degrees — the yard line direction
    perp_angle: float       # degrees — the sideline/hash mark direction


# ── White mask with player removal ───────────────────────────────────────────

def _build_player_mask(image: np.ndarray, conf: float = 0.25, pad: int = 8) -> np.ndarray:
    """Detect people and return a mask covering their bounding boxes."""
    h, w = image.shape[:2]
    model = _get_yolo()
    results = model(image, conf=conf, classes=[0], verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        mask[y1:y2, x1:x2] = 255
    return mask


def detect_white_mask_clean(
    image: np.ndarray,
    field_mask: np.ndarray | None = None,
    player_mask: np.ndarray | None = None,
    s_thresh: int = 80,
    v_offset: int = 40,
) -> np.ndarray:
    """Isolate white field markings with player contamination removed.

    Steps:
      1. Build green field mask (or use provided)
      2. Subtract player bounding boxes from field mask
      3. Threshold for white (low saturation, high value) on clean field
      4. Morphological cleanup to bridge line gaps and remove noise
      5. Tophat filtering to remove remaining large blobs (jerseys not
         caught by player detection)
    """
    if field_mask is None:
        field_mask = build_field_mask(image)
    if player_mask is None:
        player_mask = _build_player_mask(image)

    # Field minus players
    clean_field = cv2.bitwise_and(field_mask, cv2.bitwise_not(player_mask))

    # Adaptive brightness threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    field_pixels = gray[clean_field > 0]
    if len(field_pixels) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    mean_val = np.mean(field_pixels)
    v_thresh = max(120, int(mean_val + v_offset))

    # White detection in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, v_thresh]), np.array([180, s_thresh, 255]))
    white = cv2.bitwise_and(white, white, mask=clean_field)

    # Morphological cleanup: bridge small gaps in lines, remove noise
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k_open)

    # Tophat filter: keeps only thin structures (field markings ~4 inches wide),
    # removes large blobs (residual jerseys/equipment not caught by player mask).
    # Kernel size controls max thickness preserved — 11px works for most resolutions.
    h = image.shape[0]
    tophat_k = max(9, min(15, h // 100))
    k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_k, tophat_k))
    white = cv2.morphologyEx(white, cv2.MORPH_TOPHAT, k_tophat)

    return white


def detect_local_contrast_mask(
    image: np.ndarray,
    field_mask: np.ndarray,
    player_mask: np.ndarray,
    contrast_thresh: int = 25,
) -> np.ndarray:
    """Build a white mask using local contrast instead of global thresholds.

    Subtracts a blurred background from the grayscale image, then thresholds
    to find anything locally brighter than its surroundings. This catches
    dim or worn hash marks that fail global HSV thresholds.

    The blur kernel is sized to treat hash marks (~2ft wide) as local
    features while smoothing out the field background.
    """
    h = image.shape[0]
    clean_field = cv2.bitwise_and(field_mask, cv2.bitwise_not(player_mask))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Blur kernel: large enough to smooth over hash marks but not yard lines.
    # Cap at 51 to avoid over-smoothing on large images.
    blur_k = min(51, max(31, (h // 20) | 1))
    background = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    local_contrast = gray - background

    _, mask = cv2.threshold(local_contrast, contrast_thresh, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = cv2.bitwise_and(mask, mask, mask=clean_field)

    # Light cleanup — remove single-pixel noise but preserve tiny hash marks
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    return mask


# ── Line segment detection ───────────────────────────────────────────────────

def _segment_angle(x1, y1, x2, y2) -> float:
    """Angle of a segment in degrees [0, 180)."""
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180


def _segment_length(x1, y1, x2, y2) -> float:
    return np.hypot(x2 - x1, y2 - y1)


def detect_segments(
    white_mask: np.ndarray,
    img_h: int,
) -> list[Segment]:
    """Detect line segments using a two-pass Hough approach.

    Pass 1: Long lines (yard lines, sidelines) — high min length, large gap bridging
    Pass 2: Short segments (hash marks) — low min length, small gap
    """
    edges = cv2.Canny(white_mask, 50, 150)
    segments = []
    seen = set()  # deduplicate across passes

    # Pass 1: Long lines
    min_len_long = max(40, img_h // 12)
    lines_long = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=min_len_long, maxLineGap=25,
    )
    if lines_long is not None:
        for l in lines_long.reshape(-1, 4):
            key = tuple(l)
            if key not in seen:
                seen.add(key)
                segments.append(Segment(
                    x1=l[0], y1=l[1], x2=l[2], y2=l[3],
                    angle=_segment_angle(*l),
                    length=_segment_length(*l),
                ))

    # Pass 2: Short segments (hash marks, tick marks)
    min_len_short = max(10, img_h // 40)
    lines_short = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=20,
        minLineLength=min_len_short, maxLineGap=5,
    )
    if lines_short is not None:
        for l in lines_short.reshape(-1, 4):
            key = tuple(l)
            if key not in seen:
                seen.add(key)
                seg = Segment(
                    x1=l[0], y1=l[1], x2=l[2], y2=l[3],
                    angle=_segment_angle(*l),
                    length=_segment_length(*l),
                )
                # Only keep short segments (long ones already captured in pass 1)
                if seg.length < min_len_long:
                    segments.append(seg)

    return segments


# ── Angle clustering ─────────────────────────────────────────────────────────

def _angle_diff(a1: float, a2: float) -> float:
    """Unsigned angular distance in [0, 90] degrees."""
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def cluster_by_angle(
    segments: list[Segment],
    threshold: float = 20.0,
) -> tuple[list[Segment], list[Segment], float, float]:
    """Split segments into two angle groups.

    Uses a robust two-step approach:
      1. Find the dominant angle from the longest segments (yard lines).
      2. Find the secondary angle from the remaining longest segments.
         In perspective views, the two groups may NOT be exactly 90° apart
         in image space, so we find the actual secondary angle rather than
         assuming perpendicularity.

    Returns (group_dominant, group_secondary, dominant_angle, secondary_angle).
    """
    if not segments:
        return [], [], 0.0, 90.0

    angles = np.array([s.angle for s in segments])
    lengths = np.array([s.length for s in segments])

    # Use the top 25% longest segments to determine dominant angle
    length_cutoff = np.percentile(lengths, 75)
    long_mask = lengths >= length_cutoff
    if long_mask.sum() < 2:
        long_mask = np.ones(len(segments), dtype=bool)

    # Weighted circular mean of the long segments
    long_angles = angles[long_mask]
    long_lengths = lengths[long_mask]

    rad2 = np.deg2rad(long_angles * 2)
    wx = np.sum(long_lengths * np.cos(rad2))
    wy = np.sum(long_lengths * np.sin(rad2))
    dominant_angle = (np.degrees(np.arctan2(wy, wx)) / 2) % 180

    # Assign dominant group
    group_dom = []
    remaining = []
    for seg in segments:
        d_dom = _angle_diff(seg.angle, dominant_angle)
        if d_dom <= threshold:
            seg.group = 0
            group_dom.append(seg)
        else:
            remaining.append(seg)

    # Find secondary angle from the remaining segments (not assumed perpendicular)
    if remaining:
        rem_angles = np.array([s.angle for s in remaining])
        rem_lengths = np.array([s.length for s in remaining])

        # Weight by length to find secondary direction
        rem_cutoff = np.percentile(rem_lengths, 50) if len(remaining) > 3 else 0
        rem_long = rem_lengths >= rem_cutoff
        if rem_long.sum() < 2:
            rem_long = np.ones(len(remaining), dtype=bool)

        rad2_r = np.deg2rad(rem_angles[rem_long] * 2)
        wts = rem_lengths[rem_long]
        wx2 = np.sum(wts * np.cos(rad2_r))
        wy2 = np.sum(wts * np.sin(rad2_r))
        secondary_angle = (np.degrees(np.arctan2(wy2, wx2)) / 2) % 180
    else:
        secondary_angle = (dominant_angle + 90) % 180

    # Assign secondary group
    group_sec = []
    for seg in remaining:
        d_sec = _angle_diff(seg.angle, secondary_angle)
        if d_sec <= threshold:
            seg.group = 1
            group_sec.append(seg)
        # else: stays unclassified (group = -1)

    return group_dom, group_sec, dominant_angle, secondary_angle


# ── Parallel line clustering ─────────────────────────────────────────────────

def _perp_position(seg: Segment, ref_angle: float) -> float:
    """Project segment midpoint onto the axis perpendicular to ref_angle."""
    mx, my = seg.midpoint
    # Perpendicular direction to the reference angle
    perp_rad = np.deg2rad(ref_angle + 90)
    return mx * np.cos(perp_rad) + my * np.sin(perp_rad)


def cluster_parallel_segments(
    segments: list[Segment],
    ref_angle: float,
    min_gap: float,
) -> list[LineCluster]:
    """Cluster parallel segments by perpendicular distance.

    Returns one LineCluster per distinct line, sorted by position.
    """
    if not segments:
        return []

    # Compute perpendicular positions
    positions = [(_perp_position(s, ref_angle), s) for s in segments]
    positions.sort(key=lambda x: x[0])

    clusters: list[list[Segment]] = []
    cur_cluster = [positions[0][1]]
    cur_pos = positions[0][0]

    for pos, seg in positions[1:]:
        if pos - cur_pos < min_gap:
            cur_cluster.append(seg)
        else:
            clusters.append(cur_cluster)
            cur_cluster = [seg]
        cur_pos = pos

    clusters.append(cur_cluster)

    # Build LineCluster objects — representative = longest segment
    result = []
    for cluster_segs in clusters:
        lengths = [s.length for s in cluster_segs]
        best = cluster_segs[np.argmax(lengths)]
        avg_pos = np.mean([_perp_position(s, ref_angle) for s in cluster_segs])
        result.append(LineCluster(
            segments=cluster_segs,
            label="",
            position=avg_pos,
            representative=best.as_array(),
        ))

    return result


# ── Classification ───────────────────────────────────────────────────────────

def classify_markings(
    group_dom: list[Segment],
    group_perp: list[Segment],
    dominant_angle: float,
    perp_angle: float,
    img_h: int,
    img_w: int,
) -> tuple[list[LineCluster], list[LineCluster], list[Segment]]:
    """Classify detected segments into yard lines, sidelines, and hash marks.

    Logic:
      - Dominant-angle group: long segments → yard lines, short → hash marks
        (hash marks are parallel to yard lines, just much shorter: 2ft vs 160ft)
      - Perpendicular group: long segments → sidelines
      - Length thresholds are adaptive to image size

    Returns (yard_lines, sidelines, hash_marks).
    """
    img_diag = np.hypot(img_h, img_w)

    # Threshold for "long" vs "short" — a yard line should span a meaningful
    # fraction of the image; hash marks are very short (2 feet physical)
    long_thresh = img_diag * 0.06  # ~6% of diagonal

    # ── Yard lines: long dominant-direction segments ──
    long_dom = [s for s in group_dom if s.length >= long_thresh]
    min_gap = max(15, img_h // 30)
    yard_lines = cluster_parallel_segments(long_dom, dominant_angle, min_gap)
    for yl in yard_lines:
        yl.label = "yard_line"
        for s in yl.segments:
            s.label = "yard_line"
            s.confidence = min(1.0, s.length / (img_diag * 0.15))

    # ── Sidelines: long perpendicular segments ──
    long_perp = [s for s in group_perp if s.length >= long_thresh]
    sidelines = cluster_parallel_segments(long_perp, perp_angle, min_gap)
    for sl in sidelines:
        sl.label = "sideline"
        for s in sl.segments:
            s.label = "sideline"
            s.confidence = min(1.0, s.length / (img_diag * 0.15))

    # ── Hash marks: short dominant-direction segments ──
    # Hash marks are physically parallel to yard lines (both run across the
    # field width), just much shorter (2 feet on a 160-foot wide field).
    # Perspective makes near-camera hash marks appear longer than distant ones,
    # so the max length is generous — up to half the yard line threshold.
    # The validation step (grid + row clustering) handles false positives.
    hash_max_len = long_thresh * 0.85  # anything under ~85% of yard line threshold
    hash_min_len = max(8, img_h // 80)  # minimum to avoid single-pixel noise
    hash_marks = []
    for s in group_dom:
        if hash_min_len <= s.length < hash_max_len and s.label == "":
            s.label = "hash_mark"
            s.confidence = 0.4
            hash_marks.append(s)

    # Short perpendicular segments are likely noise, yard number fragments,
    # or other artifacts — not hash marks
    for s in group_perp:
        if s.label == "":
            s.label = "perp_fragment"
            s.confidence = 0.2

    return yard_lines, sidelines, hash_marks


# ── Geometric validation ─────────────────────────────────────────────────────

def _merge_near_duplicates(
    yard_lines: list[LineCluster],
    min_gap_fraction: float = 0.35,
) -> list[LineCluster]:
    """Merge yard line clusters that are too close together.

    When two clusters are closer than min_gap_fraction * median_gap,
    they're likely the same physical line split by the clustering step.
    Keeps the one with more/longer segments.
    """
    if len(yard_lines) < 2:
        return yard_lines

    positions = [yl.position for yl in yard_lines]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not gaps:
        return yard_lines

    median_gap = np.median(gaps)
    min_gap = median_gap * min_gap_fraction

    merged = [yard_lines[0]]
    for i in range(1, len(yard_lines)):
        gap = positions[i] - positions[i - 1]
        if gap < min_gap:
            # Merge: keep the one with more total segment length
            prev = merged[-1]
            curr = yard_lines[i]
            prev_len = sum(s.length for s in prev.segments)
            curr_len = sum(s.length for s in curr.segments)
            if curr_len > prev_len:
                merged[-1] = curr
            # else: keep prev, skip curr
        else:
            merged.append(yard_lines[i])

    return merged


def _score_gap(positions: list[float], gap: float, tolerance: float) -> tuple[int, float, list[int]]:
    """Score how well a gap fits the positions. Returns (score, total_residual, indices)."""
    n = len(positions)
    anchor = positions[0]
    slot_best = {}  # grid_k → (line_idx, residual)

    for li, p in enumerate(positions):
        offset = (p - anchor) / gap
        nearest_k = round(offset)
        residual_px = abs(offset - nearest_k) * gap
        # Perspective tolerance: allow more slack for larger k (further from anchor)
        local_tol = tolerance * (1 + 0.3 * abs(nearest_k))
        if residual_px < local_tol:
            if nearest_k not in slot_best or residual_px < slot_best[nearest_k][1]:
                slot_best[nearest_k] = (li, residual_px)

    indices = [-1] * n
    for gk, (li, _) in slot_best.items():
        indices[li] = gk

    score = sum(1 for idx in indices if idx >= 0)
    total_res = sum(res for _, res in slot_best.values())
    return score, total_res, indices


def _find_best_spacing(
    positions: list[float],
    tolerance: float,
) -> tuple[float, list[int]]:
    """Find the gap size that best explains the detected positions.

    Tries every pair of positions as a candidate single-step gap,
    then checks how many other positions fall on that grid.
    Returns (best_gap, grid_indices) where grid_indices[i] is the
    grid slot for positions[i], or -1 if it doesn't fit.

    Prefers larger gaps when a smaller gap and its multiple fit similar
    numbers of lines — this avoids picking half-sized gaps that happen
    to fit all points but don't match the 5-yard physical spacing.
    """
    n = len(positions)
    if n < 2:
        return 0.0, list(range(n))

    # Collect all pairwise gaps divided by 1, 2, 3... as candidate unit gaps
    candidate_gaps = set()
    for i in range(n):
        for j in range(i + 1, n):
            raw = positions[j] - positions[i]
            for k in range(1, 6):
                g = raw / k
                if g > tolerance * 2:
                    candidate_gaps.add(g)

    # Score each candidate gap
    results = []  # (score, total_res, gap, indices)
    for gap in candidate_gaps:
        score, total_res, indices = _score_gap(positions, gap, tolerance)
        results.append((score, total_res, gap, indices))

    if not results:
        return 0.0, [-1] * n

    # Sort by score desc, then residual asc
    results.sort(key=lambda r: (-r[0], r[1]))
    best_score = results[0][0]
    best = results[0]

    # Check if a larger gap (roughly 2x the best) fits the same number of lines.
    # If so, prefer the larger gap — the smaller one likely subdivides
    # the real 5-yard interval. Only upgrade if we don't lose ANY lines.
    for r in results:
        if r[0] < best_score:
            continue
        # Is this gap roughly a multiple of the current best gap?
        ratio = r[2] / best[2] if best[2] > 0 else 0
        if ratio > 1.5 and r[0] == best_score:
            best = r

    return best[2], best[3]


def validate_yard_line_spacing(
    yard_lines: list[LineCluster],
    img_h: int,
) -> list[LineCluster]:
    """Validate yard lines using the equal-spacing constraint.

    Steps:
      1. Merge near-duplicate clusters (same physical line split by noise)
      2. Find the gap size that best fits the positions
      3. Reject lines that don't fit the grid

    Returns validated yard lines sorted by position.
    """
    if len(yard_lines) < 2:
        return yard_lines

    # Step 1: Merge near-duplicates
    yard_lines = _merge_near_duplicates(yard_lines)

    if len(yard_lines) < 2:
        return yard_lines

    # Step 2: Find best grid
    positions = [yl.position for yl in yard_lines]
    tolerance = max(15, img_h / 30)

    gap, grid_indices = _find_best_spacing(positions, tolerance)

    if gap < tolerance:
        return yard_lines

    # Step 3: Keep only lines on the grid
    valid = [yl for yl, idx in zip(yard_lines, grid_indices) if idx >= 0]

    # Step 4: If any consecutive pair has gap < 0.6 * median_gap, one is likely
    # a false positive. Try removing each candidate and pick the removal that
    # yields the best grid fit (all gaps are closer to integer multiples of
    # the base gap).
    changed = True
    while changed and len(valid) > 2:
        changed = False
        v_positions = [yl.position for yl in valid]
        v_gaps = [v_positions[i + 1] - v_positions[i] for i in range(len(v_positions) - 1)]
        median_gap = np.median(v_gaps)

        for i in range(len(v_gaps)):
            if v_gaps[i] < median_gap * 0.6:
                # Try removing index i vs i+1, keep whichever gives better grid fit
                def _grid_residual(lines):
                    if len(lines) < 2:
                        return float('inf')
                    ps = [yl.position for yl in lines]
                    _, total_res, _ = _score_gap(ps, tolerance * 2, tolerance)
                    # Try the minimum gap as the base
                    gs = [ps[j+1] - ps[j] for j in range(len(ps) - 1)]
                    base = min(gs)
                    if base <= 0:
                        return float('inf')
                    # How well do all gaps fit as integer multiples of base?
                    res = sum(abs(g / base - round(g / base)) for g in gs)
                    return res

                option_a = [yl for j, yl in enumerate(valid) if j != i]
                option_b = [yl for j, yl in enumerate(valid) if j != i + 1]
                res_a = _grid_residual(option_a)
                res_b = _grid_residual(option_b)
                valid = option_a if res_a < res_b else option_b
                changed = True
                break

    return valid


def predict_missing_yard_lines(
    yard_lines: list[LineCluster],
    white_mask: np.ndarray,
    dominant_angle: float,
    img_shape: tuple,
    min_evidence: float = 0.15,
) -> list[LineCluster]:
    """Predict and fill in missing yard lines using the equal-spacing constraint.

    Given 2+ validated yard lines, predicts where missing lines should be
    and scans the white mask for evidence at those positions. If enough white
    pixels exist along the predicted line, creates a synthetic LineCluster.

    Args:
        yard_lines: validated yard lines (at least 2)
        white_mask: clean white pixel mask
        dominant_angle: angle of yard lines in degrees
        img_shape: (height, width, ...)
        min_evidence: minimum fraction of white pixels along predicted line
                      to accept it (0.0-1.0)

    Returns:
        Extended list of yard lines including predicted ones.
    """
    if len(yard_lines) < 2:
        return yard_lines

    h, w = img_shape[:2]
    positions = [yl.position for yl in yard_lines]
    tolerance = max(10, h / 40)

    gap, grid_indices = _find_best_spacing(positions, tolerance)
    if gap < tolerance:
        return yard_lines

    # Find grid range: what slots do existing lines occupy?
    valid_ks = [idx for idx in grid_indices if idx >= 0]
    if not valid_ks:
        return yard_lines

    k_min = min(valid_ks)
    k_max = max(valid_ks)

    # Compute anchor: the perpendicular position of the first line minus its grid index * gap
    # Find the first line that has a valid grid index
    anchor = positions[0]  # default
    for li, idx in enumerate(grid_indices):
        if idx >= 0:
            anchor = positions[li] - idx * gap
            break

    # Existing grid slots
    existing_ks = set(valid_ks)

    # Predict missing lines between (and one beyond) existing lines
    result = list(yard_lines)
    angle_rad = np.deg2rad(dominant_angle)
    perp_rad = np.deg2rad(dominant_angle + 90)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cos_p, sin_p = np.cos(perp_rad), np.sin(perp_rad)

    for k in range(k_min - 1, k_max + 2):
        if k in existing_ks:
            continue

        pred_pos = anchor + k * gap

        # Convert perpendicular position to a point on the line
        # The line passes through (cx, cy) with direction (cos_a, sin_a)
        # where (cx, cy) is the point at perpendicular distance pred_pos
        cx = pred_pos * cos_p
        cy = pred_pos * sin_p

        # Extend this line across the image
        t_range = max(h, w) * 2
        x1 = int(cx - t_range * cos_a)
        y1 = int(cy - t_range * sin_a)
        x2 = int(cx + t_range * cos_a)
        y2 = int(cy + t_range * sin_a)

        # Sample white mask along this line
        line_length = int(np.hypot(x2 - x1, y2 - y1))
        if line_length == 0:
            continue

        # Sample a band of pixels along the line (±band_width perpendicular)
        # to handle slight position errors in the grid prediction
        band_width = max(5, int(gap * 0.05))
        num_samples = min(line_length, 500)
        ts = np.linspace(0, 1, num_samples)
        center_xs = x1 + ts * (x2 - x1)
        center_ys = y1 + ts * (y2 - y1)

        best_evidence = 0.0
        best_offset = 0
        # Scan at different perpendicular offsets within the band
        for off in range(-band_width, band_width + 1):
            xs = (center_xs + off * cos_p).astype(int)
            ys = (center_ys + off * sin_p).astype(int)

            in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs_v = xs[in_bounds]
            ys_v = ys[in_bounds]

            if len(xs_v) < 10:
                continue

            white_count = np.sum(white_mask[ys_v, xs_v] > 0)
            ev = white_count / len(xs_v)
            if ev > best_evidence:
                best_evidence = ev
                best_offset = off

        # Use the best offset for the representative line
        xs_final = (center_xs + best_offset * cos_p).astype(int)
        ys_final = (center_ys + best_offset * sin_p).astype(int)
        in_bounds = (xs_final >= 0) & (xs_final < w) & (ys_final >= 0) & (ys_final < h)
        xs_valid = xs_final[in_bounds]
        ys_valid = ys_final[in_bounds]
        evidence = best_evidence

        if evidence >= min_evidence:
            # Create a synthetic LineCluster at this position
            # Use the in-bounds endpoints as the representative segment
            rep = np.array([xs_valid[0], ys_valid[0], xs_valid[-1], ys_valid[-1]])
            extended = _extend_line(rep, img_shape)
            cluster = LineCluster(
                segments=[],
                label="yard_line",
                position=pred_pos,
                representative=rep,
                extended=extended,
            )
            result.append(cluster)

    # Re-sort by position
    result.sort(key=lambda yl: yl.position)
    return result


def _lateral_position(seg: Segment, ref_angle: float) -> float:
    """Project segment midpoint onto the axis parallel to ref_angle.

    While _perp_position gives the "along the field" position (which yard),
    _lateral_position gives the "across the field" position (which row).
    """
    mx, my = seg.midpoint
    rad = np.deg2rad(ref_angle)
    return mx * np.cos(rad) + my * np.sin(rad)


def detect_hash_marks_cc(
    white_mask: np.ndarray,
    yard_lines: list[LineCluster],
    dominant_angle: float,
) -> list[Segment]:
    """Detect hash marks using connected components instead of Hough lines.

    Hash marks are tiny (2ft × 4in physically) and appear as small elongated
    blobs in the white mask — too small for reliable HoughLinesP detection.

    This approach:
      1. Find all connected components in the white mask
      2. Filter by area (small) and aspect ratio (elongated)
      3. Check orientation matches yard line angle
      4. Return as Segment objects for downstream validation
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8,
    )
    if num_labels < 2:
        return []

    img_h, img_w = white_mask.shape[:2]
    img_diag = np.hypot(img_h, img_w)

    # Hash mark size bounds — physically 2ft on a 160ft field = 1.25% of width.
    # In pixels this varies hugely with perspective, but we can set reasonable
    # bounds. Min area scales with image size to reject turf texture noise on
    # large images while still catching small marks on small images.
    max_area = img_h * img_w * 0.005
    min_area = max(8, int(img_diag * 0.012))  # ~8px for 600px diag, ~33 for 2800px
    max_dim = img_diag * 0.05

    # Yard line angle in radians (the direction hash marks are elongated)
    dom_rad = np.deg2rad(dominant_angle)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        max_side = max(bw, bh)
        min_side = max(1, min(bw, bh))

        if max_side > max_dim:
            continue

        # Must be somewhat elongated (aspect ratio > 1.5)
        aspect = max_side / min_side
        if aspect < 1.5:
            continue

        # Check orientation: the long axis should align with yard line angle.
        # For near-vertical yard lines (angle ~85-95°), hash marks should be
        # taller than wide. For more angled lines, use the bounding box ratio.
        # Dominant angle ~90° means vertical → height > width expected.
        # Dominant angle ~0° means horizontal → width > height expected.
        if dominant_angle > 45:
            # Yard lines are more vertical; hash marks should be taller
            if bh < bw * 1.2:
                continue
        else:
            # Yard lines are more horizontal; hash marks should be wider
            if bw < bh * 1.2:
                continue

        cx, cy = centroids[i]
        # Create a segment along the long axis of the component
        half_len = max_side / 2.0
        dx = np.cos(dom_rad) * half_len
        dy = np.sin(dom_rad) * half_len
        seg = Segment(
            x1=int(cx - dx), y1=int(cy - dy),
            x2=int(cx + dx), y2=int(cy + dy),
            angle=dominant_angle,
            length=float(max_side),
            label="hash_mark",
            confidence=0.4,
            group=0,
        )
        candidates.append(seg)

    return candidates


def _filter_row_by_spacing(
    marks: list[Segment],
    dominant_angle: float,
    one_yd_gap: float,
) -> list[Segment]:
    """Keep only marks in a row that fit a regular 1-yard spacing pattern.

    Sorts marks by perpendicular position (along the field), then checks
    that consecutive marks are spaced at approximately integer multiples of
    the 1-yard gap. Marks that break the spacing pattern are removed.
    """
    if len(marks) < 2 or one_yd_gap < 1:
        return marks

    # Sort by perp position
    perps = [(m, _perp_position(m, dominant_angle)) for m in marks]
    perps.sort(key=lambda x: x[1])

    tolerance = one_yd_gap * 0.35  # 35% tolerance for perspective distortion

    # Score each mark: how many neighbors are at ~integer × 1-yard spacing?
    scores = []
    for i, (m, p) in enumerate(perps):
        score = 0
        for j, (m2, p2) in enumerate(perps):
            if i == j:
                continue
            dist = abs(p2 - p)
            # Check if distance is close to an integer multiple of 1-yard gap
            ratio = dist / one_yd_gap
            nearest_int = round(ratio)
            if nearest_int >= 1 and abs(ratio - nearest_int) * one_yd_gap < tolerance:
                score += 1
        scores.append(score)

    # Keep marks with at least 2 correctly-spaced neighbors
    min_neighbors = 2 if len(perps) >= 4 else 1
    filtered = [perps[i][0] for i in range(len(perps)) if scores[i] >= min_neighbors]

    # Size consistency: hash marks in the same row should be similar sizes
    # (they're at similar distances from the camera). Remove outliers.
    if len(filtered) >= 3:
        lengths = np.array([m.length for m in filtered])
        med_len = float(np.median(lengths))
        # Keep marks within 40% of median length
        filtered = [m for m, l in zip(filtered, lengths)
                    if 0.6 * med_len <= l <= 1.4 * med_len]

    return filtered


def validate_hash_marks(
    hash_marks: list[Segment],
    yard_lines: list[LineCluster],
    dominant_angle: float,
    img_h: int,
) -> tuple[list[Segment], list["HashMarkRow"]]:
    """Validate hash marks using physical constraints.

    Physical model (college/NFL):
      - Two rows of inbound hash marks in the interior of the field
        College: 40ft from each sideline (60ft apart)
        NFL: 70ft 9in from each sideline (18ft 6in apart)
      - Each mark is parallel to yard lines, 24in long × 4in wide
      - Marks at 1-yard intervals BETWEEN the 5-yard lines
        (4 marks per row between consecutive yard lines)
      - Hash marks do NOT sit on the yard lines themselves

    Validation:
      1. Must be between yard lines (not on or outside them)
      2. Must be in the interior of the field (not near sidelines)
      3. Cluster into lateral rows
      4. Within each row, check 1-yard spacing consistency
      5. Keep at most 2 best rows

    Returns (validated_hash_marks, hash_rows).
    """
    if not yard_lines or not hash_marks or len(yard_lines) < 2:
        return [], []

    yl_positions = sorted([yl.position for yl in yard_lines])
    five_yd_gap = np.median([
        yl_positions[i + 1] - yl_positions[i]
        for i in range(len(yl_positions) - 1)
    ])
    one_yd_gap = five_yd_gap / 5.0

    # ── Step 1: Must be BETWEEN yard lines, not on them ──
    yl_margin = five_yd_gap * 0.10
    pos_min = yl_positions[0] - five_yd_gap
    pos_max = yl_positions[-1] + five_yd_gap

    between = []
    for hm in hash_marks:
        pos = _perp_position(hm, dominant_angle)
        if pos < pos_min or pos > pos_max:
            continue
        min_dist_yl = min(abs(pos - ylp) for ylp in yl_positions)
        if min_dist_yl < yl_margin:
            continue
        between.append(hm)

    if not between:
        return [], []

    # ── Step 2: Compute lateral field extent ──
    # We don't filter by lateral position here — the density-based row
    # clustering in step 3 handles noise rejection. Hash marks can be
    # anywhere between the sidelines, and yard lines may not extend to
    # the full field width in the camera view.
    all_yl_lats = []
    for yl in yard_lines:
        for s in yl.segments:
            all_yl_lats.append(_lateral_position(s, dominant_angle))
    lat_min = min(all_yl_lats)
    lat_max = max(all_yl_lats)
    lat_span = lat_max - lat_min
    if lat_span < 1:
        return [], []

    middle = between  # pass all between-yard-line marks to clustering

    if len(middle) < 2:
        return middle, []

    # ── Step 3: Find 2 lateral rows using density peaks ──
    # Real hash mark rows are dense clusters at specific lateral positions.
    # Random noise is scattered uniformly. Use a histogram to find the
    # densest lateral bands, then assign marks to the nearest peak.
    laterals = np.array([_lateral_position(hm, dominant_angle) for hm in middle])

    # Bin width: ~5% of the 5-yard gap (roughly 1 yard of lateral field)
    bin_width = max(five_yd_gap * 0.05, 5)
    lat_lo = laterals.min() - bin_width
    lat_hi = laterals.max() + bin_width
    n_bins = max(5, int((lat_hi - lat_lo) / bin_width))
    hist, bin_edges = np.histogram(laterals, bins=n_bins, range=(lat_lo, lat_hi))

    # Find the top 2 peak bins (with minimum separation)
    min_peak_sep = max(3, n_bins // 5)  # peaks must be at least this far apart
    peaks = []
    for _ in range(2):
        if hist.max() < 2:
            break
        peak_idx = int(np.argmax(hist))
        peak_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        peaks.append((peak_idx, peak_center, hist[peak_idx]))
        # Suppress this peak and neighbors
        lo = max(0, peak_idx - min_peak_sep)
        hi = min(len(hist), peak_idx + min_peak_sep + 1)
        hist[lo:hi] = 0

    if not peaks:
        return middle, []

    # Assign marks to nearest peak (within 2× bin_width tolerance)
    row_tolerance = bin_width * 3
    row_marks = {i: [] for i in range(len(peaks))}
    for hm, lat in zip(middle, laterals):
        best_dist = float("inf")
        best_peak = -1
        for pi, (_, center, _) in enumerate(peaks):
            d = abs(lat - center)
            if d < best_dist and d < row_tolerance:
                best_dist = d
                best_peak = pi
        if best_peak >= 0:
            row_marks[best_peak].append(hm)

    # ── Step 4: Validate each row with spacing consistency ──
    hash_rows = []
    for pi in range(len(peaks)):
        marks = row_marks[pi]
        if len(marks) < 2:
            continue
        avg_lat = float(np.mean([_lateral_position(m, dominant_angle) for m in marks]))

        # Filter by 1-yard spacing
        filtered = _filter_row_by_spacing(marks, dominant_angle, one_yd_gap)

        if len(filtered) >= 3:
            for hm in filtered:
                hm.confidence = min(1.0, 0.6 + 0.1 * len(filtered))
            hash_rows.append(HashMarkRow(
                marks=filtered,
                lateral_position=avg_lat,
            ))

    # Collect all validated marks
    validated = []
    for r in hash_rows:
        validated.extend(r.marks)

    # Label rows
    if hash_rows:
        hash_rows.sort(key=lambda r: r.lateral_position)
        if len(hash_rows) == 2:
            hash_rows[0].label = "near_inbound"
            hash_rows[1].label = "far_inbound"
        else:
            hash_rows[0].label = "inbound"

    return validated, hash_rows


# ── Extend lines to full image span ──────────────────────────────────────────

def _extend_line(line: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Extend a line segment to span the full image."""
    x1, y1, x2, y2 = line.astype(float)
    h, w = img_shape[:2]

    if abs(x2 - x1) < 1:
        return np.array([int(x1), 0, int(x2), h - 1])

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    pts = []
    y_at_0 = intercept
    if 0 <= y_at_0 < h:
        pts.append((0, y_at_0))
    y_at_w = slope * (w - 1) + intercept
    if 0 <= y_at_w < h:
        pts.append((w - 1, y_at_w))
    if abs(slope) > 1e-6:
        x_at_0 = -intercept / slope
        if 0 <= x_at_0 < w:
            pts.append((x_at_0, 0))
        x_at_h = (h - 1 - intercept) / slope
        if 0 <= x_at_h < w:
            pts.append((x_at_h, h - 1))

    if len(pts) < 2:
        return line

    pts = np.array(pts)
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return np.array([int(pts[i][0]), int(pts[i][1]), int(pts[j][0]), int(pts[j][1])])


# ── Main detection pipeline ──────────────────────────────────────────────────

def detect_field_markings(
    image: np.ndarray,
    field_mask: np.ndarray | None = None,
    player_mask: np.ndarray | None = None,
) -> FieldMarkings:
    """Full pipeline: detect and classify all field markings.

    Args:
        image: BGR input image
        field_mask: optional pre-computed green field mask
        player_mask: optional pre-computed player bounding box mask

    Returns:
        FieldMarkings with classified yard lines, sidelines, hash marks
    """
    h, w = image.shape[:2]

    # Step 1: Clean white mask
    if field_mask is None:
        field_mask = build_field_mask(image)
    if player_mask is None:
        player_mask = _build_player_mask(image)

    white_mask = detect_white_mask_clean(image, field_mask, player_mask)

    # Step 2: Detect segments (two-pass Hough)
    segments = detect_segments(white_mask, h)

    if not segments:
        return FieldMarkings(
            yard_lines=[], sidelines=[], hash_marks=[], hash_rows=[],
            all_segments=[], white_mask=white_mask,
            dominant_angle=0, perp_angle=90,
        )

    # Step 3: Cluster by angle into two perpendicular groups
    group_dom, group_perp, dom_angle, perp_angle = cluster_by_angle(segments)

    # Step 4: Classify into yard lines, sidelines, hash marks
    yard_lines, sidelines, hash_marks = classify_markings(
        group_dom, group_perp, dom_angle, perp_angle, h, w,
    )

    # Step 5: Validate with geometric constraints (reject false positives)
    yard_lines = validate_yard_line_spacing(yard_lines, h)

    # Step 6: Predict missing yard lines using equal-spacing constraint
    yard_lines = predict_missing_yard_lines(
        yard_lines, white_mask, dom_angle, image.shape,
    )

    # Step 6b: Detect hash marks via connected components.
    # Use a local contrast mask (gray - blurred background) which catches
    # dim/worn hash marks that fail global HSV thresholds. Also run on the
    # strict white mask. Merge all candidates with deduplication.
    lc_mask = detect_local_contrast_mask(image, field_mask, player_mask)
    cc_hash_lc = detect_hash_marks_cc(lc_mask, yard_lines, dom_angle)
    cc_hash_strict = detect_hash_marks_cc(white_mask, yard_lines, dom_angle)
    # Merge CC passes + Hough candidates, deduplicating by proximity
    all_candidates = list(hash_marks)
    for cc in cc_hash_lc + cc_hash_strict:
        ccx, ccy = cc.midpoint
        duplicate = False
        for existing in all_candidates:
            hx, hy = existing.midpoint
            if abs(ccx - hx) < 10 and abs(ccy - hy) < 10:
                duplicate = True
                break
        if not duplicate:
            all_candidates.append(cc)
    hash_marks = all_candidates

    hash_marks, hash_rows = validate_hash_marks(hash_marks, yard_lines, dom_angle, h)

    # Step 7: Extend yard lines and sidelines to full image span
    for yl in yard_lines:
        if len(yl.extended) != 4:
            yl.extended = _extend_line(yl.representative, image.shape)
    for sl in sidelines:
        sl.extended = _extend_line(sl.representative, image.shape)

    return FieldMarkings(
        yard_lines=yard_lines,
        sidelines=sidelines,
        hash_marks=hash_marks,
        hash_rows=hash_rows,
        all_segments=segments,
        white_mask=white_mask,
        dominant_angle=dom_angle,
        perp_angle=perp_angle,
    )


# ── Debug visualization ──────────────────────────────────────────────────────

COLORS = {
    "yard_line": (0, 255, 0),       # green
    "sideline": (255, 100, 0),      # blue
    "hash_mark": (0, 255, 255),     # yellow
    "short_fragment": (0, 180, 255),# orange
    "perp_fragment": (180, 130, 0), # teal
    "unknown": (128, 128, 128),     # gray
}


def draw_markings_debug(
    image: np.ndarray,
    markings: FieldMarkings,
) -> np.ndarray:
    """Draw all classified markings on the image for debugging.

    - Green: yard lines (extended)
    - Blue: sidelines (extended)
    - Yellow: hash marks
    - Orange: tick marks
    - Gray: unclassified segments
    """
    debug = image.copy()
    h, w = image.shape[:2]

    # Draw all raw segments lightly first
    for seg in markings.all_segments:
        color = COLORS.get(seg.label, COLORS["unknown"])
        alpha_color = tuple(c // 3 for c in color)
        cv2.line(debug, (seg.x1, seg.y1), (seg.x2, seg.y2), alpha_color, 1)

    # Draw extended yard lines prominently
    yl_colors = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
        (0, 128, 255), (255, 128, 0),
    ]
    for i, yl in enumerate(markings.yard_lines):
        color = yl_colors[i % len(yl_colors)]
        ex = yl.extended
        if len(ex) == 4:
            cv2.line(debug, (ex[0], ex[1]), (ex[2], ex[3]), color, 3)
            mx = (ex[0] + ex[2]) // 2
            my = (ex[1] + ex[3]) // 2
            cv2.putText(debug, f"YL{i}", (mx - 15, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw extended sidelines
    for i, sl in enumerate(markings.sidelines):
        ex = sl.extended
        if len(ex) == 4:
            cv2.line(debug, (ex[0], ex[1]), (ex[2], ex[3]), (255, 100, 0), 3)
            mx = (ex[0] + ex[2]) // 2
            my = (ex[1] + ex[3]) // 2
            cv2.putText(debug, f"SL{i}", (mx + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    # Draw hash mark row lines — a line fit through all marks in each row
    row_colors = [
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (0, 200, 200),   # dark yellow
        (200, 0, 200),   # dark magenta
    ]
    for ri, row in enumerate(markings.hash_rows):
        color = row_colors[ri % len(row_colors)]
        # Collect all endpoints from marks in this row
        pts = []
        for hm in row.marks:
            pts.append([hm.x1, hm.y1])
            pts.append([hm.x2, hm.y2])
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.float32)
            # Fit a line through the mark endpoints using the perpendicular
            # angle (sideline direction), since the row runs along the field
            fit = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = float(fit[0][0]), float(fit[1][0]), float(fit[2][0]), float(fit[3][0])
            # Extend to span the marks with some padding
            t_vals = [((p[0] - x0) * vx + (p[1] - y0) * vy) for p in pts]
            t_min, t_max = min(t_vals), max(t_vals)
            pad = (t_max - t_min) * 0.1
            lx1 = int(x0 + (t_min - pad) * vx)
            ly1 = int(y0 + (t_min - pad) * vy)
            lx2 = int(x0 + (t_max + pad) * vx)
            ly2 = int(y0 + (t_max + pad) * vy)
            cv2.line(debug, (lx1, ly1), (lx2, ly2), color, 2, cv2.LINE_AA)
            # Label the row
            label = row.label or f"row{ri}"
            cv2.putText(debug, f"HM:{label}({len(row.marks)})",
                        (lx1 + 5, ly1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw individual hash marks on top
    for hm in markings.hash_marks:
        # Thicker marks with confidence-based brightness
        bright = int(155 + 100 * hm.confidence)
        color = (0, bright, bright)
        cv2.line(debug, (hm.x1, hm.y1), (hm.x2, hm.y2), color, 3)

    # Summary text
    summary = (
        f"YL:{len(markings.yard_lines)} SL:{len(markings.sidelines)} "
        f"HM:{len(markings.hash_marks)} rows:{len(markings.hash_rows)} "
        f"dom={markings.dominant_angle:.0f}deg"
    )
    cv2.putText(debug, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return debug


def draw_markings_panel(
    image: np.ndarray,
    markings: FieldMarkings,
) -> np.ndarray:
    """Create a 2x2 debug panel showing pipeline stages.

    Top-left:  Clean white mask
    Top-right: All detected segments (color by angle group)
    Bot-left:  Classified markings
    Bot-right: Extended lines + hash marks
    """
    h, w = image.shape[:2]
    ph, pw = h // 2, w // 2

    def _resize(img):
        return cv2.resize(img, (pw, ph))

    # Panel 1: White mask
    mask_bgr = cv2.cvtColor(markings.white_mask, cv2.COLOR_GRAY2BGR)
    p1 = _resize(mask_bgr)
    cv2.putText(p1, "Clean white mask", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Panel 2: All segments by angle group
    p2_full = image.copy()
    for seg in markings.all_segments:
        if seg.group == 0:
            color = (0, 255, 0)   # dominant = green
        elif seg.group == 1:
            color = (255, 0, 0)   # perpendicular = blue
        else:
            color = (0, 0, 255)   # unclassified = red
        cv2.line(p2_full, (seg.x1, seg.y1), (seg.x2, seg.y2), color, 2)
    p2 = _resize(p2_full)
    cv2.putText(p2, f"Segments: green=dom({markings.dominant_angle:.0f}d) blue=perp", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Panel 3: Classified markings
    p3_full = image.copy()
    for seg in markings.all_segments:
        color = COLORS.get(seg.label, COLORS["unknown"])
        thickness = 2 if seg.label in ("yard_line", "sideline") else 1
        cv2.line(p3_full, (seg.x1, seg.y1), (seg.x2, seg.y2), color, thickness)
    p3 = _resize(p3_full)
    cv2.putText(p3, "Classified: grn=YL blue=SL yel=HM", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Panel 4: Final extended lines + hash marks
    p4_full = draw_markings_debug(image, markings)
    p4 = _resize(p4_full)
    cv2.putText(p4, "Extended lines + hash marks", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Assemble 2x2
    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    panel = np.vstack([top, bot])

    return panel


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Detect field markings")
    parser.add_argument("images", nargs="*", default=["steelers.jpg", "texas.jpg"])
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for img_path in args.images:
        print(f"\n{'='*60}")
        print(f"Processing {img_path}")
        print(f"{'='*60}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  Could not read {img_path}")
            continue

        markings = detect_field_markings(image)

        print(f"  Dominant angle: {markings.dominant_angle:.1f}°")
        print(f"  Yard lines: {len(markings.yard_lines)}")
        print(f"  Sidelines: {len(markings.sidelines)}")
        print(f"  Hash marks: {len(markings.hash_marks)}")
        print(f"  Total segments: {len(markings.all_segments)}")

        stem = Path(img_path).stem
        debug = draw_markings_debug(image, markings)
        cv2.imwrite(str(output_dir / f"{stem}_markings_debug.jpg"), debug)

        panel = draw_markings_panel(image, markings)
        cv2.imwrite(str(output_dir / f"{stem}_markings_panel.jpg"), panel)

        print(f"  Saved debug images to output/")
