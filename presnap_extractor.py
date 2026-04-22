"""
Pre-snap frame extractor.

Processes a football game video and automatically extracts one pre-snap
frame per play using a 3-layer detection approach:

  1. **Scene cuts** — histogram correlation to segment video into plays
  2. **Motion valleys** — find the stillest moment in each segment (pre-snap)
  3. **Line-is-set** — blob detection to confirm players are in formation

Optimised for speed: single sequential pass through the video with
downscaled analysis frames. A 41-minute game should finish in ~2-4 minutes.

Usage:
    results = extract_presnap_frames("game.mp4", "output/presnap/")
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


# ── Constants ───────────────────────────────────────────────────────────

# HSV range for green grass (same as detect_players.py)
GREEN_LOW = np.array([30, 40, 40])
GREEN_HIGH = np.array([80, 255, 255])

# Minimum contour area (pixels) to count as a player blob (at 360-480p)
MIN_PLAYER_AREA = 40
MAX_PLAYER_AREA = 25_000

# Minimum number of player-sized blobs for a valid formation
MIN_FORMATION_BLOBS = 6

# Analysis resolution — frames are downscaled to this height for speed
ANALYSIS_HEIGHT = 480


# ── View Classification ────────────────────────────────────────────────

# Lazy import of the hash-intersection model from the homography module.
# This avoids a hard dependency at module level so the extractor can
# still be imported even if interactive_homography isn't available.
_hash_predict_fn = None


def _get_hash_predict():
    """Return the hash-intersection predict function (lazy load)."""
    global _hash_predict_fn
    if _hash_predict_fn is None:
        from interactive_homography import _hash_intersection_predict
        _hash_predict_fn = _hash_intersection_predict
    return _hash_predict_fn


def _classify_view(frame_bgr: np.ndarray) -> str:
    """Classify a frame as 'sideline', 'endzone', or 'scoreboard'.

    Uses two signals:

      1. **Green coverage** — scoreboard / graphics have little green.
      2. **Hash-mark pair orientation** — runs the hash-yard-intersection
         YOLO model on the frame.  For each detection, its nearest
         neighbour is found; if the pair is separated more in x than y
         the pair is "horizontal" (sideline view), otherwise "vertical"
         (endzone view).  Sideline views show hash marks running
         left→right; endzone views show them running top→bottom.

    If the model returns fewer than 3 detections the frame is treated
    as non-sideline (conservative — if we can't see hash marks clearly
    it's probably not a clean pre-snap sideline shot).
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ── Green field mask ────────────────────────────────────────────
    green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    green_pct = np.count_nonzero(green_mask) / (h * w) * 100

    if green_pct < 35:
        return "scoreboard"

    rows = np.where(green_mask.any(axis=1))[0]
    if len(rows) > 10:
        row_cov = green_mask.sum(axis=1).astype(float) / 255.0 / w
        row_std = float(np.std(row_cov[rows[0] : rows[-1] + 1]))
    else:
        row_std = 0.0

    if green_pct < 55 and row_std < 0.25:
        return "scoreboard"

    # ── Hash-mark orientation via YOLO model ────────────────────────
    try:
        predict = _get_hash_predict()
        preds = predict(frame_bgr, confidence=15)
    except Exception:
        # If hash model unavailable, allow the frame through
        return "sideline"

    if len(preds) < 8:
        # Sideline views produce many clear hash detections (typically
        # 12-21).  Endzone views give far fewer (3-7) because hash marks
        # are foreshortened.  Requiring ≥ 8 detections cleanly separates
        # the two.  Frames below this threshold are treated as non-
        # sideline.
        return "endzone"

    centers = np.array([(p["x"], p["y"]) for p in preds])

    h_pairs = 0  # nearest-neighbour separated horizontally
    v_pairs = 0  # nearest-neighbour separated vertically
    for i in range(len(centers)):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = 1e9
        j = int(np.argmin(dists))
        dx = abs(centers[j][0] - centers[i][0])
        dy = abs(centers[j][1] - centers[i][1])
        if dx > dy:
            h_pairs += 1
        else:
            v_pairs += 1

    if h_pairs > v_pairs:
        return "sideline"
    return "endzone"


# ── Helpers ─────────────────────────────────────────────────────────────

def _downscale(frame: np.ndarray, target_h: int = ANALYSIS_HEIGHT) -> np.ndarray:
    """Downscale frame to target height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h <= target_h:
        return frame
    scale = target_h / h
    return cv2.resize(frame, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def _build_field_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Create binary mask where True = green field pixels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute normalised HSV histogram for scene-cut comparison."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def _frame_to_timestamp(frame_num: int, fps: float) -> str:
    """Convert frame number to MM:SS.mmm string."""
    if fps <= 0:
        return "0:00.000"
    secs = frame_num / fps
    mins = int(secs // 60)
    rem = secs % 60
    return f"{mins}:{rem:06.3f}"


def _score_formation(small_bgr: np.ndarray) -> tuple[float, int]:
    """Score how much a (downscaled) frame looks like a pre-snap formation.

    Returns (score, blob_count) where score is 0.0-1.0 indicating
    likelihood of a football formation.
    """
    h, w = small_bgr.shape[:2]

    # Use a TIGHT field mask (no dilation) so players aren't swallowed
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    field_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    # Light dilation to fill small grass gaps but NOT eat players
    kern_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    field_mask = cv2.dilate(field_mask, kern_sm, iterations=1)

    # Players = non-green pixels that are near the field area
    # Use a wider mask just for the "on field" check
    field_region = cv2.dilate(field_mask, kern_sm, iterations=3)
    inv_field = cv2.bitwise_not(field_mask)
    player_candidates = cv2.bitwise_and(inv_field, inv_field, mask=field_region)

    # Clean up noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    player_candidates = cv2.morphologyEx(player_candidates, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    player_candidates = cv2.morphologyEx(player_candidates, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        player_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter to player-sized blobs
    player_xs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_PLAYER_AREA <= area <= MAX_PLAYER_AREA:
            x, y, bw, bh = cv2.boundingRect(cnt)
            player_xs.append(x + bw // 2)

    blob_count = len(player_xs)
    if blob_count < 5:
        return 0.0, blob_count

    # Check horizontal spread
    field_cols = np.where(field_mask.any(axis=0))[0]
    if len(field_cols) < 10:
        field_left, field_right = 0, w
    else:
        field_left, field_right = int(field_cols[0]), int(field_cols[-1])

    field_width = max(1, field_right - field_left)
    xs = np.array(player_xs)
    blob_spread = (xs.max() - xs.min()) / field_width

    # Blob count: 0 at 4, 1.0 at 10+
    count_score = min(1.0, max(0.0, (blob_count - 4) / 6.0))
    # Spread: 0 at 0.10, 1.0 at 0.35+
    spread_score = min(1.0, max(0.0, (blob_spread - 0.10) / 0.25))

    return count_score * 0.5 + spread_score * 0.5, blob_count


# ── Single-Pass Extraction ──────────────────────────────────────────────

def extract_presnap_frames(
    video_path: str,
    output_dir: str,
    min_segment_duration: float = 2.0,
    scene_threshold: float = 0.65,
    progress_callback=None,
    sideline_only: bool = True,
) -> list[dict]:
    """Extract one pre-snap frame per play from a football game video.

    Uses a **single sequential read** through the video for speed:
      Pass 1 (fast): read every 3rd frame at 480p → detect scene cuts
                     + compute per-frame motion score
      Pass 2 (targeted): for each segment, pick the motion-valley frame,
                         seek only to that one frame for formation check
                         and full-res PNG save.

    Parameters
    ----------
    video_path : str
        Path to the game video file.
    output_dir : str
        Directory to save extracted PNG frames.
    min_segment_duration : float
        Ignore segments shorter than this (seconds).
    scene_threshold : float
        Histogram correlation threshold for scene cuts (0-1).
    progress_callback : callable or None
        fn(current_play: int, total_segments: int, status: str)

    Returns
    -------
    list of dicts with keys: play_number, frame_number, timestamp,
    output_path, segment, confidence
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Cannot read video metadata: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: fast sequential scan ────────────────────────────────
    # Sequential grab() is 10-50x faster than random seek on H.264.
    # We grab() every frame but only retrieve()+decode every Nth one.
    # N = sample_step ≈ fps/3 → ~3 decoded frames per second.
    sample_step = max(1, int(fps / 3))       # decode every ~10 frames
    min_cut_gap = int(fps)                    # 1 second between cuts

    if progress_callback:
        progress_callback(0, 0, "Pass 1: scanning video for scene cuts & motion...")

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []

    first_small = _downscale(first_frame)
    prev_hist = _hist(first_small)
    prev_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)
    prev_mask = _build_field_mask(first_small)
    prev_gray_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=prev_mask)

    # Storage: (frame_number, motion_value) for motion analysis later
    samples = [(0, 0.0)]   # first frame has 0 motion
    cut_frames = [0]        # scene boundaries

    frame_idx = 0
    read_count = 0

    while True:
        # grab() is cheap — just advances the codec without full decode
        grabbed = cap.grab()
        frame_idx += 1
        if not grabbed:
            break

        # Only fully decode every sample_step-th frame
        if frame_idx % sample_step != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        read_count += 1
        small = _downscale(frame)

        # Scene-cut check
        h = _hist(small)
        corr = cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)
        if corr < scene_threshold and frame_idx - cut_frames[-1] >= min_cut_gap:
            cut_frames.append(frame_idx)
        prev_hist = h

        # Motion measurement (every decoded frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = _build_field_mask(small)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        diff = cv2.absdiff(prev_gray_masked, gray_masked)
        combined = cv2.bitwise_or(prev_mask, mask)
        n_pixels = np.count_nonzero(combined)
        motion_val = float(np.sum(diff[combined > 0])) / max(n_pixels, 1)

        samples.append((frame_idx, motion_val))
        prev_gray_masked = gray_masked
        prev_mask = mask

        # Progress update every 100 decoded frames
        if progress_callback and read_count % 100 == 0:
            pct = int(100 * frame_idx / total_frames)
            progress_callback(0, 0, f"Pass 1: {pct}% ({read_count} frames decoded)")

    cap.release()

    if progress_callback:
        progress_callback(0, 0, f"Pass 1 done — {len(cut_frames)} cuts, {len(samples)} samples")

    # ── Build segments from cuts ────────────────────────────────────
    segments = []
    for i in range(len(cut_frames)):
        start = cut_frames[i]
        end = cut_frames[i + 1] if i + 1 < len(cut_frames) else total_frames - 1
        segments.append((start, end))

    # Filter short segments
    min_frames = int(min_segment_duration * fps)
    valid_segments = [
        (s, e) for s, e in segments if (e - s) >= min_frames
    ]
    if not valid_segments:
        valid_segments = segments if segments else [(0, total_frames - 1)]

    # ── Find motion valleys per segment ─────────────────────────────
    sample_arr = np.array(samples)  # (N, 2): frame_num, motion
    frame_nums = sample_arr[:, 0].astype(int)
    motion_vals = sample_arr[:, 1]

    # Smooth motion curve (~1 second window)
    smooth_win = max(3, int(fps / sample_step))
    if len(motion_vals) >= smooth_win:
        motion_smooth = uniform_filter1d(motion_vals, size=smooth_win)
    else:
        motion_smooth = motion_vals.copy()

    def _find_valley_in_range(start_f: int, end_f: int) -> int | None:
        """Find the lowest-motion sample frame in [start_f, end_f]."""
        mask = (frame_nums >= start_f) & (frame_nums <= end_f)
        indices = np.where(mask)[0]
        if len(indices) < 2:
            return None

        seg_motion = motion_smooth[indices]
        seg_frames = frame_nums[indices]

        # Try scipy peak detection on inverted signal
        inverted = -seg_motion
        m_range = seg_motion.max() - seg_motion.min()
        prom = max(0.3, m_range * 0.10)

        peaks, _ = find_peaks(inverted, prominence=prom, distance=max(2, int(5 * 0.5)))

        if len(peaks) == 0:
            # No valley — use global minimum
            idx = int(np.argmin(seg_motion))
        elif len(peaks) == 1:
            idx = peaks[0]
        else:
            # Prefer valleys later in the segment (pre-snap is before snap)
            depths = inverted[peaks]
            positions = peaks / max(1, len(seg_motion) - 1)
            d_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            scores = d_norm * 0.6 + positions * 0.4
            idx = peaks[int(np.argmax(scores))]

        return int(seg_frames[idx])

    # ── Pass 2: targeted seeks for final frames ─────────────────────
    total_seg = len(valid_segments)
    if progress_callback:
        progress_callback(0, total_seg, f"Pass 2: extracting frames from {total_seg} segments...")

    results = []
    play_num = 0

    # Open video once for all seeks
    cap = cv2.VideoCapture(video_path)

    for idx, (start, end) in enumerate(valid_segments):
        if progress_callback and idx % 10 == 0:
            progress_callback(idx, total_seg, f"Pass 2: segment {idx + 1}/{total_seg}")

        candidate = _find_valley_in_range(start, end)
        if candidate is None:
            continue

        # Read candidate frame at full resolution for formation check
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ret, frame = cap.read()
        if not ret:
            continue

        # View classification — skip endzone and scoreboard frames
        if sideline_only:
            small_cls = _downscale(frame)
            view = _classify_view(small_cls)
            if view != "sideline":
                continue

        small = _downscale(frame)
        score, blob_count = _score_formation(small)
        confidence = "high" if (score >= 0.35 and blob_count >= MIN_FORMATION_BLOBS) else "medium"

        # If low score, quick check ±5 frames for a better one
        if score < 0.25:
            best_score, best_frame_data, best_fn = score, frame, candidate
            for offset in [-5, -3, 3, 5]:
                test_fn = candidate + offset * sample_step
                if test_fn < start or test_fn > end:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, test_fn)
                ret2, f2 = cap.read()
                if not ret2:
                    continue
                s2, _ = _score_formation(_downscale(f2))
                if s2 > best_score:
                    best_score, best_frame_data, best_fn = s2, f2, test_fn
            frame = best_frame_data
            candidate = best_fn
            if best_score >= 0.35:
                confidence = "high"

        play_num += 1
        filename = f"play_{play_num:03d}_frame{candidate:06d}.png"
        save_path = out_path / filename
        cv2.imwrite(str(save_path), frame)

        results.append({
            "play_number": play_num,
            "frame_number": candidate,
            "timestamp": _frame_to_timestamp(candidate, fps),
            "output_path": str(save_path),
            "segment": (start, end),
            "confidence": confidence,
        })

    cap.release()

    if progress_callback:
        progress_callback(total_seg, total_seg, f"Done! Extracted {len(results)} pre-snap frames.")

    return results
