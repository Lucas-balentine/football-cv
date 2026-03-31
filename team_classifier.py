"""
Team classification via CNN embeddings + clustering.

For each player crop, extract a feature embedding from a pretrained
ResNet-18 (ImageNet weights). Then cluster all embeddings into 2 teams
using K-Means. The CNN captures texture, pattern, color, and shape —
much richer than raw color alone.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.cluster import KMeans


# Preprocessing to match ImageNet expectations
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lazy-loaded model
_model = None


def _get_model() -> torch.nn.Module:
    """Load ResNet-18 with ImageNet weights, remove the classification head."""
    global _model
    if _model is None:
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        # Remove the final FC layer — we want the 512-d embedding
        _model = torch.nn.Sequential(*list(resnet.children())[:-1])
        _model.eval()
    return _model


def extract_embeddings(crops: list[np.ndarray]) -> np.ndarray:
    """Extract 512-d feature embeddings for a list of BGR crops.

    Args:
        crops: list of BGR numpy arrays (any size).

    Returns:
        (N, 512) float32 array of L2-normalized embeddings.
    """
    model = _get_model()

    # Batch all crops together
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(_preprocess(rgb))

    batch = torch.stack(tensors)

    with torch.no_grad():
        features = model(batch)  # (N, 512, 1, 1)

    embeddings = features.squeeze(-1).squeeze(-1).numpy()  # (N, 512)

    # L2 normalize so K-Means uses cosine-like distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings


def classify_teams(
    crops: list[np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Classify a list of player crops into two teams using CNN embeddings.

    Returns:
        team_labels: list of team assignments (0 or 1) per player.
        embeddings: (N, 512) array of feature embeddings.
    """
    if len(crops) < 2:
        return [-1] * len(crops), np.zeros((len(crops), 512))

    embeddings = extract_embeddings(crops)

    # K-Means into 2 teams
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(embeddings)

    team_labels = [int(l) for l in kmeans.labels_]

    return team_labels, embeddings


# ── Multi-signal team classification ─────────────────────────────────────────
# Position zones — distance from LOS to confidence mapping
_ZONE_LINEMAN = 1.5   # ±1.5 yd from LOS — ambiguous
_ZONE_NEAR = 4.0      # 1.5–4 yd — moderate confidence
_ZONE_MID = 8.0       # 4–8 yd — high confidence
# 8+ yd — deep zone, very high confidence


def _extract_dominant_hsv(crop: np.ndarray) -> tuple | None:
    """Extract dominant jersey HSV from the torso region of a masked crop.

    Uses 15-55% of crop height to skip helmet (top) and legs (bottom),
    focusing on the jersey/torso region.

    Returns (H, S, V, white_ratio) tuple or None if too few valid pixels.
    white_ratio is the fraction of non-green, non-black pixels that are
    white/near-white (S < 40, V >= 80). This separates white-jersey teams
    from colored-jersey teams even when helmets are the same color.
    """
    h_crop, w_crop = crop.shape[:2]
    if h_crop < 10:
        return None

    # Torso region: skip helmet (top 15%) and legs (bottom 45%)
    y_start = max(0, int(h_crop * 0.15))
    y_end = max(y_start + 1, int(h_crop * 0.55))
    jersey = crop[y_start:y_end, :]
    if jersey.size == 0:
        return None

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
        return None

    white_ratio = n_white / total_meaningful

    # Dominant hue from COLORED pixels (not white, not green, not black)
    if n_colored >= 30:
        valid_h = h_ch[colored].astype(float)
        valid_s = s_ch[colored].astype(float)
        valid_v = v_ch[colored].astype(float)

        hist, _ = np.histogram(valid_h, bins=180, range=(0, 180))
        kernel = np.ones(5) / 5
        hist_smooth = np.convolve(hist, kernel, mode="same")
        dom_hue = float(np.argmax(hist_smooth))

        return (dom_hue, float(np.mean(valid_s)), float(np.mean(valid_v)), white_ratio)
    else:
        # Mostly white jersey — use white-pixel hue (won't be meaningful, but include)
        valid_h = h_ch[white].astype(float)
        dom_hue = float(np.median(valid_h)) if len(valid_h) > 0 else 0.0
        return (dom_hue, 0.0, float(np.mean(v_ch[white])), white_ratio)


def _circular_mean(hues: list[float]) -> float:
    """Compute circular mean of hues (OpenCV range 0-179) handling wraparound."""
    if not hues:
        return 0.0
    angles = [h * 2.0 * np.pi / 180.0 for h in hues]  # hue → radians (full circle)
    mean_sin = np.mean([np.sin(a) for a in angles])
    mean_cos = np.mean([np.cos(a) for a in angles])
    mean_angle = np.arctan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return mean_angle * 180.0 / (2.0 * np.pi)  # back to 0-179


def _circular_distance(h1: float, h2: float) -> float:
    """Angular distance between two OpenCV hues (0-179). Returns 0-90."""
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def _extract_crop_masked(image: np.ndarray, player: dict) -> np.ndarray | None:
    """Extract a masked crop from the image for a player dict."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = player["bbox"]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = image[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None

    mask = player.get("mask")
    if mask is not None:
        crop_mask = mask[y1c:y2c, x1c:x2c]
        crop = crop.copy()
        crop[crop_mask == 0] = 0

    return crop


def classify_teams_multi(
    field_players: list[dict],
    image: np.ndarray,
    ball_yard: int | None = None,
    offense_direction: str | None = None,
) -> tuple[list[int], list[dict]]:
    """Team classification using LOS position + jersey color.

    Two signals:
      1. Position relative to LOS (requires ball_yard + offense_direction)
      2. Jersey color (HSV hue + white_ratio from torso region)

    The offense_direction parameter (from the user's Game Setup tab) tells us
    which way the offense is moving:
      "right" → offense has higher yard numbers (moving toward 100)
      "left"  → offense has lower yard numbers (moving toward 0)

    Position zones from LOS determine confidence:
      < 1.5 yd  = lineman zone (ambiguous, color only)
      1.5–4 yd  = near zone (moderate)
      4–8 yd    = mid zone (high confidence)
      8+ yd     = deep zone (very high confidence)

    Color profiles are built from players with strong position signals (4+ yd
    from LOS). When no position seeds exist (no homography), falls back to
    unsupervised K-Means on [hue_cos, hue_sin, white_ratio].

    Args:
        field_players: player dicts with 'bbox', 'mask', and optionally 'yard'.
        image: original BGR image.
        ball_yard: yard line of the ball (0-100). None to skip position signal.
        offense_direction: "left" or "right". None to skip position signal.

    Returns:
        team_labels: list[int] per player (0=offense, 1=defense, -1=unknown).
        diagnostics: list[dict] per player with signal breakdown.
    """
    n = len(field_players)
    if n < 2:
        diags = [{"position_signal": None, "position_zone": None,
                  "color_signal": None, "color_hsv": None,
                  "confidence": 0.0, "conflict": False}] * n
        return [-1] * n, diags

    has_field_pos = all("yard" in p for p in field_players)

    # ── Phase 1: Position signal from LOS + direction ─────────────────────
    position_signals: list[int | None] = [None] * n
    position_zones: list[str | None] = [None] * n
    position_confs: list[float] = [0.0] * n

    if has_field_pos and ball_yard is not None and offense_direction is not None:
        for i, p in enumerate(field_players):
            yd = p["yard"]
            dist = abs(yd - ball_yard)

            # Determine which side of LOS this player is on
            if offense_direction == "right":
                # Offense has higher yard numbers
                on_offense_side = yd > ball_yard
            else:
                # Offense has lower yard numbers
                on_offense_side = yd < ball_yard

            # Assign zone and confidence
            if dist < _ZONE_LINEMAN:
                position_zones[i] = "lineman"
                position_signals[i] = None  # Too close, ambiguous
                position_confs[i] = 0.0
            elif dist < _ZONE_NEAR:
                position_zones[i] = "near"
                position_signals[i] = 0 if on_offense_side else 1
                position_confs[i] = 0.65
            elif dist < _ZONE_MID:
                position_zones[i] = "mid"
                position_signals[i] = 0 if on_offense_side else 1
                position_confs[i] = 0.85
            else:
                position_zones[i] = "deep"
                position_signals[i] = 0 if on_offense_side else 1
                position_confs[i] = 0.95

    # ── Phase 2: Color profiles from position seeds ───────────────────────
    # Use players 4+ yards from LOS (mid/deep zones) as reliable seeds
    offense_seeds = [i for i in range(n)
                     if position_signals[i] == 0
                     and position_zones[i] in ("mid", "deep")]
    defense_seeds = [i for i in range(n)
                     if position_signals[i] == 1
                     and position_zones[i] in ("mid", "deep")]

    # Extract dominant HSV + white_ratio for every player
    player_hsv: list[tuple | None] = []
    for p in field_players:
        crop = _extract_crop_masked(image, p)
        if crop is not None:
            player_hsv.append(_extract_dominant_hsv(crop))
        else:
            player_hsv.append(None)

    color_signals: list[int | None] = [None] * n

    # Build color profiles from position-seeded players
    off_hues = [player_hsv[i][0] for i in offense_seeds
                if player_hsv[i] is not None]
    def_hues = [player_hsv[i][0] for i in defense_seeds
                if player_hsv[i] is not None]
    off_whites = [player_hsv[i][3] for i in offense_seeds
                  if player_hsv[i] is not None]
    def_whites = [player_hsv[i][3] for i in defense_seeds
                  if player_hsv[i] is not None]

    if len(off_hues) >= 2 and len(def_hues) >= 2:
        off_color = _circular_mean(off_hues)
        def_color = _circular_mean(def_hues)
        profile_dist = _circular_distance(off_color, def_color)

        off_white_med = float(np.median(off_whites))
        def_white_med = float(np.median(def_whites))
        white_gap = abs(off_white_med - def_white_med)

        if profile_dist >= 10:
            # Hues are distinguishable — use hue distance
            for i, hsv in enumerate(player_hsv):
                if hsv is None:
                    continue
                d_off = _circular_distance(hsv[0], off_color)
                d_def = _circular_distance(hsv[0], def_color)
                color_signals[i] = 0 if d_off < d_def else 1

        elif white_gap >= 0.15:
            # Hues too similar but white_ratio separates teams
            for i, hsv in enumerate(player_hsv):
                if hsv is None:
                    continue
                wr = hsv[3]
                d_off = abs(wr - off_white_med)
                d_def = abs(wr - def_white_med)
                color_signals[i] = 0 if d_off < d_def else 1

    elif len(off_hues) >= 2 or len(def_hues) >= 2:
        # Only one side has a color profile — use single anchor
        if off_hues:
            anchor_color = _circular_mean(off_hues)
            anchor_side = 0
        else:
            anchor_color = _circular_mean(def_hues)
            anchor_side = 1

        for i, hsv in enumerate(player_hsv):
            if hsv is None:
                continue
            dist = _circular_distance(hsv[0], anchor_color)
            if dist < 20:
                color_signals[i] = anchor_side
            elif dist > 40:
                color_signals[i] = 1 - anchor_side

    else:
        # No position seeds — unsupervised K-Means on color features
        valid_indices = [i for i, hsv in enumerate(player_hsv) if hsv is not None]
        if len(valid_indices) >= 2:
            feature_vecs = []
            for i in valid_indices:
                h = player_hsv[i][0]
                wr = player_hsv[i][3]
                angle = h * 2.0 * np.pi / 180.0
                feature_vecs.append([np.cos(angle), np.sin(angle), wr * 2.0])
            X = np.array(feature_vecs)
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            km.fit(X)
            for idx, ki in enumerate(valid_indices):
                color_signals[ki] = int(km.labels_[idx])

    # ── Phase 3: Consensus ────────────────────────────────────────────────
    team_labels: list[int] = []
    diagnostics: list[dict] = []

    for i in range(n):
        ps = position_signals[i]
        cs = color_signals[i]
        zone = position_zones[i]
        pos_conf = position_confs[i]

        diag = {
            "position_signal": ps,
            "position_zone": zone,
            "color_signal": cs,
            "color_hsv": player_hsv[i],
            "confidence": 0.0,
            "conflict": False,
        }

        if ps is not None and cs is not None:
            if ps == cs:
                # Position and color agree — use position zone confidence
                final = ps
                conf = pos_conf
            else:
                # Disagree — trust position but flag conflict, reduce confidence
                final = ps
                conf = pos_conf * 0.7
                diag["conflict"] = True
        elif ps is not None:
            # Position only
            final = ps
            conf = pos_conf
        elif cs is not None:
            # Color only (lineman zone, or no homography)
            final = cs
            conf = 0.4
        else:
            final = -1
            conf = 0.0

        diag["confidence"] = conf
        team_labels.append(final)
        diagnostics.append(diag)

    return team_labels, diagnostics
