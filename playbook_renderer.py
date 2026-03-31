"""
Playbook-style renderer for football field diagrams.

Takes projected player position data from the interactive homography pipeline
and renders a clean, modern whiteboard-style playbook diagram using Pillow.

Usage:
    from playbook_renderer import render_playbook

    img = render_playbook(field_players, team_labels, ball_yard=25, field_type="college")
    img.save("playbook.png")
"""

import platform
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from field_homography import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    TEMPLATE_H,
    TEMPLATE_SCALE,
    TEMPLATE_W,
)
from interactive_homography import HASH_POSITIONS, yard_to_template_y


# ── Palette ─────────────────────────────────────────────────────────────────

BG_COLOR = (252, 250, 247)            # warm off-white
FIELD_LINE_COLOR = (215, 215, 210)    # light warm gray for yard lines
FIELD_LINE_MAJOR = (195, 195, 190)    # slightly darker for 10-yard lines
LOS_COLOR = (70, 130, 200)            # muted blue for line of scrimmage
HASH_COLOR = (205, 205, 200)          # subtle gray for hash marks
YARD_NUM_COLOR = (190, 190, 185)      # light gray for yard numbers
SIDELINE_COLOR = (225, 225, 220)      # very subtle sideline indicators
LABEL_COLOR = (80, 80, 80)            # dark gray for position labels
LEGEND_TEXT_COLOR = (120, 120, 120)   # medium gray for legend text

OFFENSE_COLOR = (30, 100, 200)        # blue for offense
DEFENSE_COLOR = (200, 50, 50)         # red for defense
QB_COLOR = (30, 100, 200)             # same blue, filled
UNKNOWN_COLOR = (150, 150, 150)       # gray for unclassified
BALL_COLOR = (180, 130, 40)           # amber for ball marker


# ── Layout constants ────────────────────────────────────────────────────────

OUTPUT_WIDTH = 960                    # output image width in pixels
PADDING_YARDS = 10                    # padding beyond player bounding box
MIN_VISIBLE_YARDS = 20                # minimum vertical yard span to show

PLAYER_RADIUS = 14                    # radius for O and X symbols
PLAYER_STROKE = 3                     # line width for player symbols
LABEL_FONT_SIZE = 13                  # font size for position labels
YARD_NUM_FONT_SIZE = 24               # font size for yard numbers
LEGEND_FONT_SIZE = 12                 # font size for legend text
LOS_LABEL_FONT_SIZE = 11             # font size for LOS yard label


# ── Role label mapping ──────────────────────────────────────────────────────

ROLE_LABELS = {
    "qb": "QB",
    "oline": "OL",
    "skill": "SK",
    "defense": "DE",
    "unknown": "",
}


# ── Font loading ────────────────────────────────────────────────────────────

_FONT_PATHS = {
    "Darwin": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFPro.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ],
}


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a clean sans-serif font at the given size, with platform fallbacks."""
    system = platform.system()
    candidates = _FONT_PATHS.get(system, [])

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    # Pillow 10.1+ has a better default font with size support
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Coordinate helpers ──────────────────────────────────────────────────────

def _compute_crop_region(
    field_players: list[dict],
    ball_yard: int | None = None,
    padding_yards: float = PADDING_YARDS,
) -> tuple[float, float, float, float]:
    """Compute the crop region in template coordinates.

    Always spans full field width (sideline to sideline).
    Y axis is cropped to the player bounding box + padding.

    Returns (crop_x0, crop_y0, crop_x1, crop_y1) in template pixels.
    """
    if not field_players:
        # Default: center around ball yard or 50-yard line
        center_yard = ball_yard if ball_yard is not None else 50
        center_y = yard_to_template_y(center_yard)
        half_span = MIN_VISIBLE_YARDS * TEMPLATE_SCALE / 2
        return (0, max(0, center_y - half_span), TEMPLATE_W, min(TEMPLATE_H, center_y + half_span))

    ys = [p["field_pos"][1] for p in field_players]
    y_min = min(ys)
    y_max = max(ys)

    # Include ball position in bounds if provided
    if ball_yard is not None:
        ball_y = yard_to_template_y(ball_yard)
        y_min = min(y_min, ball_y)
        y_max = max(y_max, ball_y)

    # Add padding
    pad_px = padding_yards * TEMPLATE_SCALE
    y_min = max(0, y_min - pad_px)
    y_max = min(TEMPLATE_H, y_max + pad_px)

    # Enforce minimum visible span
    min_span = MIN_VISIBLE_YARDS * TEMPLATE_SCALE
    current_span = y_max - y_min
    if current_span < min_span:
        center = (y_min + y_max) / 2
        y_min = max(0, center - min_span / 2)
        y_max = min(TEMPLATE_H, y_min + min_span)
        if y_max - y_min < min_span:
            y_min = max(0, y_max - min_span)

    # X axis: always full field width
    return (0, y_min, TEMPLATE_W, y_max)


def _template_to_canvas(
    tx: float,
    ty: float,
    crop_region: tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
) -> tuple[int, int]:
    """Map template coordinates to output canvas pixel coordinates (portrait)."""
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_region
    cx = int((tx - crop_x0) / (crop_x1 - crop_x0) * canvas_w)
    cy = int((ty - crop_y0) / (crop_y1 - crop_y0) * canvas_h)
    return cx, cy


def _template_to_canvas_landscape(
    tx: float,
    ty: float,
    crop_region: tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
    direction: str = "right",
) -> tuple[int, int]:
    """Map portrait template coords to landscape canvas coords.

    Template Y (along field) → canvas X (horizontal).
    Template X (across field) → canvas Y (vertical, inverted so near sideline at bottom).
    direction="left" flips the horizontal axis.
    """
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_region
    # Yards → horizontal
    cx = int((ty - crop_y0) / (crop_y1 - crop_y0) * canvas_w)
    # Lateral → vertical (inverted: template x=0 → bottom, x=W → top)
    cy = int((1.0 - (tx - crop_x0) / (crop_x1 - crop_x0)) * canvas_h)
    if direction == "left":
        cx = canvas_w - 1 - cx
    return cx, cy


# ── Field background drawing ───────────────────────────────────────────────

def _draw_field_background(
    draw: ImageDraw.Draw,
    canvas_w: int,
    canvas_h: int,
    crop_region: tuple[float, float, float, float],
    field_type: str = "college",
) -> None:
    """Draw the whiteboard field markings: yard lines, hash marks, numbers."""
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_region

    # Sideline indicators (very subtle vertical lines at field edges)
    left_x, _ = _template_to_canvas(0, crop_y0, crop_region, canvas_w, canvas_h)
    right_x, _ = _template_to_canvas(TEMPLATE_W, crop_y0, crop_region, canvas_w, canvas_h)
    draw.line([(left_x, 0), (left_x, canvas_h)], fill=SIDELINE_COLOR, width=2)
    draw.line([(right_x - 1, 0), (right_x - 1, canvas_h)], fill=SIDELINE_COLOR, width=2)

    # Hash mark positions
    near_yd, far_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    near_hash_tx = near_yd * TEMPLATE_SCALE
    far_hash_tx = far_yd * TEMPLATE_SCALE

    # Yard number font
    num_font = _load_font(YARD_NUM_FONT_SIZE)

    # Draw yard lines every 5 yards across the playing field (yard 0 to 100)
    for yard in range(0, 101, 5):
        template_y = yard_to_template_y(yard)

        if template_y < crop_y0 or template_y > crop_y1:
            continue

        _, cy = _template_to_canvas(0, template_y, crop_region, canvas_w, canvas_h)

        # Line style
        if yard % 10 == 0:
            color = FIELD_LINE_MAJOR
            width = 2
        else:
            color = FIELD_LINE_COLOR
            width = 1

        draw.line([(0, cy), (canvas_w, cy)], fill=color, width=width)

        # Yard numbers at every 10-yard mark (skip 0 and 100)
        if yard % 10 == 0 and 0 < yard < 100:
            display_num = yard if yard <= 50 else 100 - yard
            num_str = str(display_num)

            # Left side (1/6 of width)
            left_x, _ = _template_to_canvas(TEMPLATE_W / 6, template_y, crop_region, canvas_w, canvas_h)
            bbox = num_font.getbbox(num_str)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((left_x - tw // 2, cy - th // 2 - 2), num_str, fill=YARD_NUM_COLOR, font=num_font)

            # Right side (5/6 of width)
            right_x, _ = _template_to_canvas(TEMPLATE_W * 5 / 6, template_y, crop_region, canvas_w, canvas_h)
            draw.text((right_x - tw // 2, cy - th // 2 - 2), num_str, fill=YARD_NUM_COLOR, font=num_font)

    # Hash marks: small ticks at each yard (skipping yard lines)
    for yard in range(0, 101):
        template_y = yard_to_template_y(yard)
        if template_y < crop_y0 or template_y > crop_y1:
            continue
        if yard % 5 == 0:
            continue  # already drawn as yard lines

        _, cy = _template_to_canvas(0, template_y, crop_region, canvas_w, canvas_h)

        # Near hash
        hx_near, _ = _template_to_canvas(near_hash_tx, template_y, crop_region, canvas_w, canvas_h)
        draw.line([(hx_near - 4, cy), (hx_near + 4, cy)], fill=HASH_COLOR, width=1)

        # Far hash
        hx_far, _ = _template_to_canvas(far_hash_tx, template_y, crop_region, canvas_w, canvas_h)
        draw.line([(hx_far - 4, cy), (hx_far + 4, cy)], fill=HASH_COLOR, width=1)


def _draw_line_of_scrimmage(
    draw: ImageDraw.Draw,
    ball_yard: int,
    crop_region: tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Draw the line of scrimmage as a muted blue line across the field (portrait)."""
    template_y = yard_to_template_y(ball_yard)
    _, cy = _template_to_canvas(0, template_y, crop_region, canvas_w, canvas_h)

    draw.line([(0, cy), (canvas_w, cy)], fill=LOS_COLOR, width=2)

    # Small yard label on the right edge
    los_font = _load_font(LOS_LABEL_FONT_SIZE)
    label = f"{ball_yard} yd"
    bbox = los_font.getbbox(label)
    tw = bbox[2] - bbox[0]
    draw.text((canvas_w - tw - 8, cy - 16), label, fill=LOS_COLOR, font=los_font)


# ── Landscape field drawing ───────────────────────────────────────────────

def _draw_field_background_landscape(
    draw: ImageDraw.Draw,
    canvas_w: int,
    canvas_h: int,
    crop_region: tuple[float, float, float, float],
    field_type: str = "college",
    direction: str = "right",
) -> None:
    """Draw the whiteboard field markings in landscape orientation.

    Yard lines are vertical, sidelines are horizontal (top/bottom),
    hash marks are short vertical ticks.
    """
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_region

    # Sideline indicators (horizontal lines at top and bottom)
    draw.line([(0, 0), (canvas_w, 0)], fill=SIDELINE_COLOR, width=2)
    draw.line([(0, canvas_h - 1), (canvas_w, canvas_h - 1)], fill=SIDELINE_COLOR, width=2)

    # Hash mark positions
    near_yd, far_yd = HASH_POSITIONS.get(field_type, HASH_POSITIONS["college"])
    near_hash_tx = near_yd * TEMPLATE_SCALE
    far_hash_tx = far_yd * TEMPLATE_SCALE

    # Yard number font
    num_font = _load_font(YARD_NUM_FONT_SIZE)

    # Yard lines every 5 yards (vertical lines)
    for yard in range(0, 101, 5):
        template_y = yard_to_template_y(yard)
        if template_y < crop_y0 or template_y > crop_y1:
            continue

        cx, _ = _template_to_canvas_landscape(
            0, template_y, crop_region, canvas_w, canvas_h, direction,
        )

        # Line style
        if yard % 10 == 0:
            color = FIELD_LINE_MAJOR
            width = 2
        else:
            color = FIELD_LINE_COLOR
            width = 1

        draw.line([(cx, 0), (cx, canvas_h)], fill=color, width=width)

        # Yard numbers at every 10-yard mark (skip 0 and 100)
        if yard % 10 == 0 and 0 < yard < 100:
            display_num = yard if yard <= 50 else 100 - yard
            num_str = str(display_num)

            bbox = num_font.getbbox(num_str)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Top edge (far sideline)
            _, top_cy = _template_to_canvas_landscape(
                TEMPLATE_W * 5 / 6, template_y, crop_region, canvas_w, canvas_h, direction,
            )
            draw.text((cx - tw // 2, top_cy - th // 2 - 2), num_str, fill=YARD_NUM_COLOR, font=num_font)

            # Bottom edge (near sideline)
            _, bot_cy = _template_to_canvas_landscape(
                TEMPLATE_W / 6, template_y, crop_region, canvas_w, canvas_h, direction,
            )
            draw.text((cx - tw // 2, bot_cy - th // 2 - 2), num_str, fill=YARD_NUM_COLOR, font=num_font)

    # Hash marks: small vertical ticks at each yard (skipping yard lines)
    for yard in range(0, 101):
        template_y = yard_to_template_y(yard)
        if template_y < crop_y0 or template_y > crop_y1:
            continue
        if yard % 5 == 0:
            continue

        # Near hash
        hx_near, hy_near = _template_to_canvas_landscape(
            near_hash_tx, template_y, crop_region, canvas_w, canvas_h, direction,
        )
        draw.line([(hx_near, hy_near - 4), (hx_near, hy_near + 4)], fill=HASH_COLOR, width=1)

        # Far hash
        hx_far, hy_far = _template_to_canvas_landscape(
            far_hash_tx, template_y, crop_region, canvas_w, canvas_h, direction,
        )
        draw.line([(hx_far, hy_far - 4), (hx_far, hy_far + 4)], fill=HASH_COLOR, width=1)


def _draw_line_of_scrimmage_landscape(
    draw: ImageDraw.Draw,
    ball_yard: int,
    crop_region: tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
    direction: str = "right",
) -> None:
    """Draw the line of scrimmage as a vertical line in landscape orientation."""
    template_y = yard_to_template_y(ball_yard)
    cx, _ = _template_to_canvas_landscape(
        0, template_y, crop_region, canvas_w, canvas_h, direction,
    )

    draw.line([(cx, 0), (cx, canvas_h)], fill=LOS_COLOR, width=2)

    # Small yard label at the top
    los_font = _load_font(LOS_LABEL_FONT_SIZE)
    label = f"{ball_yard} yd"
    bbox = los_font.getbbox(label)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, 4), label, fill=LOS_COLOR, font=los_font)


# ── Player symbol drawing ──────────────────────────────────────────────────

def _draw_player_offense(
    draw: ImageDraw.Draw,
    cx: int,
    cy: int,
    role: str,
    color: tuple[int, int, int],
    radius: int = PLAYER_RADIUS,
) -> None:
    """Draw an offensive player symbol (hollow circle, filled for QB)."""
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]

    if role == "qb":
        # QB gets a filled circle
        draw.ellipse(bbox, fill=color, outline=color, width=PLAYER_STROKE)
    else:
        # Other offense: hollow circle
        draw.ellipse(bbox, outline=color, width=PLAYER_STROKE)


def _draw_player_defense(
    draw: ImageDraw.Draw,
    cx: int,
    cy: int,
    color: tuple[int, int, int],
    radius: int = PLAYER_RADIUS,
) -> None:
    """Draw a defensive player symbol (X mark)."""
    r = int(radius * 0.85)  # slightly smaller than the circle radius
    draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=PLAYER_STROKE)
    draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=PLAYER_STROKE)


def _draw_player_label(
    draw: ImageDraw.Draw,
    cx: int,
    cy: int,
    role: str,
    radius: int = PLAYER_RADIUS,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    above: bool = False,
) -> None:
    """Draw a position label near the player symbol.

    Args:
        above: If True, place label above the symbol (used for defense).
               If False, place below (used for offense).
    """
    label = ROLE_LABELS.get(role, "")
    if not label:
        return

    if font is None:
        font = _load_font(LABEL_FONT_SIZE)

    bbox = font.getbbox(label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if above:
        draw.text(
            (cx - tw // 2, cy - radius - th - 4),
            label,
            fill=LABEL_COLOR,
            font=font,
        )
    else:
        draw.text(
            (cx - tw // 2, cy + radius + 3),
            label,
            fill=LABEL_COLOR,
            font=font,
        )


def _draw_ball_marker(
    draw: ImageDraw.Draw,
    cx: int,
    cy: int,
) -> None:
    """Draw a small diamond marker for the ball position."""
    size = 6
    points = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    draw.polygon(points, fill=BALL_COLOR)


# ── Legend ──────────────────────────────────────────────────────────────────

def _draw_legend(
    draw: ImageDraw.Draw,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Draw a compact legend in the top-left corner."""
    font = _load_font(LEGEND_FONT_SIZE)
    x0 = 12
    y = 12
    spacing = 20
    r = 6  # symbol radius for legend

    # Offense: hollow circle
    draw.ellipse(
        [x0 - r, y - r, x0 + r, y + r],
        outline=OFFENSE_COLOR,
        width=2,
    )
    draw.text((x0 + r + 6, y - 7), "offense", fill=LEGEND_TEXT_COLOR, font=font)
    y += spacing

    # QB: filled circle
    draw.ellipse(
        [x0 - r, y - r, x0 + r, y + r],
        fill=QB_COLOR,
        outline=QB_COLOR,
        width=2,
    )
    draw.text((x0 + r + 6, y - 7), "qb", fill=LEGEND_TEXT_COLOR, font=font)
    y += spacing

    # Defense: X mark
    xr = int(r * 0.85)
    draw.line([(x0 - xr, y - xr), (x0 + xr, y + xr)], fill=DEFENSE_COLOR, width=2)
    draw.line([(x0 - xr, y + xr), (x0 + xr, y - xr)], fill=DEFENSE_COLOR, width=2)
    draw.text((x0 + r + 6, y - 7), "defense", fill=LEGEND_TEXT_COLOR, font=font)


# ── Main entry point ───────────────────────────────────────────────────────

MAX_LANDSCAPE_HEIGHT = 600                # cap vertical size for landscape mode


def render_playbook(
    field_players: list[dict],
    team_labels: list[int],
    ball_yard: int | None = None,
    field_type: str = "college",
    orientation: str = "horizontal",
    direction: str = "right",
) -> Image.Image:
    """Render a clean playbook-style diagram from projected player positions.

    Args:
        field_players: list of player dicts with 'field_pos', 'role', etc.
                       (output from project_players_to_field).
        team_labels: list of team assignments (0=offense, 1=defense, -1=unknown).
        ball_yard: yard line where the ball is placed (0-100), or None.
        field_type: "college" or "nfl" — determines hash mark positions.
        orientation: "vertical" (portrait) or "horizontal" (landscape).
        direction: "right" or "left" — offense direction (landscape only).

    Returns:
        PIL Image of the playbook diagram.
    """
    is_landscape = orientation == "horizontal"

    # Step 1: Compute crop region (always in portrait template coords)
    crop_region = _compute_crop_region(field_players, ball_yard)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_region

    # Step 2: Compute canvas dimensions
    crop_w = crop_x1 - crop_x0    # lateral (across field)
    crop_h = crop_y1 - crop_y0    # along field (yards)

    if is_landscape:
        # Landscape: width = yards (long axis), height = lateral
        canvas_w = OUTPUT_WIDTH
        canvas_h = min(MAX_LANDSCAPE_HEIGHT, max(200, int(OUTPUT_WIDTH * crop_w / crop_h)))
    else:
        canvas_w = OUTPUT_WIDTH
        canvas_h = max(200, int(OUTPUT_WIDTH * crop_h / crop_w))

    # Step 3: Create canvas
    img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Coordinate mapping helper
    def _to_canvas(fx: float, fy: float) -> tuple[int, int]:
        if is_landscape:
            return _template_to_canvas_landscape(
                fx, fy, crop_region, canvas_w, canvas_h, direction,
            )
        return _template_to_canvas(fx, fy, crop_region, canvas_w, canvas_h)

    # Step 4: Draw field background
    if is_landscape:
        _draw_field_background_landscape(
            draw, canvas_w, canvas_h, crop_region, field_type, direction,
        )
    else:
        _draw_field_background(draw, canvas_w, canvas_h, crop_region, field_type)

    # Step 5: Draw line of scrimmage
    if ball_yard is not None:
        if is_landscape:
            _draw_line_of_scrimmage_landscape(
                draw, ball_yard, crop_region, canvas_w, canvas_h, direction,
            )
        else:
            _draw_line_of_scrimmage(draw, ball_yard, crop_region, canvas_w, canvas_h)

    # Step 6: Load label font
    label_font = _load_font(LABEL_FONT_SIZE)

    # Step 7: Separate players by team label for layered drawing
    defense_indices = []
    offense_indices = []
    unknown_indices = []

    for i in range(min(len(team_labels), len(field_players))):
        team = team_labels[i]
        if team == 0:
            offense_indices.append(i)
        elif team == 1:
            defense_indices.append(i)
        else:
            unknown_indices.append(i)

    # Draw defense first (X marks, behind offense if overlapping)
    for i in defense_indices:
        p = field_players[i]
        fx, fy = p["field_pos"]
        cx, cy = _to_canvas(fx, fy)
        _draw_player_defense(draw, cx, cy, DEFENSE_COLOR)
        _draw_player_label(draw, cx, cy, "defense", font=label_font, above=True)

    # Draw unknown players
    for i in unknown_indices:
        p = field_players[i]
        fx, fy = p["field_pos"]
        cx, cy = _to_canvas(fx, fy)
        _draw_player_offense(draw, cx, cy, "unknown", UNKNOWN_COLOR)

    # Draw offense on top (circles)
    for i in offense_indices:
        p = field_players[i]
        fx, fy = p["field_pos"]
        cx, cy = _to_canvas(fx, fy)
        _draw_player_offense(draw, cx, cy, "offense", OFFENSE_COLOR)
        _draw_player_label(draw, cx, cy, "offense", font=label_font)

    # Step 8: Draw ball marker
    if ball_yard is not None:
        ball_ty = yard_to_template_y(ball_yard)
        ball_tx = TEMPLATE_W / 2
        bx, by = _to_canvas(ball_tx, ball_ty)
        _draw_ball_marker(draw, bx, by)

    # Step 9: Draw legend
    _draw_legend(draw, canvas_w, canvas_h)

    return img
