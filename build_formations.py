"""
Build canonical NFL formation templates from Big Data Bowl tracking data.

Parses the BDB 2020 train.csv (one row per player per play at handoff),
normalizes coordinates relative to the ball/LOS, classifies defensive schemes,
and aggregates median positions to produce formation_templates.json.

Also provides a formation classifier that matches detected player positions
(from the CV pipeline) to the nearest template using Hungarian matching.

Usage:
    python build_formations.py                              # build templates
    python build_formations.py --data-dir path/to/csvs
    python build_formations.py --render SHOTGUN_11          # render one formation
    python build_formations.py --render NICKEL              # render a defense scheme
    python build_formations.py --render SHOTGUN_11_vs_NICKEL  # render matchup
    python build_formations.py --render-all                 # render all formations
    python build_formations.py --stats                      # print frequency table
    python build_formations.py --classify SHOTGUN_11        # test classifier against a template
    python build_formations.py --demo-classify              # demo classifier with noisy positions
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from field_homography import TEMPLATE_SCALE, TEMPLATE_W
from interactive_homography import yard_to_template_y
from playbook_renderer import render_playbook

# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("datasets/big_data_bowl")
OUTPUT_DIR = Path("output/formations")
TEMPLATES_PATH = Path("formation_templates.json")

# ── NFL field constants ──────────────────────────────────────────────────────

FIELD_WIDTH_YD = 53.3
FIELD_CENTER_Y = FIELD_WIDTH_YD / 2  # 26.65


# ── Position → role mapping ─────────────────────────────────────────────────

NFL_POSITION_TO_ROLE = {
    # Offense
    "QB": "qb",
    "C": "oline", "G": "oline", "OG": "oline", "T": "oline", "OT": "oline",
    "WR": "skill", "TE": "skill", "RB": "skill", "HB": "skill", "FB": "skill",
    # Defense
    "DT": "defense", "DE": "defense", "NT": "defense", "DL": "defense",
    "ILB": "defense", "MLB": "defense", "OLB": "defense", "LB": "defense",
    "CB": "defense", "SS": "defense", "FS": "defense", "S": "defense", "DB": "defense",
}

# Positions that are definitively offense or defense
OFFENSE_POSITIONS = {"QB", "C", "G", "OG", "T", "OT", "WR", "TE", "RB", "HB", "FB"}
DEFENSE_POSITIONS = {
    "DT", "DE", "NT", "DL",
    "ILB", "MLB", "OLB", "LB",
    "CB", "SS", "FS", "S", "DB",
}

# Position groups for defensive scheme classification
DL_POSITIONS = {"DT", "DE", "NT", "DL"}
LB_POSITIONS = {"ILB", "MLB", "OLB", "LB"}
DB_POSITIONS = {"CB", "SS", "FS", "S", "DB"}

# OL positions for sub-position inference
OL_POSITIONS = {"C", "G", "OG", "T", "OT"}


# ── Defensive formation classification ──────────────────────────────────────

DEFENSE_SCHEMES = {
    (4, 3, 4): "4-3",
    (3, 4, 4): "3-4",
    (4, 2, 5): "NICKEL",
    (3, 3, 5): "NICKEL_3-3",
    (4, 1, 6): "DIME",
    (3, 2, 6): "DIME_3-2",
    (3, 1, 7): "QUARTER",
    (4, 1, 7): "QUARTER_4-1",
    (4, 4, 3): "4-4",
    (5, 2, 4): "5-2",
    (3, 4, 3): "3-4_HEAVY",  # not standard but appears in data
    (5, 3, 3): "5-3",
    (4, 3, 3): "GOAL_LINE",
}


def classify_defense_formation(dl_count: int, lb_count: int, db_count: int) -> str:
    """Classify defensive scheme from position group counts."""
    key = (dl_count, lb_count, db_count)
    if key in DEFENSE_SCHEMES:
        return DEFENSE_SCHEMES[key]
    # Fallback: use raw counts
    return f"{dl_count}-{lb_count}-{db_count}"


# ── Personnel parsing ────────────────────────────────────────────────────────

def parse_offense_personnel(personnel_str: str) -> str:
    """Parse '1 RB, 1 TE, 3 WR' → '11' (RB count + TE count)."""
    if not personnel_str or pd.isna(personnel_str):
        return "unknown"
    rb = te = 0
    for part in personnel_str.split(","):
        part = part.strip()
        match = re.match(r"(\d+)\s+(RB|TE|WR|OL|QB|DL|LB|DB|FB)", part)
        if match:
            count = int(match.group(1))
            pos = match.group(2)
            if pos == "RB":
                rb = count
            elif pos == "TE":
                te = count
    return f"{rb}{te}"


def parse_defense_personnel(personnel_str: str) -> tuple[int, int, int]:
    """Parse '2 DL, 3 LB, 6 DB' → (2, 3, 6)."""
    if not personnel_str or pd.isna(personnel_str):
        return (0, 0, 0)
    dl = lb = db = 0
    for part in personnel_str.split(","):
        part = part.strip()
        match = re.match(r"(\d+)\s+(DL|LB|DB)", part)
        if match:
            count = int(match.group(1))
            pos = match.group(2)
            if pos == "DL":
                dl = count
            elif pos == "LB":
                lb = count
            elif pos == "DB":
                db = count
    return (dl, lb, db)


# ── Data loading ─────────────────────────────────────────────────────────────

# BDB 2020 uses different abbreviations for some teams
TEAM_ABBR_FIX = {
    "BLT": "BAL",  # Baltimore Ravens
    "CLV": "CLE",  # Cleveland Browns
    "ARZ": "ARI",  # Arizona Cardinals
    "HST": "HOU",  # Houston Texans
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load the BDB 2020 train.csv and prepare for template extraction."""
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    print(f"Loading {csv_path} ...")

    cols = [
        "GameId", "PlayId", "Team", "X", "Y",
        "DisplayName", "Position",
        "PossessionTeam", "HomeTeamAbbr", "VisitorTeamAbbr",
        "OffenseFormation", "OffensePersonnel", "DefensePersonnel",
        "PlayDirection", "YardLine", "FieldPosition",
    ]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)

    # Normalize team abbreviations (BDB 2020 uses BLT/CLV/ARZ/HST inconsistently)
    df["PossessionTeam"] = df["PossessionTeam"].replace(TEAM_ABBR_FIX)
    df["HomeTeamAbbr"] = df["HomeTeamAbbr"].replace(TEAM_ABBR_FIX)
    df["VisitorTeamAbbr"] = df["VisitorTeamAbbr"].replace(TEAM_ABBR_FIX)

    print(f"  Loaded {len(df):,} rows ({df['PlayId'].nunique():,} plays)")
    return df


# ── Coordinate normalization ─────────────────────────────────────────────────

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize player coordinates relative to the ball/LOS.

    BDB 2020 coordinate system:
      X = 0-120 yards along the field (left end zone to right end zone)
      Y = 0-53.3 yards across the field

    We determine which team is offense based on PossessionTeam,
    then compute dx/dy relative to the center of the offensive line.
    """
    df = df.copy()

    # Determine offense/defense: compare Team (home/away) with PossessionTeam
    is_home_offense = df["PossessionTeam"] == df["HomeTeamAbbr"]
    df["is_offense"] = (
        ((df["Team"] == "home") & is_home_offense) |
        ((df["Team"] == "away") & ~is_home_offense)
    )

    # Normalize play direction: flip so offense always goes left → right
    # When PlayDirection == "left", offense moves toward lower X, so flip
    flip = df["PlayDirection"].str.lower() == "left"
    df["norm_x"] = df["X"].copy()
    df["norm_y"] = df["Y"].copy()
    df.loc[flip, "norm_x"] = 120.0 - df.loc[flip, "X"]
    df.loc[flip, "norm_y"] = FIELD_WIDTH_YD - df.loc[flip, "Y"]

    # Compute ball/LOS position per play: median X of OL players
    ol_mask = df["Position"].isin(OL_POSITIONS) & df["is_offense"]
    ol_x = df.loc[ol_mask].groupby("PlayId")["norm_x"].median()
    ol_x.name = "los_x"
    df = df.merge(ol_x, on="PlayId", how="left")

    # Fallback: if no OL found, use median X of all offensive players
    missing = df["los_x"].isna()
    if missing.any():
        off_x = df.loc[df["is_offense"]].groupby("PlayId")["norm_x"].median()
        off_x.name = "los_x_fallback"
        df = df.merge(off_x, on="PlayId", how="left")
        df.loc[missing, "los_x"] = df.loc[missing, "los_x_fallback"]
        df.drop(columns=["los_x_fallback"], inplace=True)

    # dx = player_x - los_x (positive = downfield / toward defense)
    # dy = player_y - field_center (positive = right side of field)
    df["dx"] = df["norm_x"] - df["los_x"]
    df["dy"] = df["norm_y"] - FIELD_CENTER_Y

    return df


# ── OL sub-position labeling ─────────────────────────────────────────────────

def assign_oline_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Infer LT/LG/C/RG/RT from lateral ordering of OL players per play.

    Vectorized: sorts OL players by dy within each play, assigns rank → label.
    """
    df = df.copy()
    df["sub_position"] = df["Position"]

    ol_mask = df["Position"].isin(OL_POSITIONS) & df["is_offense"]
    ol_rows = df.loc[ol_mask].copy()

    if ol_rows.empty:
        return df

    # Only process plays with exactly 5 OL
    ol_counts = ol_rows.groupby("PlayId").size()
    valid_plays = ol_counts[ol_counts == 5].index
    ol_valid = ol_rows[ol_rows["PlayId"].isin(valid_plays)].copy()

    if ol_valid.empty:
        return df

    # Rank by dy within each play (0-4)
    ol_valid["rank"] = ol_valid.groupby("PlayId")["dy"].rank(method="first").astype(int) - 1
    label_map = {0: "LT", 1: "LG", 2: "C", 3: "RG", 4: "RT"}
    ol_valid["sub_position"] = ol_valid["rank"].map(label_map)

    df.loc[ol_valid.index, "sub_position"] = ol_valid["sub_position"]
    return df


# ── Offensive skill position labeling ────────────────────────────────────────

SKILL_POSITIONS = {"WR", "TE", "RB", "HB", "FB"}


def assign_skill_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Number offensive skill players (WR_1, WR_2, TE_1, etc.) sorted by dy.

    Vectorized: ranks skill players within each (PlayId, Position) group by dy.
    """
    df = df.copy()

    skill_mask = df["Position"].isin(SKILL_POSITIONS) & df["is_offense"]
    skill_rows = df.loc[skill_mask].copy()

    if skill_rows.empty:
        return df

    # Preserve original index
    skill_rows["_orig_idx"] = skill_rows.index

    # Count how many of each position per play
    pos_counts = skill_rows.groupby(["PlayId", "Position"]).size().reset_index(name="pos_count")
    skill_rows = skill_rows.merge(pos_counts, on=["PlayId", "Position"], how="left")

    # Rank within each (PlayId, Position) by dy
    skill_rows["rank"] = skill_rows.groupby(["PlayId", "Position"])["dy"].rank(
        method="first"
    ).astype(int)

    # Build sub_position: "WR_1" if multiple, "WR" if single
    skill_rows["sub_position"] = np.where(
        skill_rows["pos_count"] > 1,
        skill_rows["Position"] + "_" + skill_rows["rank"].astype(str),
        skill_rows["Position"],
    )

    # Write back using preserved original indices
    df.loc[skill_rows["_orig_idx"].values, "sub_position"] = skill_rows["sub_position"].values
    return df


# ── Defensive sub-position labeling ──────────────────────────────────────────

def assign_defense_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Number defensive players by position group sorted by dy.

    Vectorized: classifies into DL/LB/DB groups, ranks by dy, numbers them.
    """
    df = df.copy()

    def_mask = ~df["is_offense"]
    def_rows = df.loc[def_mask].copy()

    if def_rows.empty:
        return df

    # Classify each defensive player into position group
    conditions = [
        def_rows["Position"].isin(DL_POSITIONS),
        def_rows["Position"].isin(LB_POSITIONS),
        def_rows["Position"].isin(DB_POSITIONS),
    ]
    choices = ["DL", "LB", "DB"]
    def_rows["pos_group"] = np.select(conditions, choices, default=def_rows["Position"])

    # Preserve original index through transformations
    def_rows["_orig_idx"] = def_rows.index

    # Count how many of each group per play
    group_counts = def_rows.groupby(["PlayId", "pos_group"]).size().reset_index(name="pg_count")
    def_rows = def_rows.merge(group_counts, on=["PlayId", "pos_group"], how="left")

    # Rank within each (PlayId, pos_group) by dy
    def_rows["rank"] = def_rows.groupby(["PlayId", "pos_group"])["dy"].rank(
        method="first"
    ).astype(int)

    # Build sub_position label: "DL_1" if multiple, "DL" if single
    def_rows["sub_position"] = np.where(
        def_rows["pg_count"] > 1,
        def_rows["pos_group"] + "_" + def_rows["rank"].astype(str),
        def_rows["pos_group"],
    )

    # Write back using preserved original indices
    df.loc[def_rows["_orig_idx"].values, "sub_position"] = def_rows["sub_position"].values
    return df


# ── Template building ─────────────────────────────────────────────────────────

def build_formation_templates(df: pd.DataFrame) -> dict:
    """Build the full formation template library from normalized data."""
    templates = {
        "_meta": {
            "source": "NFL Big Data Bowl 2020",
            "total_plays": int(df["PlayId"].nunique()),
        },
        "offense": {},
        "defense": {},
        "matchups": {},
    }

    # Parse personnel
    df["off_personnel"] = df["OffensePersonnel"].apply(parse_offense_personnel)
    dl_lb_db = df["DefensePersonnel"].apply(parse_defense_personnel)
    df["def_dl"] = dl_lb_db.apply(lambda x: x[0])
    df["def_lb"] = dl_lb_db.apply(lambda x: x[1])
    df["def_db"] = dl_lb_db.apply(lambda x: x[2])
    df["def_scheme"] = df.apply(
        lambda r: classify_defense_formation(int(r["def_dl"]), int(r["def_lb"]), int(r["def_db"])),
        axis=1,
    )

    # Filter: only plays with exactly 11 offense + 11 defense
    play_counts = df.groupby(["PlayId", "is_offense"]).size().unstack(fill_value=0)
    valid_plays = play_counts[(play_counts.get(True, 0) == 11) & (play_counts.get(False, 0) == 11)].index
    df = df[df["PlayId"].isin(valid_plays)]
    print(f"  Valid plays (11v11): {len(valid_plays):,}")

    # ── Build offense templates ──────────────────────────────────────────
    off_df = df[df["is_offense"]].copy()

    # Base formations (no personnel split)
    for formation in off_df["OffenseFormation"].dropna().unique():
        if not formation:
            continue
        fdf = off_df[off_df["OffenseFormation"] == formation]
        positions = _aggregate_positions(fdf)
        if positions:
            templates["offense"][formation] = {
                "play_count": int(fdf["PlayId"].nunique()),
                "personnel_variants": sorted(fdf["off_personnel"].unique().tolist()),
                "positions": positions,
            }

    # Personnel variants (e.g., SHOTGUN_11)
    for (formation, personnel), fdf in off_df.groupby(["OffenseFormation", "off_personnel"]):
        if not formation or not personnel or personnel == "unknown":
            continue
        key = f"{formation}_{personnel}"
        positions = _aggregate_positions(fdf)
        if positions:
            templates["offense"][key] = {
                "play_count": int(fdf["PlayId"].nunique()),
                "positions": positions,
            }

    # ── Build defense templates ──────────────────────────────────────────
    def_df = df[~df["is_offense"]].copy()

    for scheme in def_df["def_scheme"].dropna().unique():
        if not scheme:
            continue
        scheme_plays = df.loc[df["def_scheme"] == scheme, "PlayId"].unique()
        sdf = def_df[def_df["PlayId"].isin(scheme_plays)]
        positions = _aggregate_positions(sdf)
        if positions:
            # Get DL/LB/DB counts from the scheme
            dl, lb, db = _parse_scheme_counts(scheme, sdf)
            templates["defense"][scheme] = {
                "play_count": int(sdf["PlayId"].nunique()),
                "dl_count": dl,
                "lb_count": lb,
                "db_count": db,
                "positions": positions,
            }

    # ── Build matchup templates ──────────────────────────────────────────
    # Get per-play formation + scheme
    play_meta = df.drop_duplicates("PlayId")[["PlayId", "OffenseFormation", "off_personnel", "def_scheme"]]
    for (formation, personnel, scheme), group in play_meta.groupby(
        ["OffenseFormation", "off_personnel", "def_scheme"]
    ):
        if not formation or not personnel or not scheme or personnel == "unknown":
            continue
        play_count = len(group)
        if play_count < 50:  # skip rare matchups
            continue

        key = f"{formation}_{personnel}_vs_{scheme}"
        play_ids = group["PlayId"].values
        off_positions = _aggregate_positions(off_df[off_df["PlayId"].isin(play_ids)])
        def_positions = _aggregate_positions(def_df[def_df["PlayId"].isin(play_ids)])

        if off_positions and def_positions:
            templates["matchups"][key] = {
                "play_count": int(play_count),
                "offense": off_positions,
                "defense": def_positions,
            }

    return templates


def _parse_scheme_counts(scheme: str, def_df: pd.DataFrame) -> tuple[int, int, int]:
    """Get average DL/LB/DB counts for a scheme."""
    # Try to extract from scheme name
    for (dl, lb, db), name in DEFENSE_SCHEMES.items():
        if name == scheme:
            return (dl, lb, db)
    # Parse from format like "4-2-5"
    parts = scheme.split("-")
    if len(parts) == 3:
        try:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            pass
    # Fallback: count from data
    per_play = def_df.groupby("PlayId").apply(
        lambda g: pd.Series({
            "dl": g["Position"].isin(DL_POSITIONS).sum(),
            "lb": g["Position"].isin(LB_POSITIONS).sum(),
            "db": g["Position"].isin(DB_POSITIONS).sum(),
        })
    )
    return (
        int(per_play["dl"].median()),
        int(per_play["lb"].median()),
        int(per_play["db"].median()),
    )


def _aggregate_positions(df: pd.DataFrame, min_pct: float = 0.02) -> dict:
    """Aggregate player positions to median centroids by sub_position.

    Args:
        df: DataFrame with sub_position, dx, dy, PlayId columns.
        min_pct: Minimum percentage of plays a position must appear in to be included.
                 Filters out noise from edge cases (e.g., 1 CB on offense).
    """
    if df.empty or "sub_position" not in df.columns:
        return {}

    total_plays = df["PlayId"].nunique()
    min_count = max(5, int(total_plays * min_pct))

    result = {}
    for pos, group in df.groupby("sub_position"):
        if not pos or pd.isna(pos):
            continue
        play_count = int(group["PlayId"].nunique())
        if play_count < min_count:
            continue
        result[pos] = {
            "dx": round(float(group["dx"].median()), 2),
            "dy": round(float(group["dy"].median()), 2),
            "count": play_count,
            "dx_std": round(float(group["dx"].std()), 2),
            "dy_std": round(float(group["dy"].std()), 2),
        }
    return result


# ── Rendering adapter ─────────────────────────────────────────────────────────

def template_to_field_players(
    positions: dict,
    ball_yard: int = 50,
    side: str = "offense",
) -> list[dict]:
    """Convert template positions → field_players format for render_playbook().

    Args:
        positions: dict of position_label → {dx, dy, ...}
        ball_yard: yard line where ball is placed (0-100)
        side: "offense" or "defense"
    """
    field_players = []
    for pos_label, pos_data in positions.items():
        dx = pos_data["dx"]
        dy = pos_data["dy"]

        # Convert to template pixel coordinates
        # template_x = lateral position (dy maps to width)
        # template_y = along-field position (dx maps to length)
        template_x = (FIELD_CENTER_Y + dy) * TEMPLATE_SCALE
        template_y = yard_to_template_y(ball_yard) - dx * TEMPLATE_SCALE

        # Determine role from position label
        base_pos = pos_label.rstrip("_0123456789").upper()
        if base_pos in ("LT", "LG", "C", "RG", "RT"):
            role = "oline"
        elif base_pos == "QB":
            role = "qb"
        elif base_pos in ("WR", "TE", "RB", "HB", "FB"):
            role = "skill"
        elif side == "defense" or base_pos in ("DL", "LB", "DB", "DT", "DE", "NT",
                                                 "ILB", "MLB", "OLB", "CB", "SS", "FS"):
            role = "defense"
        else:
            role = "unknown"

        field_players.append({
            "field_pos": (template_x, template_y),
            "role": role,
            "label": pos_label,
        })

    return field_players


def render_formation(
    template_key: str,
    templates: dict,
    ball_yard: int = 50,
    output_path: Path | None = None,
):
    """Render a formation template to an image."""
    field_players = []
    team_labels = []

    # Check if it's a matchup (contains "_vs_")
    if "_vs_" in template_key:
        if template_key not in templates.get("matchups", {}):
            print(f"  Matchup '{template_key}' not found in templates")
            return None
        matchup = templates["matchups"][template_key]
        # Add offense
        off_players = template_to_field_players(matchup["offense"], ball_yard, "offense")
        field_players.extend(off_players)
        team_labels.extend([0] * len(off_players))
        # Add defense
        def_players = template_to_field_players(matchup["defense"], ball_yard, "defense")
        field_players.extend(def_players)
        team_labels.extend([1] * len(def_players))
    elif template_key in templates.get("offense", {}):
        off_players = template_to_field_players(
            templates["offense"][template_key]["positions"], ball_yard, "offense"
        )
        field_players.extend(off_players)
        team_labels.extend([0] * len(off_players))
    elif template_key in templates.get("defense", {}):
        def_players = template_to_field_players(
            templates["defense"][template_key]["positions"], ball_yard, "defense"
        )
        field_players.extend(def_players)
        team_labels.extend([1] * len(def_players))
    else:
        print(f"  Formation '{template_key}' not found in templates")
        return None

    img = render_playbook(field_players, team_labels, ball_yard=ball_yard, field_type="nfl")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))
        print(f"  Saved: {output_path}")

    return img


# ── Formation classifier ──────────────────────────────────────────────────

# Role mapping: template position label → detected role
_TMPL_POS_TO_ROLE = {}
for _pos in ("LT", "LG", "C", "RG", "RT"):
    _TMPL_POS_TO_ROLE[_pos] = "oline"
_TMPL_POS_TO_ROLE["QB"] = "qb"
for _pos in ("WR", "TE", "RB", "HB", "FB"):
    _TMPL_POS_TO_ROLE[_pos] = "skill"
for _pos in ("DL", "LB", "DB", "DT", "DE", "NT", "ILB", "MLB", "OLB", "CB", "SS", "FS", "S"):
    _TMPL_POS_TO_ROLE[_pos] = "defense"


def _tmpl_pos_role(pos_label: str) -> str:
    """Get the expected CV role for a template position label like 'WR_2'."""
    base = pos_label.rstrip("_0123456789").upper()
    return _TMPL_POS_TO_ROLE.get(base, "unknown")


def field_players_to_dxdy(
    field_players: list[dict],
    ball_yard: float | None = None,
) -> tuple[list[dict], float]:
    """Convert CV pipeline field_players to dx/dy coordinates for template matching.

    The CV pipeline's project_players_to_field() produces players with:
        field_pos: (template_x, template_y) in pixel coordinates
        role: "qb", "oline", "skill", "defense", "unknown"

    This function converts those to the same dx/dy yard-based coordinate
    system used by formation_templates.json:
        dx = yards ahead of LOS (positive = downfield toward defense)
        dy = yards from field center (positive = right side)

    Args:
        field_players: list of player dicts from project_players_to_field().
        ball_yard: yard line of the LOS. If None, estimated from OL positions.

    Returns:
        (players_dxdy, estimated_ball_yard)
        players_dxdy: list of {"dx", "dy", "role", "idx"} dicts
    """
    if not field_players:
        return [], ball_yard or 50.0

    # Convert template pixel coords to yards
    players_yd = []
    for i, p in enumerate(field_players):
        tx, ty = p["field_pos"]
        # Reverse yard_to_template_y: yard = (template_y / TEMPLATE_SCALE) - 10
        yard = (ty / TEMPLATE_SCALE) - 10.0
        lateral = tx / TEMPLATE_SCALE  # 0 to 53.3 yards
        players_yd.append({
            "yard": yard,
            "lateral": lateral,
            "role": p.get("role", "unknown"),
            "idx": i,
        })

    # Estimate ball_yard from OL if not provided
    if ball_yard is None:
        ol_yards = [p["yard"] for p in players_yd if p["role"] == "oline"]
        if ol_yards:
            ball_yard = float(np.median(ol_yards))
        else:
            # Fallback: median of all offensive player yards
            off_yards = [p["yard"] for p in players_yd
                         if p["role"] in ("qb", "oline", "skill")]
            if off_yards:
                ball_yard = float(np.median(off_yards))
            else:
                ball_yard = float(np.median([p["yard"] for p in players_yd]))

    # Convert to dx/dy relative to LOS / field center.
    # Template convention: offense faces UPWARD (lower template_y = downfield).
    # dx = ball_yard - player_yard so that QB behind LOS has negative dx.
    # dy = lateral - center so that right side of field is positive.
    result = []
    for p in players_yd:
        result.append({
            "dx": ball_yard - p["yard"],
            "dy": p["lateral"] - FIELD_CENTER_Y,
            "role": p["role"],
            "idx": p["idx"],
        })

    return result, ball_yard


# Position-importance weights for scoring.
# OL positions are identical across all formations → low weight.
# QB/RB/skill positions define the formation → high weight.
_POSITION_WEIGHTS = {
    "qb": 3.0,       # QB depth is THE key differentiator
    "oline": 0.3,     # OL is always at the LOS — not discriminative
    "skill": 1.5,     # WR/TE/RB alignment matters
    "defense": 1.0,   # Standard weight for defense
    "unknown": 1.0,
}


# ── Feature-based formation classification ────────────────────────────────

def classify_offense_features(players_dxdy: list[dict]) -> dict:
    """Extract key offensive formation features from detected positions.

    Analyzes the backfield geometry, receiver spread, and TE alignment
    to characterize the offensive formation. These features can be used
    for formation classification or as diagnostic information.

    NOTE: If positions are from handoff-frame data (like BDB 2020),
    QB depth is ~4 yards for ALL formations (post-dropback) and is
    NOT discriminative between SHOTGUN/SINGLEBACK/PISTOL. Pre-snap
    frame data gives much better QB-depth discrimination.

    Returns:
        dict with raw features and best-guess family classification.
    """
    # Extract key players by role
    qbs = [p for p in players_dxdy if p["role"] == "qb"]
    skills = [p for p in players_dxdy if p["role"] == "skill"]

    features = {}

    # QB depth (behind LOS = negative)
    if qbs:
        qb_dx = qbs[0]["dx"]
        features["qb_dx"] = round(qb_dx, 2)
    else:
        qb_dx = None
        features["qb_dx"] = None

    # Categorize skill players by position on the field
    # Wide: |dy| > 8 (receivers split wide)
    # Slot: 4 < |dy| < 8 (slot receivers or wing TEs)
    # Box: |dy| < 4 (backfield — RBs, FBs, inline TEs)
    wide = [p for p in skills if abs(p["dy"]) > 8]
    slot = [p for p in skills if 4 < abs(p["dy"]) <= 8]
    box = [p for p in skills if abs(p["dy"]) <= 4]

    features["n_wide"] = len(wide)
    features["n_slot"] = len(slot)
    features["n_box"] = len(box)

    # Backfield depth analysis
    if box and qb_dx is not None:
        box_dxs = [p["dx"] for p in box]
        features["box_avg_dx"] = round(float(np.mean(box_dxs)), 2)
        # How many box players are deeper than QB?
        features["n_behind_qb"] = sum(1 for d in box_dxs if d < qb_dx - 0.5)
    else:
        features["box_avg_dx"] = None
        features["n_behind_qb"] = 0

    # Receiver spread (max |dy| of wide players)
    if wide:
        features["wr_max_spread"] = round(float(max(abs(p["dy"]) for p in wide)), 1)
    else:
        features["wr_max_spread"] = 0.0

    # Formation family guess (coarse — best effort with available data)
    family = "UNKNOWN"
    confidence = 0.3

    if len(box) == 0 and len(wide) >= 3:
        family = "EMPTY"
        confidence = 0.85
    elif features.get("n_behind_qb", 0) >= 2:
        family = "I_FORM"
        confidence = 0.7
    elif len(wide) >= 3:
        family = "SPREAD"  # SHOTGUN/PISTOL with 3+ wide
        confidence = 0.6
    elif len(wide) >= 2:
        family = "PRO"  # SINGLEBACK/SHOTGUN with 2 wide
        confidence = 0.5
    else:
        family = "HEAVY"  # JUMBO or goal-line
        confidence = 0.5

    return {
        "family": family,
        "confidence": round(confidence, 2),
        "features": features,
    }


def classify_defense_scheme(players_dxdy: list[dict]) -> dict:
    """Classify defensive scheme from player depth bands.

    Since the CV pipeline only provides role="defense" (not DL/LB/DB),
    we estimate position groups from depth (dx):
      DL: 0 ≤ dx < 1.5 (at the LOS)
      LB: 1.5 ≤ dx < 5 (second level)
      DB: dx ≥ 5 (deep coverage)

    Returns:
        dict with "scheme", "dl"/"lb"/"db" counts, "confidence".
    """
    def_players = [p for p in players_dxdy if p["role"] == "defense"]

    if len(def_players) < 7:
        return {"scheme": "UNKNOWN", "dl": 0, "lb": 0, "db": 0, "confidence": 0.1}

    # Classify by depth bands.
    # DL: right at the LOS (dx < 1.5)
    # LB: second level (1.5 ≤ dx < 4.0)
    # DB: deep coverage (dx ≥ 4.0)
    # Note: threshold at 4.0 (not 5.0) because nickel/dime DBs
    # often play at 4-5 yards depth near the LOS.
    dl = [p for p in def_players if p["dx"] < 1.5]
    lb = [p for p in def_players if 1.5 <= p["dx"] < 4.0]
    db = [p for p in def_players if p["dx"] >= 4.0]

    dl_count = len(dl)
    lb_count = len(lb)
    db_count = len(db)

    # Use existing scheme classification
    scheme = classify_defense_formation(dl_count, lb_count, db_count)

    # Confidence based on total count (should be 11)
    total = dl_count + lb_count + db_count
    confidence = 0.8 if total == 11 else max(0.3, 0.8 - abs(total - 11) * 0.1)

    return {
        "scheme": scheme,
        "dl": dl_count,
        "lb": lb_count,
        "db": db_count,
        "confidence": round(confidence, 2),
    }


def classify_formation(
    field_players: list[dict],
    templates: dict,
    ball_yard: float | None = None,
    side: str = "offense",
    top_n: int = 5,
    use_std_weights: bool = False,
    role_mismatch_penalty: float = 5.0,
    frequency_weight: float = 1.0,
    max_position_diff: int = 4,
) -> list[dict]:
    """Classify detected players against formation templates using Hungarian matching.

    Uses scipy's linear_sum_assignment (Hungarian algorithm) to find the
    optimal player-to-template-position assignment that minimizes total
    distance. Position importance is weighted: QB and skill positions
    contribute more to the score than OL (which is identical across all
    offensive formations).

    Includes a Bayesian frequency prior: common formations (many play_count)
    get a small score bonus over rare formations, preventing obscure schemes
    from dominating the rankings purely on geometric coincidence.

    Args:
        field_players: list of player dicts from project_players_to_field().
            Each needs "field_pos" (template_x, template_y) and "role".
        templates: formation_templates.json loaded as dict.
        ball_yard: yard line of the LOS. If None, estimated from OL positions.
        side: "offense" or "defense" — which template section to match.
        top_n: number of top matches to return.
        use_std_weights: weight distances by template position std devs
            (more variable positions contribute less to score).
        role_mismatch_penalty: extra cost (yards) added when a detected
            player's role doesn't match the template position's expected role.
            Set to 0 to disable role constraints.
        frequency_weight: how much to favor common formations (0=ignore,
            1=strong prior). The prior is -log10(play_count/max_count) * weight.
        max_position_diff: skip templates whose position count differs
            from detected count by more than this.

    Returns:
        list of matches sorted by score (ascending = best match first):
        [{"formation": str, "score": float, "matched": int,
          "detected": int, "template_positions": int,
          "assignment": {player_idx: template_pos_label, ...},
          "play_count": int}, ...]
    """
    from scipy.optimize import linear_sum_assignment

    # Convert field_players to dx/dy
    players_dxdy, est_ball_yard = field_players_to_dxdy(field_players, ball_yard)

    if not players_dxdy:
        return []

    # Separate players by side
    off_roles = {"qb", "oline", "skill"}
    def_roles = {"defense"}

    if side == "offense":
        detected = [p for p in players_dxdy if p["role"] in off_roles]
        # Include unknown players as candidates (could be unclassified offense)
        unknown = [p for p in players_dxdy if p["role"] == "unknown"]
        detected.extend(unknown)
    elif side == "defense":
        detected = [p for p in players_dxdy if p["role"] in def_roles]
        unknown = [p for p in players_dxdy if p["role"] == "unknown"]
        detected.extend(unknown)
    else:
        detected = players_dxdy

    if not detected:
        return []

    section_templates = templates.get(side, {})

    # Compute max play_count for frequency prior
    max_plays = max(
        (t.get("play_count", 1) for k, t in section_templates.items()
         if not k.startswith("_") and isinstance(t, dict)),
        default=1,
    )

    results = []

    # Minimum play count to consider a template reliable
    MIN_PLAY_COUNT = 30

    for tmpl_name, tmpl_data in section_templates.items():
        if tmpl_name.startswith("_"):
            continue

        # Skip templates based on too few plays (unreliable centroids)
        if tmpl_data.get("play_count", 0) < MIN_PLAY_COUNT:
            continue

        positions = tmpl_data.get("positions", {})
        if not positions:
            continue

        # Build arrays for template positions.
        # Templates may contain overlapping positions (G alongside LG/RG,
        # T alongside LT/RT, WR alongside WR_1/WR_2, etc.) because
        # sub-position assignment only works for plays with exactly 5 OL
        # or multiple skill players. We deduplicate by preferring the more
        # specific (sub-position) labels over generic ones.
        pos_items = {}
        for pos_label, pos_data in positions.items():
            if isinstance(pos_data, dict) and "dx" in pos_data:
                pos_items[pos_label] = pos_data

        if not pos_items:
            continue

        # Deduplicate: remove generic/variant positions when specific ones exist
        has_specific_ol = any(k in pos_items for k in ("LG", "RG", "LT", "RT"))
        has_numbered_wr = any(k.startswith("WR_") for k in pos_items)
        has_numbered_te = any(k.startswith("TE_") for k in pos_items)
        has_numbered_rb = any(k.startswith("RB_") for k in pos_items)
        has_rb = "RB" in pos_items  # HB is a synonym for RB

        skip_labels = set()
        if has_specific_ol:
            skip_labels.update({"G", "OG", "T", "OT"})
        if has_numbered_wr:
            skip_labels.add("WR")
        if has_numbered_te:
            skip_labels.add("TE")
        if has_numbered_rb:
            skip_labels.update({"RB", "HB"})
        elif has_rb:
            skip_labels.add("HB")  # HB is synonym for RB

        # Sort by count descending, apply dedup, trim to ~11 positions
        sorted_items = sorted(
            [(k, v) for k, v in pos_items.items() if k not in skip_labels],
            key=lambda x: x[1].get("count", 0),
            reverse=True,
        )

        # Trim to at most detected_count + 2 positions.
        n_det = len(detected)
        trim_to = n_det + 2
        sorted_items = sorted_items[:trim_to]

        tmpl_labels = []
        tmpl_dxdy = []
        tmpl_stds = []
        tmpl_roles = []
        for pos_label, pos_data in sorted_items:
            tmpl_labels.append(pos_label)
            tmpl_dxdy.append((pos_data["dx"], pos_data["dy"]))
            sx = max(pos_data.get("dx_std", 1.0) or 1.0, 0.5)
            sy = max(pos_data.get("dy_std", 1.0) or 1.0, 0.5)
            tmpl_stds.append((sx, sy))
            tmpl_roles.append(_tmpl_pos_role(pos_label))

        n_tmpl = len(tmpl_dxdy)

        # Skip templates with wildly different position counts
        if abs(n_det - n_tmpl) > max_position_diff:
            continue

        # Build cost matrix — pad to square for Hungarian algorithm
        size = max(n_det, n_tmpl)
        # Unmatched cost: high but not astronomical (allows partial matches)
        UNMATCHED_COST = 30.0
        cost = np.full((size, size), fill_value=UNMATCHED_COST)

        for i in range(n_det):
            for j in range(n_tmpl):
                ddx = detected[i]["dx"] - tmpl_dxdy[j][0]
                ddy = detected[i]["dy"] - tmpl_dxdy[j][1]

                if use_std_weights:
                    # Mahalanobis-like distance with diagonal covariance
                    dist = np.sqrt(
                        (ddx / tmpl_stds[j][0]) ** 2
                        + (ddy / tmpl_stds[j][1]) ** 2
                    )
                else:
                    # Euclidean distance in yards
                    dist = np.sqrt(ddx ** 2 + ddy ** 2)

                # Weight by position importance (QB high, OL low)
                tmpl_role = tmpl_roles[j]
                pos_weight = _POSITION_WEIGHTS.get(tmpl_role, 1.0)
                dist *= pos_weight

                # Role mismatch penalty
                if role_mismatch_penalty > 0:
                    det_role = detected[i]["role"]
                    if det_role != "unknown" and tmpl_role != "unknown":
                        if det_role != tmpl_role:
                            dist += role_mismatch_penalty

                cost[i, j] = dist

        # Solve optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # Score from real (non-dummy) matches
        matched_cost = 0.0
        matched_count = 0
        assignment = {}

        for r, c in zip(row_ind, col_ind):
            if r < n_det and c < n_tmpl and cost[r, c] < UNMATCHED_COST:
                matched_cost += cost[r, c]
                matched_count += 1
                assignment[detected[r]["idx"]] = tmpl_labels[c]

        if matched_count == 0:
            continue

        # Score = mean distance per matched position
        score = matched_cost / matched_count

        # Penalty for count mismatch (missing or extra players)
        count_diff = abs(n_det - n_tmpl)
        score += count_diff * 2.0

        # Frequency prior: penalize rare formations
        # -log10(play_count / max_plays) ranges from 0 (most common) to ~3 (rare)
        play_count = max(tmpl_data.get("play_count", 1), 1)
        if frequency_weight > 0 and max_plays > 0:
            rarity = -np.log10(play_count / max_plays)
            score += rarity * frequency_weight

        results.append({
            "formation": tmpl_name,
            "section": side,
            "score": round(float(score), 3),
            "matched": matched_count,
            "detected": n_det,
            "template_positions": n_tmpl,
            "assignment": assignment,
            "play_count": int(play_count),
        })

    # Sort by score (lower = better match)
    results.sort(key=lambda r: r["score"])
    return results[:top_n]


def match_formation(
    field_players: list[dict],
    templates: dict,
    ball_yard: float | None = None,
    top_n: int = 5,
) -> dict:
    """Identify both offensive and defensive formations from detected players.

    Two-level classification:
    1. Feature-based: extracts key features (QB depth, backfield geometry,
       depth bands) for robust family-level classification.
    2. Template matching: Hungarian matching within the identified family
       (and across all templates) for detailed personnel identification.

    Args:
        field_players: list of player dicts from project_players_to_field().
        templates: formation_templates.json loaded as dict.
        ball_yard: yard line of the LOS. If None, estimated from OL positions.
        top_n: number of top matches per side to return.

    Returns:
        {"ball_yard": float,
         "offense_family": {"family": str, "confidence": float, "features": dict},
         "defense_scheme": {"scheme": str, "dl": int, "lb": int, "db": int, ...},
         "offense_matches": [top_n template matches...],
         "defense_matches": [top_n template matches...]}
    """
    # Convert to dx/dy once for all analyses
    players_dxdy, est_ball_yard = field_players_to_dxdy(field_players, ball_yard)

    # Level 1: Feature-based classification
    off_family = classify_offense_features(players_dxdy)
    def_scheme = classify_defense_scheme(players_dxdy)

    # Level 2: Template matching (uses all templates, ranked by score)
    off_matches = classify_formation(
        field_players, templates,
        ball_yard=est_ball_yard, side="offense", top_n=top_n,
    )
    def_matches = classify_formation(
        field_players, templates,
        ball_yard=est_ball_yard, side="defense", top_n=top_n,
    )

    return {
        "ball_yard": round(est_ball_yard, 1),
        "offense_family": off_family,
        "defense_scheme": def_scheme,
        "offense_matches": off_matches,
        "defense_matches": def_matches,
    }


def demo_classify(templates: dict, formation_key: str | None = None) -> None:
    """Demo the classifier by generating noisy positions from a template.

    Takes a template formation, adds random noise to simulate CV detection
    error, then runs classify_formation() to see if it recovers the correct
    formation.

    Args:
        templates: formation_templates.json loaded as dict.
        formation_key: specific formation to test (e.g. "SHOTGUN_11").
            If None, tests a few common formations.
    """
    rng = np.random.default_rng(42)
    ball_yard = 50

    test_keys = []
    if formation_key:
        test_keys = [formation_key]
    else:
        # Offense: prefer personnel-specific templates (e.g., SHOTGUN_11)
        # which have exactly the right positions for matching.
        # Base templates aggregate across all personnel variants.
        off_data = templates.get("offense", {})
        personnel_keys = sorted(
            [k for k in off_data if not k.startswith("_") and "_" in k
             and any(c.isdigit() for c in k.split("_")[-1])],
            key=lambda k: off_data[k].get("play_count", 0),
            reverse=True,
        )
        test_keys.extend(personnel_keys[:4])

        # Defense: top 3 by play count
        def_data = templates.get("defense", {})
        def_keys = sorted(
            [k for k in def_data if not k.startswith("_")],
            key=lambda k: def_data[k].get("play_count", 0),
            reverse=True,
        )
        test_keys.extend(def_keys[:3])

    for key in test_keys:
        # Find which section this key is in
        section = None
        positions = None
        for s in ["offense", "defense"]:
            if key in templates.get(s, {}):
                section = s
                positions = templates[s][key].get("positions", {})
                break
        if section is None or not positions:
            print(f"  '{key}' not found in templates, skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"Testing: {key} ({section})")
        print(f"{'─'*60}")

        # Generate synthetic field_players from template with noise.
        # Apply same dedup as classifier: remove generic positions when
        # specific ones exist, then take top-11 by count.
        all_pos = {
            k: v for k, v in positions.items()
            if isinstance(v, dict) and "dx" in v
        }
        has_spec_ol = any(k in all_pos for k in ("LG", "RG", "LT", "RT"))
        has_num_wr = any(k.startswith("WR_") for k in all_pos)
        has_num_te = any(k.startswith("TE_") for k in all_pos)
        has_num_rb = any(k.startswith("RB_") for k in all_pos)
        has_rb = "RB" in all_pos
        skip = set()
        if has_spec_ol:
            skip.update({"G", "OG", "T", "OT"})
        if has_num_wr:
            skip.add("WR")
        if has_num_te:
            skip.add("TE")
        if has_num_rb:
            skip.update({"RB", "HB"})
        elif has_rb:
            skip.add("HB")

        pos_items = sorted(
            [(k, v) for k, v in all_pos.items() if k not in skip],
            key=lambda x: x[1].get("count", 0),
            reverse=True,
        )[:11]  # 11 players per side in a real game

        field_players = []
        for pos_label, pos_data in pos_items:
            dx = pos_data["dx"]
            dy = pos_data["dy"]

            # Add noise proportional to the position's natural variation
            sx = pos_data.get("dx_std", 1.0) or 1.0
            sy = pos_data.get("dy_std", 1.0) or 1.0
            noise_dx = rng.normal(0, min(sx * 0.5, 1.5))
            noise_dy = rng.normal(0, min(sy * 0.5, 2.0))

            noisy_dx = dx + noise_dx
            noisy_dy = dy + noise_dy

            # Convert back to template pixel coordinates
            template_x = (FIELD_CENTER_Y + noisy_dy) * TEMPLATE_SCALE
            template_y = yard_to_template_y(ball_yard) - noisy_dx * TEMPLATE_SCALE

            # Determine role from position label
            role = _tmpl_pos_role(pos_label)

            field_players.append({
                "field_pos": (template_x, template_y),
                "role": role,
                "label": pos_label,
            })

        print(f"  Synthetic players: {len(field_players)} with noise σ~0.5×std")

        # Convert to dx/dy for feature-based classification
        players_dxdy, _ = field_players_to_dxdy(field_players, ball_yard)

        # Level 1: Feature-based classification
        if section == "offense":
            family_result = classify_offense_features(players_dxdy)
            features = family_result["features"]
            print(f"  Features: QB depth={features.get('qb_dx', '?')}yd"
                  f"  wide={features.get('n_wide', '?')}"
                  f"  slot={features.get('n_slot', '?')}"
                  f"  box={features.get('n_box', '?')}")
            print(f"  Family guess: {family_result['family']}"
                  f" ({family_result['confidence']:.0%})")
        else:
            scheme_result = classify_defense_scheme(players_dxdy)
            print(f"  Scheme: {scheme_result['scheme']}"
                  f" ({scheme_result['dl']}DL-{scheme_result['lb']}LB-{scheme_result['db']}DB)"
                  f" (confidence={scheme_result['confidence']:.0%})")

        # Level 2: Template matching
        matches = classify_formation(
            field_players, templates,
            ball_yard=ball_yard, side=section, top_n=5,
        )

        if not matches:
            print("  No template matches found!")
            continue

        # Print results
        rank1 = matches[0]
        correct = rank1["formation"] == key
        print(f"  {'✓' if correct else '✗'} Best template: {rank1['formation']}"
              f"  (score={rank1['score']:.2f}, matched={rank1['matched']}/{rank1['template_positions']})")

        if not correct:
            for i, m in enumerate(matches):
                if m["formation"] == key:
                    print(f"    Correct template at rank {i+1}: score={m['score']:.2f}")
                    break
            else:
                print(f"    Correct template not in top {len(matches)}")

        # Show top 5
        print(f"\n  {'Rank':<6s} {'Formation':<25s} {'Score':>7s} {'Matched':>8s} {'Plays':>8s}")
        print(f"  {'─'*58}")
        for i, m in enumerate(matches):
            marker = " ◀" if m["formation"] == key else ""
            print(f"  {i+1:<6d} {m['formation']:<25s} {m['score']:>7.2f}"
                  f" {m['matched']:>4d}/{m['template_positions']:<3d}"
                  f" {m['play_count']:>8,d}{marker}")


# ── Stats printing ────────────────────────────────────────────────────────────

def print_stats(templates: dict) -> None:
    """Print a frequency table of formations and defensive schemes."""
    meta = templates.get("_meta", {})
    print(f"\n{'='*60}")
    print(f"Formation Template Statistics")
    print(f"Source: {meta.get('source', 'unknown')}")
    print(f"Total plays: {meta.get('total_plays', 0):,}")
    print(f"{'='*60}")

    print(f"\n── Offensive Formations ──")
    print(f"{'Formation':<25s} {'Plays':>8s}  {'%':>6s}")
    print(f"{'-'*45}")
    off = templates.get("offense", {})
    # Only show base formations (no underscore suffix with digits)
    base_formations = {
        k: v for k, v in off.items()
        if not any(c.isdigit() for c in k.split("_")[-1]) or "_" not in k
    }
    total_off = sum(v["play_count"] for v in base_formations.values())
    for k in sorted(base_formations, key=lambda k: -base_formations[k]["play_count"]):
        v = base_formations[k]
        pct = v["play_count"] / total_off * 100 if total_off else 0
        print(f"  {k:<23s} {v['play_count']:>8,d}  {pct:>5.1f}%")

    print(f"\n── Defensive Schemes ──")
    print(f"{'Scheme':<25s} {'Plays':>8s}  {'%':>6s}  {'DL-LB-DB':>10s}")
    print(f"{'-'*55}")
    defe = templates.get("defense", {})
    total_def = sum(v["play_count"] for v in defe.values())
    for k in sorted(defe, key=lambda k: -defe[k]["play_count"]):
        v = defe[k]
        pct = v["play_count"] / total_def * 100 if total_def else 0
        counts = f"{v.get('dl_count', '?')}-{v.get('lb_count', '?')}-{v.get('db_count', '?')}"
        print(f"  {k:<23s} {v['play_count']:>8,d}  {pct:>5.1f}%  {counts:>10s}")

    print(f"\n── Top Matchups ──")
    print(f"{'Matchup':<40s} {'Plays':>8s}")
    print(f"{'-'*50}")
    matchups = templates.get("matchups", {})
    for k in sorted(matchups, key=lambda k: -matchups[k]["play_count"])[:15]:
        v = matchups[k]
        print(f"  {k:<38s} {v['play_count']:>8,d}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build NFL formation templates")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Path to data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--render", type=str, default=None,
        help="Render a specific formation (e.g., SHOTGUN_11, NICKEL, SHOTGUN_11_vs_NICKEL)",
    )
    parser.add_argument(
        "--render-all", action="store_true",
        help="Render all formations to output/formations/",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print formation frequency statistics",
    )
    parser.add_argument(
        "--ball-yard", type=int, default=50,
        help="Yard line for rendering (default: 50)",
    )
    parser.add_argument(
        "--classify", type=str, default=None,
        help="Test classifier against a specific template (e.g., SHOTGUN_11)",
    )
    parser.add_argument(
        "--demo-classify", action="store_true",
        help="Demo classifier with noisy synthetic positions",
    )
    args = parser.parse_args()

    # If just rendering/classifying from existing templates
    if (args.render or args.render_all or args.stats
            or args.classify or args.demo_classify) and TEMPLATES_PATH.exists():
        with open(TEMPLATES_PATH) as f:
            templates = json.load(f)

        if args.stats:
            print_stats(templates)
            return

        if args.classify:
            demo_classify(templates, args.classify)
            return

        if args.demo_classify:
            demo_classify(templates)
            return

        if args.render:
            render_formation(
                args.render, templates, args.ball_yard,
                OUTPUT_DIR / f"{args.render}.png",
            )
            return

        if args.render_all:
            _render_all(templates, args.ball_yard)
            return

    # Build templates from data
    print(f"\n{'='*60}")
    print("Building NFL Formation Templates")
    print(f"{'='*60}\n")

    df = load_data(args.data_dir)

    print("\nNormalizing coordinates...")
    df = normalize_coordinates(df)

    print("Assigning OL sub-positions...")
    df = assign_oline_labels(df)

    print("Assigning offensive skill position labels...")
    df = assign_skill_labels(df)

    print("Assigning defensive sub-positions...")
    df = assign_defense_labels(df)

    print("\nBuilding templates...")
    templates = build_formation_templates(df)

    # Save
    with open(TEMPLATES_PATH, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\nSaved templates to {TEMPLATES_PATH}")

    # Print summary
    n_off = len(templates["offense"])
    n_def = len(templates["defense"])
    n_match = len(templates["matchups"])
    print(f"  Offense formations: {n_off}")
    print(f"  Defense schemes: {n_def}")
    print(f"  Matchups: {n_match}")

    # Always print stats after building
    print_stats(templates)

    # Render if requested
    if args.render:
        render_formation(
            args.render, templates, args.ball_yard,
            OUTPUT_DIR / f"{args.render}.png",
        )
    if args.render_all:
        _render_all(templates, args.ball_yard)


def _render_all(templates: dict, ball_yard: int = 50):
    """Render all formation templates to output/formations/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for section in ["offense", "defense", "matchups"]:
        for key in templates.get(section, {}):
            if key.startswith("_"):
                continue
            render_formation(key, templates, ball_yard, OUTPUT_DIR / f"{key}.png")
            count += 1

    print(f"\nRendered {count} formations to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
