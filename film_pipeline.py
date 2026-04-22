"""
Film pipeline: video upload -> pre-snap extraction -> batch analysis.

Runs as a background job with progress tracking.  The FastAPI endpoints
in api.py create a job, launch this pipeline in a daemon thread, and
let the frontend poll for status / results.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import cv2
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from presnap_extractor import extract_presnap_frames
from interactive_homography import run_interactive_homography, _detect_yard_numbers
from build_formations import match_formation


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AnalyzedPlay:
    play_number: int
    frame_number: int
    timestamp: str
    confidence: str
    screenshot_b64: str  # base64-encoded JPEG of the pre-snap frame
    ball_yard: int
    ball_yard_source: str  # "detected" or "default"
    playbook_data: dict | None
    formation_matches: dict | None
    pipeline_meta: dict | None
    error: str | None = None


@dataclass
class FilmJob:
    job_id: str
    status: str = "queued"  # queued | extracting | analyzing | complete | error
    total_plays: int = 0
    analyzed_count: int = 0
    current_step: str = ""
    plays: list[AnalyzedPlay] = dc_field(default_factory=list)
    error: str | None = None
    created_at: float = 0.0


# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, FilmJob] = {}


def create_job() -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = FilmJob(job_id=job_id, created_at=time.time())
    return job_id


def get_job(job_id: str) -> FilmJob | None:
    """Retrieve a job by ID, or None if not found."""
    return _jobs.get(job_id)


# ── Ball-yard auto-detection ──────────────────────────────────────────────────

def estimate_ball_yard(image_bgr: np.ndarray) -> tuple[int, str]:
    """Estimate ball yard line from painted yard numbers.

    Uses the Roboflow yard-number detection model (same one used by the
    homography pipeline).  If >= 2 distinct numbers are detected, returns
    the midpoint of the visible range.  Otherwise returns 35 as a safe
    default (most plays happen between the 20s and the 50).

    Returns:
        (ball_yard, source) where source is "detected" or "default".
    """
    h_img, w_img = image_bgr.shape[:2]
    if min(h_img, w_img) >= 720:
        try:
            yard_numbers = _detect_yard_numbers(image_bgr)
            if len(yard_numbers) >= 2:
                yards = [yn["yard"] for yn in yard_numbers]
                midpoint = int(round((min(yards) + max(yards)) / 2))
                return midpoint, "detected"
        except Exception:
            pass
    return 35, "default"


# ── Playbook data builder ────────────────────────────────────────────────────

_TEAM_MAP = {0: "offense", 1: "defense", -1: "unknown"}


def _build_playbook_data(
    field_players: list[dict],
    team_labels: list[int],
    ball_yard: int,
    offense_direction: str,
    field_type: str,
) -> dict:
    """Build the playbook_data dict (same format as /analyze endpoint)."""
    players_out = []
    for i, (p, lbl) in enumerate(zip(field_players, team_labels)):
        yd = p.get("yard")
        lat = p.get("lateral")
        if yd is None:
            continue
        players_out.append({
            "id": i + 1,
            "team": _TEAM_MAP.get(lbl, "unknown"),
            "yard_line": round(float(yd), 1),
            "lateral_yards": round(float(lat), 1) if lat is not None else None,
        })

    return {
        "ball_yard_line": int(ball_yard),
        "offense_direction": offense_direction,
        "field_type": field_type,
        "player_count": len(players_out),
        "players": players_out,
    }


def _run_formation_matching(
    field_players: list[dict],
    templates: dict,
    ball_yard: float,
) -> dict | None:
    """Run formation matching, returning a clean serializable dict."""
    if not templates:
        return None
    try:
        fm_result = match_formation(field_players, templates, ball_yard=ball_yard)

        def _clean_match(m):
            return {
                "formation": m.get("formation", "unknown"),
                "score": round(float(m.get("score", 999)), 3),
                "matched": m.get("matched", 0),
                "detected": m.get("detected", 0),
                "play_count": m.get("play_count", 0),
            }

        return {
            "ball_yard": fm_result.get("ball_yard"),
            "offense_family": fm_result.get("offense_family"),
            "defense_scheme": fm_result.get("defense_scheme"),
            "offense_matches": [
                _clean_match(m) for m in fm_result.get("offense_matches", [])[:3]
            ],
            "defense_matches": [
                _clean_match(m) for m in fm_result.get("defense_matches", [])[:3]
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_film_pipeline(
    job_id: str,
    video_path: str,
    field_type: str,
    offense_direction: str,
    templates: dict,
) -> None:
    """Background worker: extract pre-snap frames and analyze each one.

    Updates the FilmJob in-place so the status polling endpoint can
    report progress.
    """
    job = _jobs[job_id]

    try:
        # ── Step 1: Extract pre-snap frames ──────────────────────────
        job.status = "extracting"
        job.current_step = "Extracting pre-snap frames from video..."

        output_dir = f"videos/presnap/{Path(video_path).stem}_{job_id}"

        def _progress(current, total, status):
            job.current_step = status

        frames = extract_presnap_frames(
            video_path,
            output_dir,
            sideline_only=True,
            progress_callback=_progress,
        )

        job.total_plays = len(frames)

        if not frames:
            job.status = "complete"
            job.current_step = "No sideline pre-snap frames found."
            return

        # ── Step 2: Analyze each frame ───────────────────────────────
        job.status = "analyzing"

        for i, frame_info in enumerate(frames):
            job.analyzed_count = i
            job.current_step = f"Analyzing play {i + 1} of {len(frames)}..."

            frame_path = frame_info.get("output_path") or frame_info.get("path", "")
            img = cv2.imread(frame_path)
            if img is None:
                continue

            # Encode screenshot as base64 JPEG
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            screenshot_b64 = base64.b64encode(buf).decode("utf-8")

            # Auto-estimate ball yard
            ball_yard, ball_source = estimate_ball_yard(img)

            # Run the full CV pipeline
            try:
                (
                    _field_img,
                    _overlay,
                    _corr_debug,
                    _warped,
                    summary,
                    field_players,
                    team_labels,
                ) = run_interactive_homography(
                    img,
                    ball_yard,
                    field_type=field_type,
                    offense_direction=offense_direction,
                )

                playbook_data = _build_playbook_data(
                    field_players, team_labels,
                    ball_yard, offense_direction, field_type,
                )
                formation_matches = _run_formation_matching(
                    field_players, templates, float(ball_yard),
                )

                play = AnalyzedPlay(
                    play_number=frame_info["play_number"],
                    frame_number=frame_info["frame_number"],
                    timestamp=frame_info["timestamp"],
                    confidence=frame_info["confidence"],
                    screenshot_b64=screenshot_b64,
                    ball_yard=ball_yard,
                    ball_yard_source=ball_source,
                    playbook_data=playbook_data,
                    formation_matches=formation_matches,
                    pipeline_meta={
                        "summary": summary,
                        "player_count": len(field_players),
                    },
                )
            except Exception as e:
                play = AnalyzedPlay(
                    play_number=frame_info["play_number"],
                    frame_number=frame_info["frame_number"],
                    timestamp=frame_info["timestamp"],
                    confidence=frame_info["confidence"],
                    screenshot_b64=screenshot_b64,
                    ball_yard=ball_yard,
                    ball_yard_source=ball_source,
                    playbook_data=None,
                    formation_matches=None,
                    pipeline_meta=None,
                    error=str(e),
                )

            job.plays.append(play)

        job.analyzed_count = len(frames)
        job.status = "complete"
        job.current_step = f"Done! {len(job.plays)} plays analyzed."

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.current_step = f"Error: {e}"
