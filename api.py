"""
TendencyIQ CV API — FastAPI wrapper for the football CV pipeline.

Exposes the interactive_homography + formation matching pipeline
as a REST API that the React frontend can call.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from interactive_homography import run_interactive_homography
from build_formations import match_formation
from film_pipeline import create_job, get_job, run_film_pipeline

UPLOAD_DIR = Path("videos/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TendencyIQ CV API",
    description="Football sideline image analysis pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load formation templates once at startup ──────────────────────────────────

TEMPLATES_PATH = Path(__file__).parent / "formation_templates.json"
TEMPLATES = {}

if TEMPLATES_PATH.exists():
    with open(TEMPLATES_PATH) as f:
        TEMPLATES = json.load(f)
    print(f"✓ Loaded formation templates ({TEMPLATES_PATH.name})")
else:
    print(f"⚠ formation_templates.json not found — formation matching disabled")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "templates_loaded": bool(TEMPLATES),
    }


# ── Main Analysis Endpoint ────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    ball_yard: int = Form(...),
    offense_direction: str = Form("right"),
    field_type: str = Form("college"),
    conf: float = Form(0.3),
    ball_image_x: int | None = Form(None),
    ball_image_y: int | None = Form(None),
):
    """Analyze a sideline football image.

    Accepts a JPEG/PNG image + game parameters.
    Optional ball_image_x/y: pixel position of the ball in the image.
    Providing this gives the homography solver an additional anchor and
    significantly improves accuracy.
    Returns player positions, team assignments, and formation matches.
    """
    t0 = time.time()

    # ── 1. Read image ──
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG or PNG.")

    # ── 2. Run the full CV pipeline ──
    ball_image_pos = None
    if ball_image_x is not None and ball_image_y is not None:
        ball_image_pos = (int(ball_image_x), int(ball_image_y))

    try:
        field_img, overlay, corr_debug, warped, summary, field_players, team_labels = (
            run_interactive_homography(
                img_bgr,
                ball_yard,
                ball_image_pos=ball_image_pos,
                conf=conf,
                field_type=field_type,
                offense_direction=offense_direction,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CV pipeline error: {str(e)}")

    # ── 3. Build playbook data (matches cvAnnotator.js expected input) ──
    _TEAM_MAP = {0: "offense", 1: "defense", -1: "unknown"}
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

    playbook_data = {
        "ball_yard_line": int(ball_yard),
        "offense_direction": offense_direction,
        "field_type": field_type,
        "player_count": len(players_out),
        "players": players_out,
    }

    # ── 4. Run formation matching (NFL template enrichment) ──
    formation_matches = None
    if TEMPLATES:
        try:
            fm_result = match_formation(
                field_players, TEMPLATES, ball_yard=float(ball_yard)
            )
            # Serialize — keep top 3 matches, strip non-JSON-safe values
            def _clean_match(m):
                return {
                    "formation": m.get("formation", "unknown"),
                    "score": round(float(m.get("score", 999)), 3),
                    "matched": m.get("matched", 0),
                    "detected": m.get("detected", 0),
                    "play_count": m.get("play_count", 0),
                }

            formation_matches = {
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
            # Formation matching is optional — don't fail the whole request
            formation_matches = {"error": str(e)}

    # ── 5. Return everything ──
    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "playbook_data": playbook_data,
        "formation_matches": formation_matches,
        "pipeline_meta": {
            "players_on_field": len(players_out),
            "processing_time_ms": elapsed_ms,
            "summary": summary,
        },
    }


# ── Film Upload & Batch Analysis ─────────────────────────────────────────────

@app.post("/upload-film")
async def upload_film(
    video: UploadFile = File(...),
    offense_direction: str = Form("right"),
    field_type: str = Form("college"),
):
    """Upload a video file and start the full analysis pipeline.

    Returns a job_id that can be polled via /job-status/{id}.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Must be a video file (MP4, MOV, etc.)")

    job_id = create_job()
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"

    # Stream in 1 MB chunks to handle large files
    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    # Start background processing
    thread = threading.Thread(
        target=run_film_pipeline,
        args=(job_id, str(video_path), field_type, offense_direction, TEMPLATES),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll the status of a film analysis job (lightweight, no play data)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_plays": job.total_plays,
        "analyzed_count": job.analyzed_count,
        "current_step": job.current_step,
        "error": job.error,
    }


@app.get("/job-results/{job_id}")
def job_results(job_id: str):
    """Get the full results of a completed film analysis job.

    Returns all analyzed plays with base64 screenshots and CV data.
    Only call when job status is 'complete'.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("complete", "error"):
        raise HTTPException(400, f"Job not ready (status: {job.status})")

    return {
        "job_id": job.job_id,
        "total_plays": len(job.plays),
        "plays": [
            {
                "play_number": p.play_number,
                "frame_number": p.frame_number,
                "timestamp": p.timestamp,
                "confidence": p.confidence,
                "screenshot_b64": p.screenshot_b64,
                "ball_yard": p.ball_yard,
                "ball_yard_source": p.ball_yard_source,
                "playbook_data": p.playbook_data,
                "formation_matches": p.formation_matches,
                "pipeline_meta": p.pipeline_meta,
                "error": p.error,
            }
            for p in job.plays
        ],
    }
