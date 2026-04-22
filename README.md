# TendencyIQ — Football CV Pipeline

End-to-end computer vision system that turns broadcast football footage into structured play data. Upload a game video, the pipeline automatically extracts pre-snap frames, maps them to field coordinates, identifies players by team, and produces a bird's-eye playbook view ready for tendency analysis.

Built to replicate what services like PlayVision and Hudl Sideline do — automatic film breakdown from raw video — on a single laptop with consumer APIs.

---

## What it does

Given an MP4 of a football game:

1. **Scans the video** and automatically extracts one pre-snap frame per play (~80 frames from a 40-minute game)
2. **Filters to clean sideline views** — drops replays, endzone cams, and scoreboard shots
3. **Detects field geometry** — yard lines, hash marks, sideline boundaries
4. **Computes a homography** mapping broadcast pixels → real field yard coordinates
5. **Segments every player** on the field (SAM-based segmentation via Roboflow)
6. **Classifies teams** — offense vs defense using jersey color + position relative to line of scrimmage
7. **Renders a playbook view** — clean top-down diagram with player positions, ball location, and yard markers

---

## Screenshots

| Stage | Output |
|---|---|
| Hash-mark detection | YOLO finds every hash × yard-line intersection |
| Field overlay | Back-projected yard lines + hash ticks match painted field |
| Segmentation | Per-player polygon masks from SAM |
| Playbook view | Offense (○), QB (●), defense (✕), ball position |

---

## Tech stack

**Computer vision**
- OpenCV (homography, perspective warping, HSV masking, contour analysis)
- YOLOv8/v11 via Ultralytics (player detection, hash-mark intersection model)
- Roboflow Serverless Inference (workflow orchestration, SAM segmentation)
- Custom geometric reasoning — grid-vector estimation, perspective-aware clustering

**Machine learning**
- Fine-tuned YOLO on custom hash-yards-intersection dataset
- K-Means clustering for unsupervised team color separation
- HSV circular-mean color profiling with white-ratio fallback
- Fused position + color signals with confidence scoring

**Video processing**
- Single-pass sequential decoding (10–50× faster than random seek)
- Histogram-based scene-cut detection
- Motion-valley detection via `cv2.absdiff` + `scipy.signal.find_peaks`
- Formation-score validation (blob detection on field-masked regions)

**Application layer**
- Gradio web UI (12-tab interactive tool for frame inspection and pipeline testing)
- FastAPI backend (`api.py`) for programmatic access
- React + Vite frontend (separate repo: [tendency-iq](https://github.com/Lucas-balentine/tendency-iq))
- Supabase for play-data persistence

---

## Architecture

```
  MP4 video
     │
     ▼
 ┌───────────────────────┐
 │  presnap_extractor.py │   Scene cuts + motion valleys + sideline filter
 └──────────┬────────────┘   → 1 clean pre-snap PNG per play
            │
            ▼
 ┌───────────────────────────┐
 │ interactive_homography.py │
 │                           │
 │  1. Hash-intersection     │   YOLO via Roboflow → (x,y,yard) per hash
 │     detection             │
 │  2. Grid estimation       │   vec_along (hash pair), vec_across (5yd)
 │  3. Group clustering      │   Project onto perpendicular axis
 │  4. Yard assignment       │   Ball click anchors group index
 │  5. Homography solve      │   findHomography w/ flip enumeration
 │  6. Player segmentation   │   SAM workflow → polygon masks
 │  7. Team classification   │   Position + HSV color fusion
 │  8. Playbook rendering    │   Back-project to template
 └───────────────────────────┘
            │
            ▼
      Playbook JSON
   { ball_yard, offense_direction, players: [...] }
```

---

## Key engineering choices

**Why a custom homography instead of off-the-shelf sports-CV libraries?**
Commercial tools assume NFL/NCAA broadcast quality. High school sideline footage has oblique angles, missing hash marks, and painted logos that break standard approaches. The pipeline handles each problem explicitly:
- **Missing hash partners** — post-clustering merge rejoins split pairs; conditional singles check drops unreliable points when they hurt the fit
- **Painted yard numbers** — removed from the Hough fallback after they were corrupting line detection
- **Camera direction ambiguity** — `offense_direction` parameter + per-image flip enumeration pick the right near/far hash assignment

**Why extract one frame per play instead of tracking?**
Tendency analysis only needs formation + personnel, not motion. Single-frame extraction is 100× cheaper than full tracking and delivers what scouts actually use.

**Why segmentation instead of just bounding boxes?**
Jersey color classification needs clean pixel masks to avoid grass bleeding into the color signal. SAM-based segmentation cleanly isolates player torsos for HSV sampling.

---

## Running locally

### 1. Install

```bash
git clone https://github.com/Lucas-balentine/football-cv
cd football-cv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API access

Create `.env`:

```
ROBOFLOW_API_KEY=your_key_here
```

You'll need:
- A Roboflow account with the `hash-yards-intersection` model and a segmentation workflow in your workspace
- (Optional) `OPENAI_API_KEY` if you want to experiment with the vision-model fallback

### 3. Launch

**Interactive UI (Gradio):**
```bash
python app.py
# http://127.0.0.1:7860
```

**API backend (FastAPI):**
```bash
uvicorn api:app --reload
# http://127.0.0.1:8000/docs
```

**Process a full video end-to-end:**
```bash
python -c "from presnap_extractor import extract_presnap_frames; extract_presnap_frames('game.mp4', 'output/')"
```

---

## Project structure

```
football-cv/
├── app.py                     # Gradio UI — 12 tabs for each pipeline stage
├── api.py                     # FastAPI server — programmatic access
├── interactive_homography.py  # Core pipeline: detection → homography → projection
├── presnap_extractor.py       # Video → pre-snap frame extraction
├── team_classifier.py         # Jersey color + position team assignment
├── playbook_renderer.py       # Top-down playbook visualization
├── field_homography.py        # Template constants and field geometry
├── field_markings.py          # Hough-based yard line detection (legacy)
├── field_numbers.py           # Painted yard number recognition
├── detect_players.py          # Field mask + player bbox utilities
├── build_formations.py        # Formation template matching
├── film_pipeline.py           # End-to-end batch processing
├── train.py                   # YOLO fine-tuning on football data
└── train_hash.py              # Hash-intersection model training
```

---

## Skills demonstrated

- **Computer vision**: projective geometry, homography estimation, perspective correction, color space analysis
- **Deep learning**: YOLO fine-tuning, transfer learning, multi-class detection, SAM/foundation models
- **Video processing**: efficient seeking, scene segmentation, motion analysis, temporal filtering
- **ML engineering**: graceful degradation, confidence thresholding, ensemble signal fusion, unsupervised clustering
- **Software engineering**: separation of concerns across focused modules, API design (FastAPI + Gradio), error handling, environment management
- **Problem-solving**: debugged a bias-amplification issue in sparse-point homography; designed a conditional-refinement algorithm that uses paired detections to validate and discard unreliable single-point correspondences
- **Domain expertise**: NCAA vs NFL hash positions, field coordinate systems, offensive/defensive formation taxonomy

---

## Status

**Working**: pre-snap extraction, hash detection, homography, segmentation, team classification, playbook rendering.
**In progress**: tightening homography tilt under oblique camera angles, role labeling (QB/WR/RB vs current offense/defense binary), multi-game tendency analysis.

---

## License

Private — portfolio use.

## Author

Lucas Balentine — [GitHub](https://github.com/Lucas-balentine) · [LinkedIn](https://linkedin.com/in/lucasbalentine)
