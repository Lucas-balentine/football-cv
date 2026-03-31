# Football Player Identification via Computer Vision

A computer vision pipeline for identifying American football players from sideline camera images. Given a single frame, the system detects players, classifies them by team, and reads jersey numbers.

## Pipeline

1. **Player Detection** -- YOLOv8m fine-tuned on football data with SAHI tiled inference for small/distant players
2. **Team Classification** -- Unsupervised HSV color clustering on jersey crops (upper 40% of each bounding box, green-masked)
3. **Jersey Number Recognition** -- PaddleOCR on torso crops with confidence filtering and roster validation

See [`football_cv_pipeline.md`](football_cv_pipeline.md) for the full design document.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will also need a `.env` file with your Roboflow API key:

```
ROBOFLOW_API_KEY=your_key_here
```

## Usage

**Detect players in an image:**

```bash
python detect_players.py --image steelers.jpg
```

**Run the Gradio web app:**

```bash
python app.py
```

**Train/fine-tune the detection model:**

```bash
python train.py
```

## Project Structure

```
detect_players.py          # Baseline YOLOv8 player detection
team_classifier.py         # HSV color clustering for team assignment
field_homography.py        # Homography estimation for bird's-eye view
field_markings.py          # Field line and marking detection
field_numbers.py           # Yard number detection
interactive_homography.py  # Interactive homography calibration tool
build_formations.py        # Formation analysis from player positions
playbook_renderer.py       # Render detected formations as playbook diagrams
app.py                     # Gradio web interface
train.py                   # YOLOv8 fine-tuning script
train_hash.py              # Hash mark detection model training
formation_templates.json   # Reference formation templates
football_cv_pipeline.md    # Full design document
```

## Development

- **Target hardware:** M3 MacBook Air (MPS backend)
- **Batch processing only** -- no real-time constraints
- Models and training artifacts are excluded from version control via `.gitignore`
