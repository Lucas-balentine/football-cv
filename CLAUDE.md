# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a computer vision pipeline for identifying American football players from sideline camera images. It is currently in the design/planning phase — `football_cv_pipeline.md` is the primary design document. Sample images (`steelers.jpg`, `texas.jpg`) are included for testing.

## Pipeline Architecture

The system has three sequential stages:

1. **Player Detection** — YOLOv8m (fine-tuned on football data) produces bounding boxes for players, excluding refs/coaches. SAHI (tiled inference) handles distant/small players.
2. **Team Classification** — Unsupervised HSV color clustering on player crops. Extracts dominant jersey color from the upper 40% of each crop after masking out green field pixels. K-Means (k=2 or k=3) groups players by team.
3. **Jersey Number Recognition** — PaddleOCR on torso crops with confidence filtering (threshold ~0.7) and roster validation. Single-frame accuracy is low (~40-50%); temporal fusion via video tracking (ByteTrack) is the planned solution.

## Key Technical Decisions

- **Batch processing only** — no real-time constraints; heavier models are acceptable
- **Single-image input** for initial version (no tracking/temporal fusion yet)
- **Development target**: M3 MacBook Air (MPS-compatible models preferred; RT-DETR lacks MPS support)
- **Training data**: Roboflow open-source football datasets, 2,000-5,000 annotated images, 50-100 epochs fine-tuning

## Implementation Priority Order

1. Baseline detection with COCO-pretrained YOLOv8m
2. Fine-tune YOLOv8m on football-specific data
3. HSV color clustering for team assignment
4. PaddleOCR for jersey numbers
5. Video extension with ByteTrack tracking + temporal OCR fusion
6. Homography estimation for bird's-eye view field mapping

## Known Hard Problems

- Lineman clusters cause merged/missed detections (lower NMS IoU threshold helps)
- Similar jersey colors between teams break color clustering (ResNet-18 classifier is the fallback)
- Oblique angles, arm occlusion, and motion blur severely degrade OCR accuracy
