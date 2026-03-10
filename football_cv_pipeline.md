# Football Player Identification via Computer Vision

**Sideline Camera · American Football · Batch Processing**

*March 2026*

---

## 1. The Problem

The goal is to take a single image captured from a sideline camera during an American football game and identify every player visible in the frame. Identification here means three things: detecting that a person in the image is a player, determining which team they belong to, and reading their jersey number so we can map them to a specific roster entry.

This is a fundamentally harder problem than the equivalent task in soccer or basketball for several reasons that are specific to the sport and the camera angle.

### Why American Football is Hard for CV

**Heavy occlusion at the line of scrimmage.** Offensive and defensive linemen cluster within inches of each other at the snap. From a sideline angle, bodies stack and overlap in ways that make individual detection extremely difficult. A standard object detector trained on COCO sees these clusters as a single person-shaped blob rather than five or six distinct athletes.

**Helmets and pads distort body geometry.** Most pose estimation and person detection models are trained on images of people in normal clothing. Shoulder pads widen the torso, helmets obscure the head-neck relationship, and the overall silhouette is bulkier and less human-shaped than what these models expect. This reduces detection confidence and increases false negatives.

**Jersey numbers are partially visible.** Numbers appear on the chest, shoulders, and back. From a sideline camera, you typically only see one of these surfaces at a time, and it is often at an oblique angle, partially obscured by an arm or another player, or motion-blurred mid-play. Single-frame OCR accuracy on football jersey numbers is estimated at 40–50% in real conditions.

**Two distinct regimes: pre-snap and post-snap.** Before the snap, players are mostly stationary and well-separated. After the snap, the scene becomes chaotic, with rapid motion, collisions, and pile-ups. A pipeline that works well pre-snap may fail entirely post-snap. Any robust solution needs to handle both regimes.

**Visual similarity between teams and officials.** Away teams often wear white jerseys, and referees also wear white-and-black striped shirts. Sideline staff, coaches, and chain crew members are also in the frame from a sideline angle. The system needs to distinguish players from all of these other people reliably.

---

## 2. Input Constraints & Assumptions

This pipeline is designed around a specific set of constraints that simplify some aspects and complicate others.

| Constraint | Implication |
|---|---|
| **Sideline camera** | Players are relatively large in frame but occlude each other laterally. Perspective distortion increases with distance from camera. |
| **American football** | Padded body shapes, helmet occlusion, extreme clustering at the line. Jersey numbers on chest, shoulders, and back. |
| **Single image input** | No temporal information available. Cannot use tracking or multi-frame fusion for this initial version. Every inference must succeed on one frame. |
| **Batch processing** | No real-time constraint. We can use heavier models, run multiple inference passes, and apply expensive post-processing without worrying about latency. |

---

## 3. Pipeline Architecture

The solution decomposes the identification problem into three sequential stages, each building on the output of the previous one. This modular approach lets us develop, test, and improve each component independently.

**The full pipeline flow is: Raw Image → Player Detection → Team Classification → Jersey Number Recognition**

### Phase 1: Player Detection

**Objective:** Produce a bounding box around every player in the image, excluding referees, coaches, and sideline personnel.

#### Model Selection

YOLOv8m (medium) is the recommended starting point. YOLO models are single-stage detectors that run a full image through a single neural network pass, producing bounding boxes and class predictions simultaneously. The medium variant balances accuracy against inference cost, and since we are doing batch processing, we can afford the slightly heavier model over the small or nano variants.

A COCO-pretrained YOLOv8m will detect the generic person class out of the box, which gives us an immediate baseline. However, COCO training data consists overwhelmingly of people in normal clothing, so the model will underperform on padded football players, especially in tight formations. Fine-tuning on football-specific data is essential.

#### Training Data

The Roboflow Open Source community hosts several American football detection datasets with bounding box annotations. These are annotated for classes like player, referee, and ball. The most practical approach is to combine two or three of these datasets, remap their class labels to a unified schema, and fine-tune YOLOv8m for approximately 50–100 epochs. A training set of 2,000–5,000 annotated images is sufficient for strong performance.

An alternative is to start with COCO person detections and train a secondary classifier on the crops to separate players from non-players, but end-to-end fine-tuning produces better results because the detector learns football-specific spatial priors (players tend to be on the field, in specific formations, etc.).

#### Expected Challenges

- Lineman clusters will produce merged or missed detections. Mitigation: lower the NMS IoU threshold to allow more overlapping boxes, and augment training data with tight-crop examples of the line of scrimmage.
- Players far from the camera (opposite sideline) will be small and low-resolution. Mitigation: use SAHI (Slicing Aided Hyper Inference) which tiles the image into overlapping patches, runs detection on each patch, and merges results.
- Sideline personnel near the field edge will trigger false positives. Mitigation: if field boundaries can be estimated (even roughly), filter detections that fall outside the playing surface.

### Phase 2: Team Classification

**Objective:** Given a set of player crops from Phase 1, assign each player to Team A, Team B, or Other (referee/staff that slipped through detection).

#### Approach: Color-Based Clustering

The most practical approach for a single-image pipeline is unsupervised color clustering. Each detected player crop is processed as follows:

1. **Convert the crop to HSV color space.** HSV separates color information (hue) from brightness (value), making the method robust to shadows and lighting variation.
2. **Mask out the green field.** Apply a threshold on the hue channel to remove any grass pixels bleeding into the bounding box. This is critical because green dominates the image and would otherwise skew clustering.
3. **Extract the shoulder pad region.** Crop the upper 40% of the bounding box to focus on the most reliably team-colored area. The lower portion often includes legs, which are covered by uniform pants that may not match the jersey color.
4. **Compute the dominant color.** Use K-Means with k=3 on the remaining pixels in the shoulder region (to capture jersey, skin, and any secondary uniform element). Take the largest cluster that is not skin-toned as the dominant jersey color.
5. **Cluster all players globally.** Collect the dominant colors from all detected players and run K-Means with k=2 or k=3 across the full image. Each cluster corresponds to a team or referees.

This method requires no labeled data and works across any matchup. Its main weakness is that it fails when both teams have similar jersey colors (e.g., dark navy vs. dark green under stadium lighting). For those edge cases, a fine-tuned ResNet-18 classifier trained on labeled team crops would be more robust, but that requires per-game or per-team training data.

### Phase 3: Jersey Number Recognition

**Objective:** Read the jersey number from each player crop to map the detection to a specific roster entry.

#### Why This is the Hardest Stage

Jersey number recognition on American football players from a sideline camera is arguably the most technically challenging part of this pipeline. The numbers are large on the jersey but the viewing conditions are hostile: oblique angles mean the digits are perspectively distorted, arms cross over chest numbers during motion, helmets occlude shoulder numbers, and motion blur smears the digits during play. Single-frame OCR accuracy in these conditions is low enough that a naive approach will produce more noise than signal.

#### Approach: OCR with Confidence Filtering

The recommended approach for single-image inference is:

1. Crop the torso region from each player detection (middle 50% vertically, centered horizontally).
2. Run PaddleOCR or EasyOCR on the crop. Both are open-source OCR engines that work well on scene text. PaddleOCR tends to perform slightly better on short numeric strings.
3. Filter by confidence threshold. Only accept readings above a tuned threshold (start at 0.7, adjust based on validation). Low-confidence reads are marked as unidentified rather than guessed.
4. Validate against roster. If the OCR returns a number that does not exist on either team's roster, reject it. This simple post-processing step eliminates a significant fraction of misreads.

#### Future Enhancement: Temporal Fusion

When the pipeline eventually moves to video, the accuracy problem largely solves itself through temporal fusion. A tracker like ByteTrack maintains consistent identity across frames, and OCR results can be aggregated over the entire track. A player whose number is unreadable in 20 frames but clear in 3 frames can still be identified with high confidence through majority voting. This is the single biggest accuracy lever for jersey number recognition and is the primary motivation for eventually extending to video.

---

## 4. Model Comparison

The following table summarizes the key model options for each pipeline stage and their trade-offs in the context of this specific problem.

| Stage | Model | Strengths | Weaknesses |
|---|---|---|---|
| Detection | YOLOv8m | Fast, MPS-compatible, proven on sports data | Struggles with tight clusters without fine-tuning |
| Detection | RT-DETR | Transformer attention handles occlusion better | Heavier inference, no MPS support |
| Detection | YOLOv8m + SAHI | Recovers small/distant players via tiled inference | 2–4x slower due to multiple passes |
| Team ID | HSV + K-Means | No labeled data needed, works across any matchup | Fails on similar jersey colors |
| Team ID | ResNet-18 classifier | More robust to color ambiguity | Needs labeled crops per team/game |
| Jersey OCR | PaddleOCR | Strong on short numeric strings, well-maintained | Low accuracy on oblique/blurred text |
| Jersey OCR | EasyOCR | Simple API, decent baseline | Slightly worse than PaddleOCR on digits |
| Jersey OCR | Custom digit CNN | Can be trained on synthetic football jersey data | Requires training pipeline and data generation |

---

## 5. Future Extension: Bird's-Eye View Mapping

The eventual goal is to project detected player positions from the image plane onto a standardized overhead view of the football field. This requires estimating a homography matrix that maps pixel coordinates in the camera image to real-world field coordinates.

American football fields are actually well-suited for homography estimation because they are dense with known reference geometry:

- Yard lines every 5 yards (15 feet apart), spanning the full 53⅓-yard width
- Hash marks at known lateral positions
- Yard numbers painted at fixed locations
- Sideline and end zone boundaries with known dimensions

The workflow for homography estimation is: detect field lines and markings using semantic segmentation or Hough line detection, match detected features to known field geometry, compute the homography from four or more point correspondences, and then project each player's foot position through the matrix to get field coordinates.

The main complication for sideline cameras is that panning and zooming change the camera-to-field relationship continuously throughout the broadcast. For single-image use this is not an issue, since the homography only needs to be estimated once per frame. For video, the homography would need to be re-estimated per frame or tracked incrementally using camera motion estimation. This is a well-studied problem in sports broadcast analysis with established solutions.

---

## 6. Recommended Implementation Path

Given the constraints (single image, sideline camera, batch processing, M3 MacBook Air for development), the recommended build order is:

1. **Baseline detection with COCO-pretrained YOLOv8m.** Run inference on sample football images using the generic person class. Evaluate how many players are detected versus missed. This takes minutes and gives an immediate sense of the gap.
2. **Fine-tune YOLOv8m on football data.** Download and merge Roboflow football datasets, retrain for 50–100 epochs. Compare against the COCO baseline on the same test images. This is the single highest-impact step.
3. **Implement HSV color clustering for team assignment.** Process each detected crop through the color extraction pipeline described above. Validate visually on a handful of images with known team colors.
4. **Add PaddleOCR for jersey numbers.** Run OCR on torso crops, filter by confidence, validate against roster. Measure accuracy honestly and accept that single-frame OCR will have significant gaps.
5. **Extend to video with tracking and temporal fusion.** Add ByteTrack for multi-frame identity persistence and aggregate OCR reads over time. This is where accuracy will jump substantially.
6. **Add homography for bird's-eye view.** Build the field line detection and homography estimation module. Project player positions onto the overhead field template.

Each step produces a testable, demonstrable result. The key insight is that the pipeline is only as good as its detection stage, so investing in high-quality fine-tuned detection before worrying about OCR accuracy or bird's-eye mapping is the correct priority order.
