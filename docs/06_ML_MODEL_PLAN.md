# 06 — ML Model Plan

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Strategy

Use pretrained models wherever possible. Only the activity recognition model is fine-tuned. This fits a single RTX 4050 (6 GB VRAM) and avoids training from scratch.

| Model | Approach |
|-------|----------|
| Object detection (YOLO26) | Pretrained COCO weights, use directly |
| Tracking (ByteTrack) | Algorithm, no training |
| Face recognition | Pretrained InsightFace / ArcFace, enrolment only |
| Activity (CNN-LSTM) | Frozen pretrained CNN + trained LSTM head |

## 2. Object Detection

- Model: YOLO26 small (default). Use nano if FPS is below target, medium only for single-stream tests.
- Classes used: person, backpack, handbag, suitcase, car, motorcycle, bus, truck, bicycle.
- Input size 640 px, FP16 inference.
- Confidence threshold default 0.4, NMS settings default.
- Output per detection: class, bounding box, confidence.
- Optional later: fine-tune on site-specific footage if abandoned-object detection is weak.

## 3. Object Tracking

- ByteTrack uses both high- and low-confidence detections to keep identities stable.
- Parameters to tune: track threshold, match threshold, track buffer (frames kept for lost tracks).
- Outputs: Track ID, path, first/last seen, dwell time.

## 4. Face Recognition

**Enrolment:**
1. Administrator uploads 3–5 face images per person.
2. Detector finds the face; ArcFace produces an embedding.
3. Embeddings are stored with person ID and name.

**Recognition:**
1. For each new person track, crop the person region and detect the face.
2. Skip faces smaller than a minimum size or with poor quality.
3. Compute embedding and compare with stored embeddings by cosine similarity.
4. Above threshold → Known (person ID). Below → Unknown.
5. Re-check periodically while the track lives and keep the best-scoring result.

Parameters: similarity threshold, minimum face size, re-check interval.

## 5. Activity Recognition (CNN-LSTM)

### 5.1 Target classes

Fighting, running, loitering, normal. (Absence of object is handled by the rule engine.)

### 5.2 Architecture

```
16 frames (224×224) → Pretrained CNN (frozen) → 16 feature vectors
        → LSTM (1–2 layers, 128–256 hidden) → Dropout → Dense → Softmax
```

- CNN backbone: MobileNetV2 (lightest) or ResNet18. With 6 GB VRAM, ResNet50 is also possible if accuracy needs it.
- Only LSTM and classifier are trained.
- Optional: extract CNN features once, cache to disk, and train the LSTM on cached features for very fast training.

### 5.3 Datasets

| Dataset | Use |
|---------|-----|
| RWF-2000 | Fight vs non-fight |
| UCF-Crime (subset) | Additional abnormal activity |
| HMDB51 (Run class) / VIRAT | Running activity recognition (public source) |
| CAVIAR / ABODA | Baseline normal activity, dwell, and abandoned object |

> **Datasets and Data Handling Note (Phase 0):**
> All training and testing footage is sourced exclusively from public academic datasets (no self-recorded videos). The `running` class is sourced from public action datasets (e.g., HMDB51 CC-BY 4.0 or VIRAT). Spatial rules (restricted zones, virtual lines, occupancy limits) are applied as configuration overlays (`configs/zones.yaml`) on public footage of people. All test clips in `data/samples/` are strictly quarantined from training/validation sets to prevent data leakage (enforced by `scripts/check_leakage.py`).

A subset of a few hundred clips is enough for a first version.

### 5.4 Training setup

| Item | Value |
|------|-------|
| Clip length | 16 frames, sampled uniformly |
| Frame size | 224×224 |
| Batch size | 8–16 (use the larger value with cached features) |
| Precision | Mixed precision (AMP) |
| Optimiser | Adam, learning rate 1e-3 for LSTM head |
| Loss | Cross-entropy (class weights if imbalanced) |
| Augmentation | Horizontal flip, brightness/contrast, small crop |
| Split | 70% train, 15% validation, 15% test, split by video not by clip |
| Early stopping | On validation loss |

### 5.5 Evaluation

The reference work notes that activity classification against labelled ground truth is still to be done. This project completes it.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct |
| Precision, Recall, F1 | Per class |
| Confusion matrix | Class confusion |
| Inference time | Per clip on RTX 4050 |

### 5.5.1 Acceptance

Report metrics on the held-out test split and on a small set of self-recorded clips from the deployment-like scene.

### 5.6 Deployment

- Run per tracked person on a sliding window of 16 frames (stride configurable).
- Raise a FIGHTING / RUNNING / LOITERING event only when confidence exceeds threshold for several consecutive windows.

## 6. Whole-Pipeline Evaluation

| Test | Measure |
|------|---------|
| Detection | Person detection accuracy on sample footage |
| Tracking | ID switch count, track continuity |
| Face | True/false match rate on enrolled persons |
| Rules | Correct triggering of zone, line, dwell, occupancy, abandoned-object events |
| Correlation | Share of related events merged correctly into incidents |
| Runtime | FPS and processing ms per frame per camera (as in the dashboard runtime table) |
| Alert load | Number of alerts versus a one-alert-per-event baseline |

## 7. Model Management

- Store weights under a `models/` folder with version names.
- Record model version used in every incident for traceability.
- Keep detector, face and activity modules behind interfaces so any model can be swapped.
