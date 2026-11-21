# 02 — Technical Requirements Document (TRD)

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Purpose

Defines the technical stack, hardware, performance targets, interfaces and constraints needed to build the requirements in the PRD.

## 2. Hardware Requirements

| Item | Development target |
|------|--------------------|
| GPU | NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM (Ada Lovelace, supports FP16 / mixed precision) |
| CPU | 6+ cores recommended |
| RAM | 16 GB recommended |
| Storage | SSD, 100 GB+ free for clips, evidence and datasets |
| Input | Webcam, RTSP IP camera, or recorded video files |

Note: the reference system reports results on an Apple MPS GPU. This build targets CUDA on RTX 4050 (6 GB), so figures must be re-measured on the target hardware.

## 3. Software Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ (backend, ML), TypeScript (frontend) |
| Deep learning | PyTorch with CUDA |
| Object detection | YOLO26 (Ultralytics). Fallback to an earlier Ultralytics YOLO release if YOLO26 weights are unavailable |
| Tracking | ByteTrack (Ultralytics built-in tracker or `supervision`) |
| Face recognition | InsightFace (detector + ArcFace embeddings), pretrained |
| Activity model | Pretrained CNN backbone (MobileNetV2 or ResNet18) + LSTM head |
| Video I/O | OpenCV |
| Backend API | FastAPI with Uvicorn |
| Real-time push | WebSocket |
| Database | SQLite for development, PostgreSQL for deployment |
| Frontend | React with a charting/UI library |
| Video streaming to UI | MJPEG stream (simple) or WebRTC (optional) |
| Notifications | Dashboard alerts, optional Telegram bot or email |
| Config | YAML files |
| Packaging | Docker (optional, for deployment) |

## 4. Model Requirements

| Model | Source | Training needed |
|-------|--------|-----------------|
| YOLO26 (small as default, nano as fallback) | Pretrained COCO weights | No (optional fine-tune later) |
| ByteTrack | Algorithm | No |
| Face recognition | Pretrained InsightFace | No (enrolment only) |
| CNN-LSTM | Frozen pretrained CNN + trained LSTM head | Yes, light fine-tune |

## 5. Performance Requirements

| Metric | Target |
|--------|--------|
| Frame rate | 18–22 FPS per stream |
| Processing time per frame | 20–45 ms |
| Detection input size | 640 px (resize from 1920×1080 source) |
| LSTM clip length | 16 frames |
| Alert latency | Within a few seconds of event |
| VRAM budget | Must fit in 6 GB total, with about 1 GB kept free for the display and system |

## 6. GPU Optimisation Rules (RTX 4050, 6 GB)

- Use the YOLO small variant by default; switch to nano if FPS drops below target. Medium is possible for a single stream but leaves little headroom.
- Resize frames to 640 px before inference.
- Enable mixed precision (FP16 / AMP) for inference and training.
- Training batch size 8–16 for the LSTM head (cached features allow larger).
- Freeze the CNN backbone; train only the LSTM head.
- Run face recognition only on newly seen tracks and periodically, not on every frame.
- Run the activity model on a per-track sliding window, not on every frame.
- Process every N-th frame if the target FPS is not met.

## 7. Processing Pipeline Requirements

1. Capture thread reads frames into a bounded queue (drop oldest when full).
2. Inference worker runs detection then tracking.
3. Face worker handles person crops asynchronously.
4. Activity worker consumes 16-frame track windows asynchronously.
5. Rule engine evaluates zones, lines, dwell and occupancy per frame.
6. Threat engine consumes events and produces incidents.
7. Evidence writer saves images and video clips.
8. API layer pushes updates to the dashboard.

## 8. Interfaces

| Interface | Type | Description |
|-----------|------|-------------|
| Camera input | RTSP / file / webcam | Frames in |
| REST API | HTTP/JSON | Cameras, zones, persons, incidents, evidence |
| WebSocket | JSON messages | Live detections, status and new incidents |
| Evidence files | File system | Images and MP4 clips referenced from database |
| Notification channel | Bot / email | Optional external alerts |

## 9. Configuration Requirements

All of the following must be editable through config or the admin UI:

- Camera list and sources
- Zone polygons, line coordinates, occupancy limit (default example: 4)
- Dwell-time and loitering thresholds
- Abandoned-object time threshold
- Correlation time window
- Threat weights and level cut-offs
- Detection confidence threshold
- Face-match threshold

## 10. Logging and Monitoring

- Structured logs per module (capture, detector, tracker, face, activity, threat, api).
- Per-camera runtime metrics: people count, FPS, processing time (ms), alert state.
- Errors from one camera must not stop others.

## 11. Constraints and Dependencies

- Single laptop GPU with 6 GB VRAM shared across detector, face model and activity model. Laptop thermal limits can reduce sustained speed, so test on power and with cooling.
- Quality of face recognition depends on camera angle and resolution.
- Activity accuracy depends on the fine-tuning dataset and must be validated with labelled ground truth.
- Models must run locally; video is not required to leave the machine.

## 12. Technical Risks

| Risk | Mitigation |
|------|------------|
| FPS below target on RTX 4050 | Smaller model, frame skipping, lower input size, async workers |
| ID switches in crowded scenes | Tune ByteTrack thresholds, use track buffer |
| False alerts from activity model | Require rule or temporal confirmation before raising event |
| Face recognition misses at distance | Use face crops only above minimum size, re-check over time |
| YOLO26 package availability | Keep detector wrapper model-agnostic |
