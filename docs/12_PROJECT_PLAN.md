# 12 — Project Plan

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Approach

Build the core pipeline first, then layer features on top. Each phase ends with a working demo.

## 2. Phases

### Phase 0 — Setup
- Python environment with PyTorch CUDA, OpenCV, Ultralytics, InsightFace, FastAPI.
- Project repository and folder structure.
- Sample videos collected.
- Verify GPU is used.

**Done when:** a sample video plays through OpenCV and a pretrained YOLO runs on the GPU.

### Phase 1 — Detection and Tracking
- Frame reader (file, webcam, RTSP).
- YOLO26 detection of person, bag, vehicle.
- ByteTrack integration with stable Track IDs.
- Annotated output with boxes, confidence and IDs.
- FPS and processing-time measurement.

**Done when:** tracked IDs persist and 18+ FPS is measured.

### Phase 2 — Rules and Events
- Zone and line definitions (config file first).
- Restricted entry, line crossing, dwell time, occupancy, abandoned-object logic.
- Event objects generated and logged.

**Done when:** each rule fires correctly on test clips.

### Phase 3 — Face Recognition
- Enrolment script and embeddings store.
- Known/Unknown assignment to Track ID.
- Re-check and best-match logic.

**Done when:** enrolled people are recognised, others are Unknown.

### Phase 4 — Threat Engine and Incidents
- Event normalisation, correlation by keys and time window.
- Scoring and levels.
- Incident creation and update.
- Database tables.
- Evidence collector (images and clip from frame buffer).

**Done when:** the multi-event scenario yields one incident with complete evidence.

### Phase 5 — Backend API
- FastAPI endpoints, authentication and roles.
- WebSocket live updates.
- MJPEG stream of annotated video.

**Done when:** all API spec endpoints work and are tested.

### Phase 6 — Dashboard
- Live Monitor, Incident List and Detail, Cameras, Zones editor, Persons, Settings, Users.
- Alert presentation.

**Done when:** an officer can monitor, review evidence and update status in the browser.

### Phase 7 — Activity Recognition (CNN-LSTM)
- Dataset preparation (RWF-2000 subset plus self-recorded).
- Feature extraction with frozen CNN.
- LSTM training with AMP on RTX 4050.
- Evaluation with accuracy, precision, recall, F1 and confusion matrix.
- Integrate into pipeline as asynchronous worker.

**Done when:** activity events feed the threat engine and metrics are reported.

### Phase 8 — Notifications and Hardening
- Notification channel for High and Critical incidents.
- Retention job, logging, error handling.
- Performance tuning (frame skipping, thresholds).

### Phase 9 — Testing, Evaluation and Documentation
- Execute test plan.
- Record runtime table per camera and alert-reduction comparison.
- Final report, demo video and user guide.

## 3. Milestones

| Milestone | Outcome |
|-----------|---------|
| M1 | Detection + tracking demo |
| M2 | Rule events demo |
| M3 | Face recognition demo |
| M4 | First correlated incident with evidence |
| M5 | Full dashboard with live alerts |
| M6 | Activity recognition evaluated and integrated |
| M7 | Tested, documented, demo-ready |

## 4. Suggested Folder Structure

```
intelligent-cctv-surveillance/
├── docs/                      # these documents
├── configs/
│   ├── cameras.yaml
│   ├── zones.yaml
│   └── threat.yaml
├── models/                    # weights, versioned
├── data/
│   ├── known_persons/
│   ├── datasets/
│   └── samples/
├── src/
│   ├── capture/               # video input, frame buffer
│   ├── detection/             # YOLO wrapper
│   ├── tracking/              # ByteTrack wrapper
│   ├── face/                  # enrolment and recognition
│   ├── activity/              # CNN-LSTM model, training, inference
│   ├── rules/                 # zones, lines, dwell, occupancy, abandoned
│   ├── threat/                # correlation, scoring, incidents
│   ├── evidence/              # image and clip writer
│   ├── notify/                # alert channels
│   ├── pipeline/              # orchestration, workers
│   └── api/                   # FastAPI app, routes, websocket, auth
├── frontend/                  # React dashboard
├── evidence/                  # generated incident evidence
├── tests/
├── scripts/                   # enrol, train, evaluate, demo
└── README.md
```

## 5. Deliverables

1. Working intelligent CCTV surveillance system (pipeline, API, dashboard).
2. Trained activity model and evaluation report.
3. Documentation set in this folder.
4. Test results including runtime table.
5. Demo video of an end-to-end incident.
6. User guide for officers and administrators.

## 6. Risks

| Risk | Impact | Response |
|------|--------|----------|
| FPS below target on RTX 4050 | Poor real-time experience | Switch YOLO small to nano, frame skipping, 640 px input, async workers |
| ID switches in crowds | Wrong correlation | Tune tracker, use paths and time windows |
| Weak activity accuracy | False alerts | Require consecutive windows plus rule support, add self-recorded data |
| Face recognition unreliable at distance | Missed identities | Minimum face size, re-check, treat as Unknown safely |
| Abandoned-object false positives | Noise | Owner-distance and time thresholds |
| Dataset size or quality | Poor generalisation | Augmentation, self-recorded clips |
| Scope too large | Delays | Follow phase order; cut optional items first |

## 7. Optional Enhancements (after core is done)

- Telegram or WhatsApp alerts with evidence.
- Cross-camera tracking of a person or object.
- Edge deployment (Jetson / Raspberry Pi class device).
- Mobile app.
- Automatic reporting of violations.
- Advanced models for complex unusual behaviour.
