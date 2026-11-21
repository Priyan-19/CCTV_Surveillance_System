# Intelligent CCTV Surveillance System

An Intelligent CCTV Surveillance System Using **YOLO**, **ByteTrack**, **CNN-LSTM**, and **Face Recognition** for Automated Threat Detection and Security Event Intelligence.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v26s-00FFFF.svg)](https://docs.ultralytics.com/)
[![Tests](https://img.shields.io/badge/pytest-passing%20(10%2F10)-green.svg)](tests/)

---

## Architecture & Hardware Target

- **OS:** Windows 11
- **Target GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace architecture)
- **CUDA Runtime:** CUDA 12.4 (`torch==2.6.0+cu124`)
- **Environment & Package Manager:** Astral `uv`
- **Python Version:** CPython 3.11.18

```text
CCTV Camera / File / RTSP Input 
      │
      ▼
[VideoSource] (Multi-threaded acquisition, bounded queue, drop-oldest for live)
      │
      ▼
[ObjectDetector] (YOLO26s FP16 on CUDA, class filtering: person, vehicles, luggage)
      │
      ▼
[ByteTracker] (Spatial association, Kalman filter, track_buffer: 60)
      │
      ▼
[FrameAnnotator] (Boxes, Track IDs, centroid trails, and real-time status HUD)
      │
      ▼
[Phase 2 Rules & Phase 4 Threat Engine] ──► Dashboard & Alerting
```

---

## Project Layout

```text
intelligent-cctv-surveillance/
├── configs/                   # Runtime YAML configurations
│   ├── bytetrack.yaml         # ByteTrack thresholds (track_buffer: 60, match_thresh)
│   ├── cameras.yaml           # Camera sources (files, webcam, RTSP)
│   ├── pipeline.yaml          # Model options, FP16, imgsz (640), class filters
│   ├── threat.yaml            # Threat scoring weights, modifiers, correlation window
│   └── zones.yaml             # Spatial polygons, tripwires, occupancy limits
├── data/
│   ├── datasets/              # Activity recognition datasets (RWF-2000, UCF-Crime)
│   ├── known_persons/         # Enrolment images and face embeddings
│   ├── output/                # Annotated videos, benchmark tables, and reports
│   └── samples/               # Quarantined test video clips (CAVIAR, ABODA)
├── docs/                      # Project specifications, TRD, and phase engineering notes
│   ├── 02_TRD.md              # Technical Requirements Document
│   ├── 04_DATA_FLOW.md        # Data flow and data object specifications
│   ├── 12_PROJECT_PLAN.md     # 10-phase engineering roadmap
│   └── PHASE1_NOTES.md        # Phase 1 benchmark results & Phase 2 Track interface
├── models/                    # Model weights storage (.pt, .onnx)
├── scripts/                   # CLI tools and benchmarking scripts
│   ├── benchmark_phase1.py    # Automated matrix benchmark (models, resolutions, conf)
│   ├── check_gpu.py           # CUDA and VRAM diagnostic script
│   ├── check_imports.py       # Dependency import verification test
│   ├── check_samples.py       # Sample clip resolution and scenario auditor
│   ├── run_phase1.py          # Primary detection and tracking pipeline runner
│   ├── smoke_test.py          # Phase 0 end-to-end GPU verification script
│   └── track_report.py        # Track lifespan, ID stability, and timeline generator
├── src/                       # Core application source code
│   ├── capture/               # VideoSource abstraction (file, webcam, RTSP)
│   ├── common/                # Types (Detection, Track, FrameResult) and metrics
│   ├── detection/             # YOLO detector wrapper (FP16, warm-up, class filtering)
│   ├── pipeline/              # Annotator and Phase 1 orchestration pipeline
│   ├── tracking/              # ByteTrack wrapper and synthetic test tracker
│   ├── activity/              # CNN-LSTM activity classification (Phase 7)
│   ├── api/                   # FastAPI backend, WebSockets, MJPEG streaming (Phase 5)
│   ├── evidence/              # Evidence snapshots and clip generation (Phase 4)
│   ├── face/                  # Face recognition & ArcFace enrolment (Phase 3)
│   ├── notify/                # Notification dispatchers (Phase 8)
│   ├── rules/                 # Spatial & temporal rule engine (Phase 2)
│   └── threat/                # Event correlation & incident scoring (Phase 4)
├── tests/                     # Unit test suites (pytest)
├── pyproject.toml             # Project dependency specification
├── .gitignore                 # Git ignore rules for media, weights, and caches
└── README.md
```

---

## Quickstart Guide

### 1. Environment Setup
```powershell
# Create Python 3.11 virtual environment with UV
uv venv --python 3.11
.venv\Scripts\activate

# Install PyTorch with CUDA 12.4 acceleration
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
uv pip install -e .
```

### 2. Run Unit Tests (Zero-GPU, Fast)
```powershell
pytest tests/test_phase1.py -v
```

### 3. Run Detection & Tracking Pipeline
```powershell
# Run on a recorded video file with live display
python scripts/run_phase1.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --model yolo26s.pt --show

# Run on your local laptop webcam
python scripts/run_phase1.py --source 0 --camera-id cam_webcam --model yolo26s.pt --show --max-frames 150
```

### 4. Run Benchmarking & Track Stability Audits
```powershell
# Run automated benchmark matrix (models x resolutions x confidence levels)
python scripts/benchmark_phase1.py --clips data/samples/sample_walking_01.mp4 data/samples/sample_abandoned_bag_01.mp4 --models yolo26s.pt yolo11s.pt --img-sizes 640 960 --confs 0.25 0.40 --max-frames 200

# Generate Track ID timeline plot and contact sheet
python scripts/track_report.py --source data/samples/sample_walking_01.mp4 --model yolo26s.pt
```

---

## Measured Performance vs. TRD Targets

All benchmarks evaluated on the target **RTX 4050 Laptop GPU (6 GB VRAM)**:

| Metric | Target (`docs/02_TRD.md`) | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Inference Frame Rate** | 18–22 FPS per stream | **26.05 – 32.07 FPS** (`yolo26s.pt`) | **PASS (Exceeds target)** |
| **Pipeline Throughput** | > 18 FPS real-time | **19.97 – 21.26 FPS** (includes disk write & HUD) | **PASS** |
| **Processing Latency** | 20–45 ms per frame | **37.40 – 40.15 ms** (median: **32.36 – 33.04 ms**) | **PASS** |
| **VRAM Allocated** | < 5.0 GB (< 6.0 GB total) | **131.36 – 273.65 MB** | **PASS** |
| **VRAM Reserved** | < 5.0 GB | **154.00 – 362.00 MB** (~5.6 GB VRAM free) | **PASS** |
| **Unit Test Coverage** | 100% pass | **10 / 10 passed** | **PASS** |

---

## Engineering Roadmap

| Phase | Milestone / Domain | Status | Key Deliverables |
| :---: | :--- | :---: | :--- |
| **0** | **Setup & Hardware Validation** | Completed | CUDA verification, library imports, smoke test (34.3 FPS) |
| **1** | **Detection & Tracking** | Completed | VideoSource, YOLO26s FP16, ByteTrack (`track_buffer: 60`), HUD |
| **2** | **Rules & Events** | **Next** | Spatial zones, virtual tripwires, dwell time, abandoned objects |
| **3** | **Face Recognition** | Planned | ArcFace enrolment, Track ID identity binding, Known/Unknown logic |
| **4** | **Threat Engine & Incidents** | Planned | Multi-event correlation, scoring (0–100), incident evidence collector |
| **5** | **Backend API** | Planned | FastAPI, REST endpoints, WebSocket push, MJPEG video streaming |
| **6** | **Dashboard** | Planned | React monitoring interface, incident list, zone editor, alert HUD |
| **7** | **Activity Recognition** | Planned | CNN-LSTM action classifier (fighting, running, loitering) |
| **8** | **Notifications & Hardening** | Planned | External alert dispatchers, log retention, pipeline hardening |
| **9** | **Testing & Documentation** | Planned | Full system test plan execution, evaluation report, user guide |

Detailed Phase 1 findings, benchmark matrices, and the Phase 2 `Track` interface are documented in [docs/PHASE1_NOTES.md](docs/PHASE1_NOTES.md).
