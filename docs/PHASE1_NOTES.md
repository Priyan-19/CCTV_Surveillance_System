# Phase 1: Detection & Tracking Engineering Notes

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection  
**Hardware Target:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM), Windows 11, Python 3.11 (`.venv` via UV)

---

## 1. Overview & Accomplishments

Phase 1 constructed the core visual processing and tracking backbone:
- **Model:** Verified and deployed `yolo26s.pt` with FP16 half-precision on CUDA.
- **Tracker:** ByteTrack integration via `configs/bytetrack.yaml` maintaining persistent Track IDs across frames.
- **Capture Abstraction:** `VideoSource` supporting video files (no-drop mode), local webcams (drop-oldest bounded queuing for real-time latency), and RTSP streams (marked **UNTESTED** in local lab due to absence of an external RTSP camera).
- **Visualization:** Anti-aliased bounding boxes, Track IDs, motion history trails, and a real-time status HUD.
- **Profiling:** Detailed per-stage latency telemetry (`read_ms`, `detect_ms`, `track_ms`, `draw_ms`, `write_ms`) and VRAM tracking (`allocated`, `reserved`, `total`).

---

## 2. How to Run

### 2.1 Single Video Stream Run
```powershell
python scripts/run_phase1.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --model yolo26s.pt --show
```
**Flags:**
- `--source`: Path to MP4/AVI file, webcam device index (`0`), or RTSP stream URL (`rtsp://...`).
- `--camera-id`: Camera tag matching `configs/cameras.yaml`.
- `--model`: Model weights name (default: `yolo26s.pt`).
- `--imgsz`: Input dimension (default: `640`; upsample to `960` for distant objects).
- `--conf`: Confidence threshold (default: `0.25`).
- `--show`: Display live OpenCV window.
- `--no-save`: Disable writing video to disk.
- `--max-frames`: Limit frames to process.

### 2.2 Live Webcam Run
```powershell
python scripts/run_phase1.py --source 0 --camera-id cam_webcam --model yolo26s.pt --show --max-frames 150
```

### 2.3 Automated Benchmark Matrix
```powershell
python scripts/benchmark_phase1.py --clips data/samples/sample_walking_01.mp4 data/samples/sample_abandoned_bag_01.mp4 --models yolo26s.pt yolo11s.pt --img-sizes 640 960 --confs 0.25 0.40 --max-frames 200
```
Outputs:
- Markdown summary: `data/output/benchmarks/phase1_benchmark_results.md`
- CSV export: `data/output/benchmarks/phase1_benchmark_results.csv`

### 2.4 Track ID Stability & Fragmentation Audit
```powershell
python scripts/track_report.py --source data/samples/sample_walking_01.mp4 --model yolo26s.pt
```
Outputs:
- ID Timeline Chart: `data/output/track_reports/<clip>_id_timeline.png`
- Frame Contact Sheet: `data/output/track_reports/<clip>_contact_sheet.png`

### 2.5 Unit Tests (CPU-safe, Zero GPU needed)
```powershell
pytest tests/test_phase1.py -v
```

---

## 3. Configuration Reference

### 3.1 `configs/pipeline.yaml`
```yaml
model:
  name: "yolo26s.pt"
  device: "cuda:0"
  half_precision: true
  imgsz: 640
  conf_threshold: 0.25
  iou_threshold: 0.70

tracked_classes: [0, 1, 2, 3, 5, 7, 24, 26, 28] # person, vehicles, bags

tracker:
  config_path: "configs/bytetrack.yaml"
  persist: true
  trail_length: 30

pipeline:
  frame_skip: 0
  output_dir: "data/output"
  save_annotated_video: true
  display_window: false
```

### 3.2 `configs/bytetrack.yaml` (Tuned Settings)
```yaml
tracker_type: bytetrack
track_high_thresh: 0.5   # High-confidence detection threshold (1st association)
track_low_thresh: 0.1    # Low-confidence detection threshold (2nd association for occlusion recovery)
new_track_thresh: 0.6    # Minimum score to instantiate a new track
track_buffer: 60         # Increased to 60 frames (~2.4 sec) to prevent fragmentation during turns
match_thresh: 0.8        # Hungarian algorithm matching threshold
fuse_score: true         # Fuse detection confidence into similarity
```

---

## 4. Measured Results vs. TRD Targets

| Metric | Target (`docs/02_TRD.md`) | Measured (yolo26s.pt, 640px) | Status |
| :--- | :--- | :--- | :--- |
| **Inference Frame Rate** | 18–22 FPS per stream | **26.05 – 32.07 FPS** | **PASS (Exceeds target)** |
| **Overall Throughput** | > 18 FPS real-time | **19.97 – 21.26 FPS** (includes disk write & display) | **PASS** |
| **Frame Processing Latency**| 20–45 ms | **37.40 – 40.15 ms** (median: **32.36 – 33.04 ms**) | **PASS** |
| **Peak VRAM Allocated** | < 5.0 GB (< 6.0 GB total) | **131.36 – 273.65 MB** | **PASS** |
| **Peak VRAM Reserved** | < 5.0 GB | **154.00 – 362.00 MB** (hardware total: 6.0 GB) | **PASS (5.6 GB free)** |
| **Unit Tests** | 100% pass | **10 / 10 passed** | **PASS** |

---

## 5. Known Limitations & Engineering Observations

1. **Low-Resolution Small Object Limits (CAVIAR 384×288):**
   - The CAVIAR dataset resolution is 384×288. At this resolution, distant bags and backpacks occupy only ~10×10 pixels, which COCO pretrained weights miss at `imgsz=640`.
   - **Resolution Plan:** Upsampling to `imgsz=960` is fully viable at ~28 FPS. For Phase 2 abandoned bag validation, 720p/1080p clips (or self-recorded phone videos) provide the necessary pixel density.
2. **RTSP Stream Path (Marked UNTESTED):**
   - The multi-threaded reconnection and backoff logic is implemented in `src/capture/video_source.py`, but has not been tested against a live physical RTSP camera endpoint due to local hardware constraints.
3. **Occlusion Re-identification:**
   - ByteTrack relies on spatial-temporal motion (Kalman filter) rather than visual embeddings. Increasing `track_buffer` to 60 frames successfully bridges ~2.4 second occlusions, but prolonged exits (> 3 seconds) will assign a new Track ID. Phase 3 (Face Recognition) will provide visual re-identification across ID changes.

---

## 6. The Exact `Track` Interface for Phase 2 (Rule Engine)

Phase 2 rules (zones, tripwires, dwell time, occupancy, and abandoned objects) will consume the standard `Track` object defined in `src/common/types.py`:

```python
@dataclass
class Track:
    camera_id: str                      # Camera stream identifier (e.g. 'cam_01')
    track_id: int                       # Stable integer ID assigned by ByteTrack
    class_id: int                       # Numeric class ID (0: person, 24: backpack, etc.)
    class_name: str                     # 'person', 'backpack', 'handbag', 'suitcase', 'car'...
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float                   # Detection confidence score (0.0 to 1.0)
    path: List[Tuple[float, float]]     # Ordered history of past centroids [(cx, cy), ...]
    first_seen: float                   # Epoch timestamp of first detection
    last_seen: float                    # Epoch timestamp of most recent observation
    
    # Phase 2 Spatial & Temporal Rule Fields:
    zone_ids: List[str]                 # IDs of zones currently containing track centroid
    dwell_seconds: float                # Continuous seconds accumulated inside current zone

    @property
    def centre(self) -> Tuple[float, float]:
        """(cx, cy) centroid used for point-in-polygon and tripwire intersection."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def lifespan_seconds(self) -> float:
        """Total duration this track has persisted in the scene."""
        return max(0.0, self.last_seen - self.first_seen)
```

### Phase 2 Consumption Blueprint:
- **`RESTRICTED_ZONE_ENTRY`:** Evaluates `cv2.pointPolygonTest` on `track.centre` against zone polygons in `configs/zones.yaml`.
- **`LINE_CROSSING`:** Evaluates geometric segment intersection between `track.path[-2]` & `track.path[-1]` and virtual tripwires.
- **`PROLONGED_STAY` / `LOITERING`:** Checks if `track.dwell_seconds` exceeds the zone's threshold.
- **`OCCUPANCY_EXCEEDED`:** Aggregates `len([t for t in tracks if t.class_name == 'person' and zone_id in t.zone_ids])`.
- **`ABANDONED_OBJECT`:** Detects stationary luggage tracks (`path` displacement < threshold) where no `person` track is within proximity.
