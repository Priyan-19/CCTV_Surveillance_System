# Phase 3: Face Recognition Integration Engineering Notes

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection  
**Hardware Target:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Windows 11, Python 3.11 (`.venv` via UV)  
**Inference Engine:** ONNX Runtime GPU (`1.22.0`, CUDA 12 + cuDNN 9) / PyTorch (`2.6.0+cu124`)

---

## 1. Overview & Accomplishments

Phase 3 integrated real-time open-set facial recognition into the intelligent surveillance pipeline, enabling the system to recognise whether tracked individuals are **KNOWN** (with identity name and person ID) or **UNKNOWN** (strangers), while dynamically enriching downstream security events with identity status and authorization tags.

### Key Architectural Deliverables:
1. **Privacy-First Enrolment CLI & Verification:**
   - Strict explicit consent gate (`--consent-confirmed`) required before any biometric extraction or storage occurs.
   - Enforces a minimum of 3 high-quality face images per subject.
   - Complete CRUD CLI suite: `enroll_person.py`, `list_persons.py`, and `delete_person.py`.
2. **Relational Biometric Store (`FaceStore`):**
   - SQLite relational database (`data/cctv.db`) structured with `PRAGMA foreign_keys = ON;`.
   - `persons` table linked to `face_embeddings` table with strict `ON DELETE CASCADE`.
   - Embeddings stored as 512-dimensional IEEE-754 `float32` L2-normalized vector BLOBs (2048 bytes).
   - Physical deletion cascade automatically purges raw enrolment image files from disk upon person deletion.
3. **Three-Stage Quality Filter (`FaceQualityFilter`):**
   - **Dimension Gate:** Face crop width and height $\ge 36\text{ px}$.
   - **Sharpness Gate:** Laplacian blur variance $\text{Var}(\nabla^2 I) \ge 30.0$, rejecting motion-blurred crops.
   - **Confidence Gate:** RetinaFace detection confidence score $\ge 0.55$.
4. **Fast Cosine Similarity Matcher (`FaceMatcher`):**
   - Vectorized matrix multiplication against the in-memory gallery of L2-normalized embeddings: $S = E_{\text{query}} \cdot G^{\top}$.
   - Dual-threshold rejection: Similarity threshold $\tau = 0.35$ and ambiguity margin $\delta = 0.04$ (rejects queries where $s_{\text{top1}} - s_{\text{top2}} < \delta$).
5. **Deterministic Identity State Machine (`IdentityManager`):**
   - States: `PENDING` $\rightarrow$ `KNOWN` or `UNKNOWN`.
   - **$K$-Consensus:** Requires $K=2$ consecutive matching frames for known person identification.
   - **Stranger Hysteresis:** Requires $K=2$ non-matching frames to confirm a stranger.
   - **Periodic Re-checking:** Re-evaluates unconfirmed tracks every $0.3\text{ s}$.
   - **`no_usable_face` Timeout:** Automatically flips persistent tracks with no valid face crops to `UNKNOWN` after $2.0\text{ s}$ (`details: "no_usable_face"`).
   - **Encounter Finalization (`finalize_track`):** Resolves short-duration transit encounters upon track exit.
6. **Decoupled Asynchronous Background Worker (`FaceWorker`):**
   - Standalone background daemon thread communicating via thread-safe bounded FIFO queue (`maxsize: 16`).
   - Main video processing loop incurs $< 0.15\text{ ms}$ overhead to enqueue candidate crops.
   - Non-blocking `drop_newest` queue overflow policy protects real-time video playback throughput.
   - Telemetry tracks queue depth, jobs processed, jobs dropped, and worker drop rate.
7. **Rigorous Open-Set ChokePoint Benchmark:**
   - Evaluated on ChokePoint Portal 1 dataset (25 total subjects, 50/50 open-set split: 12 Known, 13 Strangers, seed 42).
   - Zero-leakage quarantine protocol (48 enrolment frames excluded from test scoring).
   - Separate tuning set (`P1E_S1_C2`) and held-out validation set (`P1E_S2_C1`):
     - **FAR: 0.00%** (0 false accepts out of 13 stranger encounters) -> **PASS** (Target $< 2.0\%$).
     - **FRR: 0.00%** (0 false rejects out of 12 known encounters) -> **PASS** (Target $< 5.0\%$).
     - **Accuracy: 100.00%** (25/25 encounters correctly classified) -> **PASS** (Target $> 95.0\%$).
     - **Pending Rate: 0.00%** (0 tracks unresolved at exit) -> **PASS** (Target $0.0\%$).
     - **Mean Time-to-Decision: 0.34 s** -> **PASS** (Target $< 2.0\text{ s}$).
8. **Low-Resolution CCTV Distance Limitation Analysis:**
   - Empirical validation on CAVIAR (384x288), ABODA (640x480), and ChokePoint (800x600) documenting pixel-per-face physical thresholds for reliable surveillance face recognition.

---

## 2. How to Run Phase 3 Scripts

All commands run inside the virtual environment (`.venv`) in PowerShell on Windows 11.

### 2.1 Enrolment CLI Suite

#### A. Enrol a New Person (with Explicit Consent)
```powershell
python scripts/enroll_person.py --person-id EMP_001 --name "John Doe" --images data/known_persons/john_1.jpg data/known_persons/john_2.jpg data/known_persons/john_3.jpg --consent-confirmed
```
*(Omitting `--consent-confirmed` halts execution immediately with an explicit privacy warning).*

#### B. List All Enrolled Persons & Active Embeddings
```powershell
python scripts/list_persons.py
```

#### C. Delete an Enrolled Person (Cascade Deletion)
```powershell
python scripts/delete_person.py --person-id EMP_001
```

---

### 2.2 ChokePoint Open-Set Dataset Tools

#### A. Download ChokePoint Benchmark Archives
Downloads Portal 1 ground truth XMLs and image sequences directly via the Zenodo REST API with MD5 checksum verification:
```powershell
python scripts/fetch_chokepoint.py --archives groundtruth.tar.xz P1E_S1.tar.xz P1E_S2.tar.xz
```

#### B. Generate 50/50 Split & Enrol Known Subjects
Partitions 25 subjects into 12 Known and 13 Strangers (seed 42), crops enrolment faces from Session 1, writes `configs/chokepoint_split.yaml`, and populates `data/cctv.db`:
```powershell
python scripts/enroll_from_chokepoint.py --seed 42 --consent-confirmed
```

#### C. Run Open-Set Benchmark Evaluation
Executes grid parameter tuning on `P1E_S1_C2` and held-out validation on `P1E_S2_C1`:
```powershell
python scripts/evaluate_faces.py --mode both --stride 2
```
Outputs:
- Markdown report: `data/output/evaluation/face_evaluation_report.md`
- Encounter CSV log: `data/output/evaluation/face_evaluation_report.csv`

---

### 2.3 Phase 3 Single Stream Pipeline Runner (`run_phase3.py`)

Executes the integrated YOLO26s + ByteTrack + RuleEngine + Asynchronous FaceWorker pipeline on video feeds:
```powershell
# Pedestrian corridor stream (headless)
python scripts/run_phase3.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01

# Pedestrian corridor with visual display window & HUD
python scripts/run_phase3.py --source data/samples/sample_walking_01.mp4 --camera-id cam_walk_01 --show

# Unattended luggage HD stream
python scripts/run_phase3.py --source data/samples/sample_abandoned_bag_hd.mp4 --camera-id cam_abandoned_hd --imgsz 960 --max-frames 1800 --show
```

---

### 2.4 Comprehensive Face Benchmark Suite (`benchmark_faces.py`)

Measures raw InsightFace neural latency on the RTX 4050 GPU, followed by full-pipeline throughput across all 4 benchmark clips:
```powershell
python scripts/benchmark_faces.py
```

---

### 2.5 Unit Test Suite

Run CPU-safe synthetic tests validating quality filtering, matcher ambiguity rejection, state machine consensus/hysteresis/timeouts, and SQLite cascade deletions:
```powershell
# Run Phase 3 unit tests
pytest tests/test_phase3.py -v

# Run combined Phase 2 + Phase 3 regression tests
pytest tests/test_phase2.py tests/test_phase3.py -v
```
*(19/19 tests pass in ~0.65 seconds).*

---

## 3. Database Schema & Storage Verification

Face identities and mathematical embeddings are persisted in SQLite (`data/cctv.db`) conforming to `docs/07_DATABASE_SCHEMA.md`:

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    access_level TEXT DEFAULT 'Standard',
    consent_confirmed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    quality_score REAL,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);
```

### Cascade Deletion Verification
Cascade deletion was verified both programmatically and via unit tests (`test_sqlite_cascade_delete` in `tests/test_phase3.py`):
1. `PRAGMA foreign_keys = ON;` is strictly executed on every SQLite connection opening.
2. Executing `DELETE FROM persons WHERE person_id = ?;` automatically purges all associated child rows in `face_embeddings`.
3. The application-layer `delete_person()` method queries and deletes raw image crops from disk (`data/known_persons/<person_id>/`) before committing the transaction.

---

## 4. Face Recognition Architecture & Modules

```
                           +------------------------+
                           | Incoming Video Stream  |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           |  YOLO26s + ByteTrack   |
                           +-----------+------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
          +---------------------+             +---------------------+
          | Spatial Rule Engine |             | Track Bounding Box  |
          +----------+----------+             +----------+----------+
                     |                                   |
                     |                        +----------v----------+
                     |                        | Face Crop Extractor |
                     |                        +----------+----------+
                     |                                   |
                     |                        +----------v----------+
                     |                        | FaceQualityFilter   |
                     |                        |  - Size >= 36px     |
                     |                        |  - Var(Lap) >= 30   |
                     |                        |  - Conf >= 0.55     |
                     |                        +----------+----------+
                     |                                   |
                     |                                   | (Passes Gate)
                     |                                   v
                     |                        +---------------------+
                     |                        |  Bounded FIFO Queue |
                     |                        |   (maxsize = 16)    |
                     |                        +----------+----------+
                     |                                   |
                     |                        +----------v----------+
                     |                        | FaceWorker (Thread) |
                     |                        | - InsightFace Model |
                     |                        | - Cosine Matcher    |
                     |                        | - Identity Manager  |
                     |                        +----------+----------+
                     |                                   |
                     | (Enrich Identity)                 v
                     +<-----------------------+ Updated Track ID    |
                     |                        | (KNOWN / UNKNOWN)   |
                     v                        +---------------------+
          +---------------------+
          | JSONL Event Logger  |
          | - person_id         |
          | - identity_status   |
          | - authorised (bool) |
          +---------------------+
```

### 4.1 InsightFace Model Configuration
- **Model Bundle:** `buffalo_s` (lightweight mobile architecture optimized for real-time edge/laptop surveillance).
- **Detector:** RetinaFace (ResNet50 / MobileNet backbone, detection size $320 \times 320$).
- **Recognition Feature Extractor:** ArcFace MobileNet backbone generating 512-dimensional L2-normalized embeddings.
- **Execution Provider:** `CUDAExecutionProvider` on NVIDIA RTX 4050 GPU (FP16/FP32).

### 4.2 Quality Filter Criteria (`FaceQualityFilter`)
Surveillance video frequently yields degraded, blurred, or occluded face crops. Passing poor crops to ArcFace causes severe embedding distortion and false acceptances. Three strict gates prevent low-quality inputs:
1. **Minimum Dimensions:** Both crop width and height must satisfy $\ge 36\text{ px}$.
2. **Laplacian Blur Variance:** Evaluates sharpness using the variance of the 2D Laplacian operator:
   $$\sigma^2 = \text{Var}\left( \nabla^2 I \right) \ge 30.0$$
   Crops with $\sigma^2 < 30.0$ are rejected as motion-blurred.
3. **Detection Score:** RetinaFace face confidence score must be $\ge 0.55$.

### 4.3 Similarity Matcher & Ambiguity Margin (`FaceMatcher`)
Embeddings are normalized to unit Euclidean length: $\|\mathbf{e}\|_2 = 1.0$. The similarity between a query embedding $\mathbf{q}$ and gallery embedding $\mathbf{g}_i$ simplifies to the dot product:
$$\text{Sim}(\mathbf{q}, \mathbf{g}_i) = \mathbf{q} \cdot \mathbf{g}_i$$
For an encounter to be classified as KNOWN:
1. The top cosine similarity $s_1 = \max_i \text{Sim}(\mathbf{q}, \mathbf{g}_i)$ must satisfy $s_1 \ge \tau = 0.35$.
2. The top similarity must exceed the second-best distinct candidate similarity $s_2$ by the ambiguity margin $\delta$:
   $$s_1 - s_2 \ge \delta = 0.04$$
   If $s_1 - s_2 < \delta$, the match is rejected as ambiguous, preventing false associations between similar-looking gallery subjects.

### 4.4 Identity State Machine (`IdentityManager`)
- **Initial State:** All new ByteTrack tracks begin as `IdentityStatus.PENDING`.
- **$K$-Consensus Confirmation:** Requires $K=2$ consecutive frames matching the same enrolled subject to promote a track from `PENDING` to `KNOWN`.
- **Stranger Hysteresis:** Requires $K=2$ consecutive non-matching frames to transition from `PENDING` to `UNKNOWN`.
- **Timeout Fallback:** If a track has been active for $> 2.0\text{ s}$ without any usable face crops passing the quality filter, it is assigned `IdentityStatus.UNKNOWN` with `details: "no_usable_face"`.
- **Periodic Re-checking:** Active unconfirmed tracks are resampled every $0.3\text{ s}$ (`recheck_interval_seconds: 0.3`) to conserve GPU compute while maintaining rapid responsiveness.
- **Track Finalization:** When a track exits the frame (`finalize_track()`), if it has accumulated stranger votes but has not reached the $2.0\text{ s}$ timeout, it is cleanly resolved to `UNKNOWN`.

### 4.5 Asynchronous Decoupled Worker (`FaceWorker`)
To guarantee that face detection and ArcFace feature extraction do not stall the primary YOLO26s detection and ByteTrack tracking loops:
- `FaceWorker` operates as a dedicated Python `threading.Thread`.
- Communication occurs via `queue.Queue(maxsize=16)`.
- If the queue is saturated, the newest crop is dropped (`drop_newest` policy), and an internal drop counter increments.
- Under all tested surveillance scenarios on the RTX 4050 GPU, the face drop rate was **0.0%**.

---

## 5. Measured Performance & Latency Benchmark

Benchmarks were conducted on the target hardware (**NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM, Windows 11**) comparing Phase 3 against the Phase 2 baseline:

### 5.1 Raw Neural Inference Latency (InsightFace `buffalo_s`)
- **Mean Inference Latency:** **9.06 ms** per crop
- **Median (p50):** 8.56 ms
- **95th Percentile (p95):** 17.32 ms
- **Maximum GPU Processing Capacity:** **110.4 face jobs/sec**

### 5.2 Full Pipeline Throughput & Latency Comparison

| Test Clip | Camera ID | Resolution / Config | Phase 2 Baseline FPS | Phase 3 Pipeline FPS | Frame Latency (Phase 3) | Face Jobs Queued | Face Jobs Processed | Drop Rate | TRD Compliance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sample_walking_01.mp4` | `cam_walk_01` | 384x288 (640px) | 27.92 FPS | **24.81 FPS** | 40.23 ms | 141 | 141 | **0.0%** | **PASS** |
| `sample_fight_01.mp4` | `cam_fight_01` | 384x288 (640px) | 27.42 FPS | **27.32 FPS** | 36.46 ms | 72 | 72 | **0.0%** | **PASS** |
| `sample_normal_01.mp4` | `cam_walk_01` | 384x288 (640px) | 27.67 FPS | **27.72 FPS** | 35.88 ms | 40 | 40 | **0.0%** | **PASS** |
| `sample_abandoned_bag_hd.mp4` | `cam_abandoned_hd` | 640x480 (960px) | 23.49 FPS | **18.83 FPS** | 53.03 ms* | 178 | 178 | **0.0%** | **THROUGHPUT PASS**<br>*(Latency Below Target)* |

> [!NOTE]
> **Analysis of HD Clip Latency (53.03 ms):**
> On `sample_abandoned_bag_hd.mp4`, throughput remains above target (**18.83 FPS** $\ge 18.0\text{ FPS}$). However, total frame latency reached 53.03 ms (exceeding the 45.0 ms ceiling) because the YOLO detector was configured with `imgsz=960` to capture small static luggage, which increases detector inference time to ~35 ms alone. Under standard `imgsz=640`, total frame latency drops to **37.2 ms** (fully compliant with TRD).
> Importantly, the face recognition subsystem (`FaceWorker`) operates asynchronously: its enqueue overhead on the main thread is $< 0.15\text{ ms}$, and zero face jobs were dropped across all 178 requests.

---

## 6. Open-Set Benchmark Evaluation (ChokePoint)

The face recognition subsystem was evaluated under rigorous open-set conditions using the standard **ChokePoint** biometric benchmark dataset (Portal 1).

### 6.1 Split & Zero-Leakage Protocol
- **Total Unique Subjects:** 25
- **Known / Enrolled Subjects (12):** `0007`, `0009`, `0012`, `0013`, `0014`, `0015`, `0018`, `0019`, `0020`, `0021`, `0022`, `0024`
- **Stranger Subjects (13):** `0001`, `0003`, `0004`, `0005`, `0006`, `0010`, `0011`, `0016`, `0017`, `0023`, `0025`, `0026`, `0027`
- **Enrolment Source:** `P1E_S1_C1` (Camera 1, Session 1). 4 quality face crops per known subject enrolled into SQLite (48 total embeddings).
- **Zero-Leakage Quarantine:** Exactly 48 enrolment frame numbers were quarantined and programmatically excluded from test evaluation scoring.
- **Tuning Set:** `P1E_S1_C2` (Camera 2, Session 1). Used to calibrate the similarity threshold and ambiguity margin.
- **Held-Out Validation Set:** `P1E_S2_C1` (Camera 1, Session 2). Completely unseen multi-day recording with different lighting, walking paces, and clothing.

### 6.2 Evaluation Scorecard

| Metric | Target Specification | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **False Accept Rate (FAR)** | $< 2.0\%$ | **0.00%** (0 / 13 false accepts) | **PASS** |
| **False Reject Rate (FRR)** | $< 5.0\%$ | **0.00%** (0 / 12 false rejects) | **PASS** |
| **Overall Classification Accuracy** | $> 95.0\%$ | **100.00%** (25 / 25 correct) | **PASS** |
| **Pending Rate at Exit** | $0.0\%$ | **0.00%** (0 / 25 pending) | **PASS** |
| **Mean Time-to-Decision** | $< 2.0\text{ s}$ | **0.34 s** | **PASS** |

### 6.3 Detailed Validation Encounters (`P1E_S2_C1`)

Every subject encounter was audited against ground truth:

| Subject ID | Ground Truth | Predicted Status | Name | Top Similarity | Ambiguity Margin | Decision Time | Outcome |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `0009` | KNOWN | KNOWN | Subject_0009 | 0.6526 | 0.5710 | 0.33 s | True Accept (TA) |
| `0003` | STRANGER | UNKNOWN | (Unknown) | 0.1786 | 0.0157 | 0.33 s | True Reject (TR) |
| `0017` | STRANGER | UNKNOWN | (Unknown) | 0.1523 | 0.0184 | 0.40 s | True Reject (TR) |
| `0005` | STRANGER | UNKNOWN | (Unknown) | 0.1806 | 0.0279 | 0.33 s | True Reject (TR) |
| `0015` | KNOWN | KNOWN | Subject_0015 | 0.6122 | 0.3588 | 0.33 s | True Accept (TA) |
| `0024` | KNOWN | KNOWN | Subject_0024 | 0.6879 | 0.5751 | 0.33 s | True Accept (TA) |
| `0022` | KNOWN | KNOWN | Subject_0022 | 0.6599 | 0.5080 | 0.40 s | True Accept (TA) |
| `0019` | KNOWN | KNOWN | Subject_0019 | 0.6556 | 0.5677 | 0.33 s | True Accept (TA) |
| `0021` | KNOWN | KNOWN | Subject_0021 | 0.7429 | 0.6105 | 0.33 s | True Accept (TA) |
| `0025` | STRANGER | UNKNOWN | (Unknown) | 0.1778 | 0.0544 | 0.33 s | True Reject (TR) |
| `0023` | STRANGER | UNKNOWN | (Unknown) | 0.2237 | 0.0258 | 0.33 s | True Reject (TR) |
| `0026` | STRANGER | UNKNOWN | (Unknown) | 0.4504 | 0.2146 | 0.33 s | True Reject (TR) |
| `0027` | STRANGER | UNKNOWN | (Unknown) | 0.1789 | 0.0476 | 0.33 s | True Reject (TR) |
| `0016` | STRANGER | UNKNOWN | (Unknown) | 0.3327 | 0.1571 | 0.33 s | True Reject (TR) |
| `0013` | KNOWN | KNOWN | Subject_0013 | 0.6751 | 0.6146 | 0.33 s | True Accept (TA) |
| `0020` | KNOWN | KNOWN | Subject_0020 | 0.6890 | 0.5264 | 0.40 s | True Accept (TA) |
| `0007` | KNOWN | KNOWN | Subject_0007 | 0.6845 | 0.4063 | 0.33 s | True Accept (TA) |
| `0010` | STRANGER | UNKNOWN | (Unknown) | 0.2332 | 0.0208 | 0.37 s | True Reject (TR) |
| `0006` | STRANGER | UNKNOWN | (Unknown) | 0.2296 | 0.0468 | 0.40 s | True Reject (TR) |
| `0011` | STRANGER | UNKNOWN | (Unknown) | 0.1742 | 0.0614 | 0.33 s | True Reject (TR) |
| `0018` | KNOWN | KNOWN | Subject_0018 | 0.7282 | 0.6053 | 0.33 s | True Accept (TA) |
| `0001` | STRANGER | UNKNOWN | (Unknown) | 0.2079 | 0.0610 | 0.33 s | True Reject (TR) |
| `0004` | STRANGER | UNKNOWN | (Unknown) | 0.1699 | 0.0856 | 0.33 s | True Reject (TR) |
| `0012` | KNOWN | KNOWN | Subject_0012 | 0.6668 | 0.4824 | 0.33 s | True Accept (TA) |
| `0014` | KNOWN | KNOWN | Subject_0014 | 0.7561 | 0.5814 | 0.33 s | True Accept (TA) |

---

## 7. CCTV Resolution vs Distance Limitations (Empirical Analysis)

A critical engineering question in automated surveillance is: **At what distance and camera resolution can face recognition reliably operate?**

Empirical observations across the benchmark clips reveal sharp operational boundaries:

### 7.1 Cross-Dataset Resolution Comparison

| Dataset / Clip | Native Resolution | Camera Geometry | Pedestrian Distance | Typical Face Dimensions | Recognition Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CAVIAR** (`sample_walking_01.mp4`) | $384 \times 288$ | High overhead ceiling ($> 3.5\text{m}$) | $4.0 - 12.0\text{ m}$ | **$10 \times 10\text{ px}$ to $18 \times 18\text{ px}$** | **Safely Filtered:** Faces fail `min_face_size: 36px` gate. Identity remains `PENDING` until $2.0\text{s}$ timeout flips to `UNKNOWN` (`no_usable_face`). |
| **ABODA** (`sample_abandoned_bag_hd.mp4`) | $640 \times 480$ | Wall-mounted lobby | $2.5 - 6.0\text{ m}$ | **$26 \times 26\text{ px}$ to $38 \times 38\text{ px}$** | **Marginal / Partial:** Frontal crops close to camera ($< 3\text{m}$) pass the filter and are evaluated as `UNKNOWN` (strangers). Distant frames are rejected. |
| **ChokePoint** (`P1E_S1`, `P1E_S2`) | $800 \times 600$ | Doorway portal archway | $1.5 - 2.5\text{ m}$ | **$45 \times 45\text{ px}$ to $85 \times 85\text{ px}$** | **Optimal:** 100% of encounters yield sharp, well-illuminated crops. Rapid decision in $0.34\text{ s}$ with 0% error. |

### 7.2 The Pixels-Per-Face Law in CCTV
ArcFace and RetinaFace are trained primarily on normalized $112 \times 112\text{ px}$ aligned face crops. When applied to real-world CCTV feeds:
1. **$< 20\text{ px}$ (Extreme Low-Res):** Severe Nyquist-Shannon information loss. Eye corners, nose bridges, and facial contours blur into flat pixel clusters. Attempting ArcFace inference on $< 20\text{ px}$ crops produces embedding noise, causing unpredictable false accepts. **Our `FaceQualityFilter` prevents this failure mode by dropping crops $< 36\text{ px}$.**
2. **$20 - 35\text{ px}$ (Marginal Range):** Face detection is possible, but recognition confidence drops significantly. In-plane head pose changes $> 20^\circ$ degrade similarity scores below $\tau = 0.35$.
3. **$\ge 36\text{ px}$ (Viable Surveillance Range):** Sufficient texture for inter-pupillary distance measurement and reliable deep feature extraction.
4. **$\ge 60\text{ px}$ (High-Confidence Biometrics):** Ideal operational envelope for 1:N open-set recognition with zero false accepts.

### 7.3 Operational CCTV Deployment Recommendations
- **Access Control & Portals:** Place cameras at eye-level ($1.6 - 1.8\text{ m}$ height) covering narrow chokepoints (corridors, turnstiles, doors) within $1.5 - 3.0\text{ m}$ of transit. This guarantees $\ge 50\text{ px}$ face crops even at $720\text{p}$.
- **Overhead Surveillance & Wide Corridors:** Overhead ceiling cameras ($> 3.5\text{ m}$ height) cannot deliver frontal biometrics for people walking underneath due to steep pitch angles ($> 45^\circ$) and perspective foreshortening. These cameras should focus on YOLO tracking and Phase 2 spatial rules, deferring facial identification to portal cameras.

---

## 8. Interfaces with Downstream Phases

### 8.1 Phase 4 Interface (Threat Engine)
- **Track Enrichment:** Every `Track` object now carries:
  - `track.identity.status` (`IdentityStatus.PENDING`, `KNOWN`, or `UNKNOWN`).
  - `track.identity.person_id` (e.g. `"EMP_001"` or `None`).
  - `track.identity.name` (e.g. `"John Doe"` or `None`).
  - `track.identity.confidence` (best cosine similarity score).
- **Event Enrichment:** Every security `Event` logged by `RuleEngine` includes:
  - `event.person_id`: Populated directly from `track.identity.person_id`.
  - `event.identity_status`: String value (`"PENDING"`, `"KNOWN"`, or `"UNKNOWN"`).
  - `event.authorised`: Boolean flag (`True` if enrolled authorized employee, `False` if unknown stranger or unassigned).
- **Scope Isolation:** As required by the architectural principles, Phase 3 **does not suppress or downgrade Phase 2 events**. Authorized employees who cross restricted lines still trigger `RESTRICTED_ZONE_ENTRY` events with `authorised: True`. Incident scoring and alert dampening belong strictly to Phase 4 (Threat Engine).

### 8.2 Phase 6 Interface (Dashboard & HUD)
- The Phase 3 annotator (`Phase3Annotator`) renders:
  - **Emerald Green Bounding Boxes:** Confirmed `KNOWN` persons (displays `[ID: 7] Subject_0007 (0.68)`).
  - **Amber / Orange Bounding Boxes:** Confirmed `UNKNOWN` strangers (displays `[ID: 3] UNKNOWN`).
  - **Muted Cyan Bounding Boxes:** `PENDING` tracks currently being evaluated or awaiting face crops.
  - **HUD Worker Telemetry Banner:** Top-right HUD display showing `Face Q: 0/16 | Drop: 0.0% | Active: 12 persons`.

---

## 9. Phase 3 Done Checklist

- [x] Privacy-first enrolment CLI (`scripts/enroll_person.py`, `scripts/list_persons.py`, `scripts/delete_person.py`) with mandatory `--consent-confirmed` verification.
- [x] SQLite relational storage (`persons`, `face_embeddings`) with verified `ON DELETE CASCADE` and physical file purge on delete.
- [x] Three-stage quality filter (`FaceQualityFilter`) evaluating dimensions ($\ge 36\text{px}$), blur variance ($\ge 30.0$), and detector score ($\ge 0.55$).
- [x] Cosine similarity matcher (`FaceMatcher`) with similarity threshold ($\tau = 0.35$) and ambiguity margin ($\delta = 0.04$).
- [x] Deterministic identity state machine (`IdentityManager`) managing $K$-consensus ($K=2$), stranger hysteresis ($K=2$), periodic rechecks ($0.3\text{s}$), and timeout fallbacks ($2.0\text{s}$).
- [x] Asynchronous background worker (`FaceWorker`) with bounded queue (`maxsize: 16`), non-blocking drop policy, and zero drop rate in real-world benchmarks.
- [x] Open-set benchmark on ChokePoint dataset:
  - [x] False Accept Rate (FAR) $= \mathbf{0.00\%}$ ($< 2.0\%$ target).
  - [x] False Reject Rate (FRR) $= \mathbf{0.00\%}$ ($< 5.0\%$ target).
  - [x] Overall accuracy $= \mathbf{100.00\%}$ ($> 95.0\%$ target).
  - [x] Pending rate $= \mathbf{0.00\%}$ ($0.0\%$ target).
  - [x] Decision latency $= \mathbf{0.34\text{ s}}$ ($< 2.0\text{ s}$ target).
- [x] CCTV distance and resolution limitations empirically analyzed and documented across CAVIAR, ABODA, and ChokePoint.
- [x] Comprehensive unit test suite (`tests/test_phase3.py`) passing with **100% pass rate** (9/9 tests).
- [x] End-to-end pipeline runner (`scripts/run_phase3.py`) and benchmarking suite (`scripts/benchmark_faces.py`) operational.
- [x] Complete Phase 3 engineering notes recorded in `docs/PHASE3_NOTES.md`.
