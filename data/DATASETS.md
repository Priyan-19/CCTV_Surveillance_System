# Datasets Manifest & Documentation

This document records the provenance, licensing, citations, technical properties, and usage rules for all public datasets integrated or referenced in the Intelligent CCTV Surveillance System.

---

## 1. Summary of Datasets & Phase Usage

| Dataset | Primary Use Case | Project Phase | Status in Phase 0 | Resolution & FPS | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CAVIAR** | Detection, tracking, zones, dwell, abandoned bag, fight | Phases 0–2 | Active (sample clips) | 384×288 @ 25 FPS | Academic / Non-commercial research |
| **ABODA** | High-resolution abandoned object benchmark | Phases 0–2 | Active (sample clips) | ~720×480 to 1920×1080 @ 25–30 FPS | Open academic research |
| **AVSS 2007 / i-LIDS** | Abandoned baggage in transit/public hall | Phase 2 (optional) | Referenced | 720×576 @ 25 FPS | Academic research with mandatory citation |
| **PETS 2006** | Left luggage in train station | Phase 2 (optional) | Referenced | 720×576 @ 25 FPS | Academic research |
| **ChokePoint** | Person re-identification & face enrolment / recognition | Phase 3 | Script ready (deferred) | 800×600 @ 30 FPS | Academic research (Zenodo CC BY-NC) |
| **RWF-2000** | CNN-LSTM violence & fight activity recognition | Phase 7 | Script ready (deferred) | 30 FPS, 5-sec clips (~640×360) | Academic research |
| **HMDB51 (Run class)** | CNN-LSTM running activity recognition | Phase 7 | Evaluated option | Variable (30 FPS) | CC BY 4.0 |
| **VIRAT** | Realistic outdoor surveillance activities | Phase 7 (optional) | Evaluated option | 1080p / 720p @ 30 FPS | Controlled access (Kitware agreement) |
| **UCF-Crime** | Surveillance anomaly & normal baseline | Phase 7 (optional) | Evaluated option | 320×240 @ 30 FPS | Academic research |

---

## 2. CAVIAR (Context Aware Vision using Image-based Active Recognition)

- **Source URL:** [https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- **License / Terms:** EC Project IST-2001-37540. Free for academic research, education, and benchmark evaluation. No commercial redistribution without explicit permission.
- **Required Citation:**
  > Fisher, R. B., Santos-Victor, J., & Crowley, J. (2004). CAVIAR: Context Aware Vision using Image-based Active Recognition. In *Proceedings of the 2nd Joint IEEE International Workshop on Visual Surveillance and Performance Evaluation of Tracking and Surveillance (VS-PETS)*.
- **Project Role:** Primary source of test clips for Phases 0 to 3 (person detection, ByteTrack tracking, virtual zone entry, line crossing tripwires, dwell time, abandoned luggage, and altercation/running).
- **Files Chosen for Phase 0:**
  1. `Walk1.mpg` & `wk1gt.xml` (~7 MB) — 1 person walking in corridor.
  2. `Walk2.mpg` & `wk2gt.xml` (~8 MB) — 2 people walking along lobby.
  3. `Walk3.mpg` & `wk3gt.xml` (~10 MB) — Multiple people walking and crossing paths.
  4. `LeftBag.mpg` & `lb1gt.xml` (~17 MB) — Person enters, leaves backpack near wall, walks away.
  5. `LeftBag_AtChair.mpg` & `lb2gt.xml` (~11 MB) — Person drops bag beside waiting chairs and leaves.
  6. `LeftBox.mpg` & `lbgt.xml` (~15 MB) — Person deposits unattended cardboard box.
  7. `Fight_RunAway1.mpg` & `fra1gt.xml` (~6 MB) — Two persons meet, engage in physical fight, then scatter and run away.
- **Technical Specifications:**
  - Video format: MPEG-2 (`.mpg`), converted to H.264 (`.mp4`) via ffmpeg.
  - Resolution: Half-resolution PAL (384 × 288 px).
  - Frame rate: 25.0 FPS.
  - Ground truth: CVML XML annotations providing bounding boxes, track IDs, roles, and situational activity tags.
- **Known Limitations & Mitigations:**
  - *Low Resolution Warning:* 384×288 is substantially lower than modern 1080p CCTV. YOLO models trained on 640px may have reduced confidence on small distant objects or small bags.
  - *Mitigation:* Upscale or pad to 640 during YOLO inference; use ABODA or AVSS for high-definition abandoned-bag evaluations.

---

## 3. ABODA (Abandoned Objects Dataset)

- **Source URL:** [https://github.com/kevinlin311tw/ABODA](https://github.com/kevinlin311tw/ABODA)
- **License / Terms:** Public academic research dataset. Non-commercial use.
- **Required Citation:**
  > Lin, K., Chen, S. C., Chen, C. S., Lin, D. T., & Hung, Y. P. (2015). Abandoned object detection via temporal consistency modeling and back-tracing verification. *IEEE Transactions on Information Forensics and Security*, 10(7), 1359–1370.
- **Project Role:** Secondary, higher-resolution validation for `ABANDONED_OBJECT` rule testing across diverse lighting, indoor and outdoor scenes.
- **Files / Sequences:** 11 distinct sequences (`video1.avi` to `video11.avi`) covering night, crowded, and outdoor environments.
- **Technical Specifications:**
  - Resolution: Varies from 720×480 up to 1920×1080.
  - Frame rate: 25 to 30 FPS.
  - Ground truth: Frame index where object is abandoned, bounding box coordinates, and bag owner ID.
- **Known Limitations:**
  - Variable compression and codecs across clips; requires H.264 standardisation via `convert_to_mp4.py`.

---

## 4. AVSS 2007 (i-LIDS Abandoned Baggage) & PETS 2006

- **Source URLs:**
  - AVSS 2007: [https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html](https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html)
  - PETS 2006: [http://www.cvg.rdg.ac.uk/PETS2006/data.html](http://www.cvg.rdg.ac.uk/PETS2006/data.html)
- **License / Terms:** Academic research use only. Mandatory citation acknowledgment for UK Home Office i-LIDS dataset.
- **Required Citation / Acknowledgment:**
  > "Imagery used in this work was provided by the UK Home Office Scientific Development Branch (HOSDB) under the Imagery Library for Intelligent Detection Systems (i-LIDS) programme."
  > Ferryman, J. (2006). PETS 2006 benchmark data. University of Reading.
- **Project Role:** Optional supplementary benchmark for abandoned baggage rule verification under realistic multi-camera transit station conditions.
- **Technical Specifications:** Standard definition PAL (720 × 576 px, 25 FPS).

---

## 5. ChokePoint (Face Recognition & Re-Identification)

- **Source URLs:**
  - Zenodo: [https://zenodo.org/records/815657](https://zenodo.org/records/815657) (DOI: 10.5281/zenodo.815657)
  - Project Home: [http://arma.sourceforge.net/chokepoint/](http://arma.sourceforge.net/chokepoint/)
- **License / Terms:** Creative Commons Attribution-NonCommercial (CC BY-NC). Research and academic evaluation only.
- **Required Citation:**
  > Wong, Y., Chen, S., Mau, S., Sanderson, C., & Lovell, B. C. (2011). Patch-based probabilistic active appearance models for face re-identification. In *IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 9–15.
- **Project Role:** Phase 3 face enrolment, Known vs Unknown classification, and track ID face association.
- **Planned Download Scope (Deferred to Phase 3):**
  - Download only Portal 1 entering sequences (`P1E.tar.xz`, ~66 MB or `P1E_S1.tar.xz`, ~385 MB) and `groundtruth.tar.xz` (~456 kB).
  - Do NOT download the full 12 GB dataset.
- **Technical Specifications:**
  - Frame resolution: 800 × 600 px.
  - Frame rate: 30 FPS.
  - Single-person per frame under portal doorway; eye and nose landmark ground truth.

---

## 6. RWF-2000 (Surveillance Violence & Fight Detection)

- **Source URLs:**
  - GitHub: [https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
  - Hugging Face mirror: [https://huggingface.co/datasets/DanJoshua/RWF-2000](https://huggingface.co/datasets/DanJoshua/RWF-2000)
- **License / Terms:** SMIIP Lab (Prof. Ming Cheng). Academic research and educational purposes only.
- **Required Citation:**
  > Cheng, M., Cai, K., & Li, M. (2021). RWF-2000: An open large-scale video database for violence detection. In *25th International Conference on Pattern Recognition (ICPR)*, pp. 2891–2898.
- **Project Role:** Phase 7 CNN-LSTM model training, validation, and evaluation for the `FIGHTING` activity class.
- **Dataset Structure:**
  - 2,000 video clips captured from real-world surveillance cameras.
  - 5 seconds per clip @ 30 FPS (150 frames each).
  - Official split: 80% train (800 Fight, 800 NonFight), 20% validation/test (200 Fight, 200 NonFight).
- **Execution Policy:** Setup folder skeleton in Phase 0; download deferred to Phase 7.

---

## 7. Running Activity Recognition: Candidate Public Datasets

For the `RUNNING` activity class in Phase 7, the following legitimate public sources have been evaluated:

### Option A: HMDB51 "Run" Class (Recommended)
- **Source:** Serre Lab, Brown University ([http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/))
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** — fully permissible for academic and commercial research with attribution.
- **Contents:** Dedicated `run` directory containing >100 action clips of running, jogging, and sprinting.
- **Pros:** Completely open license, instant download, standardized clip length, high visual diversity.
- **Citation:**
  > Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., & Serre, T. (2011). HMDB: a large video database for human motion recognition. In *ICCV*.

### Option B: VIRAT Video Dataset (Activity Events)
- **Source:** Kitware / DARPA ([http://www.viratdata.org/](http://www.viratdata.org/))
- **License:** Controlled Access under Kitware VIRAT Video Dataset Usage Agreement (click-through authorization required).
- **Contents:** Natural surveillance scenes containing tagged events `Running`, `Entering_Facility`, `Carrying_Object`.
- **Pros:** Authentic overhead CCTV camera angles and resolutions.
- **Cons:** Controlled access requires manual form sign-off.

---

## 8. Anti-Leakage & Data Separation Policy

1. **Quarantine Rule:** All clips copied to `data/samples/` for unit, module, or integration testing during Phases 0–3 are strictly quarantined.
2. **Exclusion from Training:** No clip or source video present in `data/samples/` may be added to `data/datasets/rwf2000/train/`, `data/datasets/rwf2000/val/`, or any future activity training set.
3. **Verification:** `scripts/check_leakage.py` computes SHA-256 hashes across `data/samples/` and `data/datasets/` to enforce zero data leakage prior to Phase 7 model training.
