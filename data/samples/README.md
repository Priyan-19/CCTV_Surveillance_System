# Test Video Clips Guide (data/samples/)

To validate our intelligent CCTV detection, tracking, rules, and threat engine pipeline, collect at least 5 sample video clips into this directory.

## Required Test Scenarios

1. **Walking People (`sample_walking.mp4`)**
   - **Scenario:** Normal pedestrian motion across camera frame (1-3 individuals).
   - **Validates:** YOLO person detection, ByteTrack ID persistence, bounding box stability.

2. **Restricted-Area Entry (`sample_restricted.mp4`)**
   - **Scenario:** Person walking across a boundary or entering a marked zone (e.g., behind a desk or doorway).
   - **Validates:** Zone polygon point-in-polygon math, `RESTRICTED_ZONE_ENTRY` rule trigger.

3. **Bag Left Behind (`sample_abandoned_bag.mp4`)**
   - **Scenario:** Person walks with a backpack/duffel bag, places it down on a chair/floor, and walks away out of frame.
   - **Validates:** Association between person track and object track, dwell timer, `ABANDONED_OBJECT` rule trigger.

4. **Crowd / 5+ People (`sample_occupancy.mp4`)**
   - **Scenario:** 5 or more people walking or gathering in one region.
   - **Validates:** Multi-target tracker performance, occlusion handling, `OCCUPANCY_EXCEEDED` rule trigger.

5. **Running (`sample_running.mp4`)**
   - **Scenario:** Person jogging or sprinting quickly across the scene.
   - **Validates:** Rapid motion tracking, CNN-LSTM running activity detection, speed calculation.

6. **Fighting / Violence (`sample_fighting.mp4`)**
   - **Scenario:** Two individuals scuffling, pushing, or fighting.
   - **Validates:** CNN-LSTM violence detection, threat engine high-priority trigger (`FIGHTING`).

7. **Normal Daily Activity (`sample_normal.mp4`)**
   - **Scenario:** Individuals sitting, working, chatting peacefully.
   - **Validates:** Baseline false positive prevention.

8. **Self-Recorded Phone Clips (`sample_phone_*.mp4`)**
   - **Scenario:** Clips recorded with your smartphone in realistic lighting and angles.

---

## Free Public Video Sources
- **VIRAT Video Dataset:** Classic surveillance video benchmark with realistic angles and events (abandoned bag, running, loitering).
- **RWF-2000 / UCF-Crime Datasets:** Benchmark video clips for surveillance fight and crime detection.
- **Pexels / Pixabay Videos:** Search for `"CCTV hallway"`, `"pedestrians walking"`, `"airport security"`, `"office security"`. Free to download in 1080p MP4.

## How to Record Your Own Phone Clips
- **Format:** MP4 or MOV, H.264 video.
- **Orientation:** Landscape (16:9), 1080p (1920x1080), 30 FPS.
- **Angle:** Mount or hold the phone steady at eye-level or slightly elevated (simulating a wall/ceiling CCTV mount).
- **Length:** 10 to 30 seconds per clip is ideal.
