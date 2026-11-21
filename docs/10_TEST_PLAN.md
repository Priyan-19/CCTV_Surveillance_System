# 10 — Test Plan

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Objectives

Verify that each module works, that modules work together, that performance targets are met, and that incidents are correct and evidenced.

## 2. Test Levels

| Level | Scope |
|-------|-------|
| Unit | Single functions: zone geometry, line crossing, dwell timer, scoring, correlation |
| Module | Detector, tracker, face, activity, rule engine, threat engine individually |
| Integration | Full pipeline from video to stored incident |
| System | Dashboard + API + pipeline on real/recorded footage |
| Performance | FPS, latency, VRAM on RTX 4050 |
| Security | Access control, data protection |
| Acceptance | Scenario-based against PRD |

## 3. Test Data

| Data | Purpose |
|------|---------|
| Short clips of people walking, standing, entering zones (CAVIAR) | Detection, tracking, zone rules |
| Clips with a bag placed and left (CAVIAR / ABODA) | Abandoned-object rule |
| Clips with more than the occupancy limit (CAVIAR crowd) | Occupancy rule |
| Clips with enrolled vs non-enrolled people (ChokePoint) | Face recognition |
| Fight / non-fight clips (CAVIAR fight clip / RWF-2000 test split) | Activity model |
| Public running, loitering, normal clips (HMDB51 / CAVIAR) | Activity model |
| Multi-event scenario clip (enter, stay, leave bag, exit) | Correlation |
| Long clip (30+ minutes) | Stability and memory |

> **Datasets and Data Handling Note (Phase 0):**
> Testing strictly utilizes public academic benchmark datasets (CAVIAR, ABODA, ChokePoint, RWF-2000, HMDB51). No self-recorded footage is required. Spatial rules (restricted zones, virtual lines, occupancy counts) are virtual boundaries drawn over public video footage via `configs/zones.yaml`. Test clips residing in `data/samples/` are isolated and quarantined from training pipelines to prevent data leakage.

## 4. Functional Test Cases

### Detection and Tracking
| ID | Test | Expected |
|----|------|----------|
| T-01 | Person walks through frame | Detected with class and confidence |
| T-02 | Same person across 100 frames | Same Track ID retained |
| T-03 | Person briefly occluded | Track ID recovered |
| T-04 | Two people cross paths | IDs not swapped (or swap counted) |

### Face Recognition
| ID | Test | Expected |
|----|------|----------|
| T-05 | Enrolled person appears | Labelled Known with name |
| T-06 | Non-enrolled person appears | Labelled Unknown |
| T-07 | Small or blurred face | No decision forced; stays pending or Unknown |

### Rule Events
| ID | Test | Expected |
|----|------|----------|
| T-08 | Person enters restricted polygon | RESTRICTED_ZONE_ENTRY, High |
| T-09 | Person crosses virtual line in forbidden direction | LINE_CROSSING |
| T-10 | Person stays beyond dwell threshold | PROLONGED_STAY |
| T-11 | Bag left, owner walks away | ABANDONED_OBJECT |
| T-12 | Five people against limit of four | OCCUPANCY_EXCEEDED, Medium |
| T-13 | Four people against limit of four | No alert |

### Activity
| ID | Test | Expected |
|----|------|----------|
| T-14 | Fight clip | FIGHTING above threshold |
| T-15 | Normal walking clip | No fighting event |
| T-16 | Running clip | RUNNING |
| T-17 | Standing in same area for long | LOITERING |

### Threat Engine
| ID | Test | Expected |
|----|------|----------|
| T-18 | Single low event | Logged, no notification |
| T-19 | Enter + stay + abandoned bag + unknown | One incident, Critical, score as per spec |
| T-20 | Two unrelated events far apart in time | Two incidents |
| T-21 | Same person repeats event | Incident updated, not duplicated |
| T-22 | Known authorised person in zone | Score reduced per modifier |

### Evidence
| ID | Test | Expected |
|----|------|----------|
| T-23 | Incident created | Person, face, object images and clip exist |
| T-24 | Open evidence in dashboard | Files load and play |
| T-25 | Face not visible | Incident still saved; face image marked unavailable |

### Dashboard and API
| ID | Test | Expected |
|----|------|----------|
| T-26 | New incident | Appears without refresh |
| T-27 | Keyword search | Matching incidents returned |
| T-28 | Status change | Updated across clients |
| T-29 | Live view | Boxes, confidence, Track IDs shown |
| T-30 | Camera offline | Tile shows offline; other cameras unaffected |

## 5. Performance Tests

| ID | Test | Target |
|----|------|--------|
| P-01 | Single 1080p stream, 5 people | 18+ FPS |
| P-02 | Per-frame processing time | 20–45 ms |
| P-03 | Two simultaneous streams | Report FPS per stream |
| P-04 | 60-minute run | No memory growth, no crash |
| P-05 | VRAM usage | Within 6 GB (target under about 5 GB) |
| P-06 | Alert latency | Incident visible within a few seconds |

## 6. Model Evaluation (Quantitative)

| Metric | Applied to |
|--------|-----------|
| Accuracy, Precision, Recall, F1, confusion matrix | Activity model on held-out test set and self-recorded clips |
| Match rate / false match rate | Face recognition |
| ID switches, track fragmentation | Tracking |
| Alerts per hour versus one-alert-per-event baseline | Threat engine correlation benefit |

## 7. Security Tests

| ID | Test | Expected |
|----|------|----------|
| S-01 | Access API without token | 401 |
| S-02 | Officer calls admin endpoint | 403 |
| S-03 | Auditor tries to change status | 403 |
| S-04 | Password storage | Hashed |
| S-05 | Evidence URL without authorisation | Denied |

## 8. Acceptance Criteria

The system is accepted when:

1. All PRD functional requirements demonstrate working on recorded and live input.
2. The multi-event scenario produces a single correct incident with complete evidence.
3. Restricted-presence (High) and occupancy (Medium) alerts reproduce correctly.
4. Sustained FPS meets target on RTX 4050.
5. Activity model metrics are reported.
6. Role-based access works.

## 9. Defect Handling

Severity: Blocker, Major, Minor. Blockers and Majors fixed before sign-off. Each defect records steps, input video, expected and actual result.
