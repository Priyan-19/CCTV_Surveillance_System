# 01 — Product Requirements Document (PRD)

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection
**Version:** 1.0

## 1. Overview

This project is an AI-based CCTV monitoring system. It analyses live or recorded video automatically and tells security staff what is happening, instead of requiring a person to watch every camera continuously. Its distinguishing feature is that it does not raise one alert per detection. It combines related detections into **one security incident** with a threat level and visual evidence.

## 2. Problem Statement

1. Traditional CCTV monitoring needs security staff to watch screens for long periods. Attention drops and events are missed.
2. Footage is mostly reviewed after something has happened, so incidents are not prevented.
3. Existing AI tools work in isolation (object detection, face recognition, violence detection, anomaly detection). Each produces its own separate output, so a situation made of several related events is reported as unrelated alerts.
4. Too many separate alerts overwhelm officers and make it hard to judge what matters.

## 3. Vision and Goals

**Vision:** Give every CCTV installation an AI layer that understands a situation as a whole.

| Goal | Description |
|------|-------------|
| G1 | Detect people, vehicles and luggage automatically in every frame |
| G2 | Keep a stable identity for each object across frames |
| G3 | Recognise whether a person is known or unknown |
| G4 | Recognise suspicious activity over time (fighting, running, loitering, object left behind) |
| G5 | Correlate events into a single incident with threat level and score |
| G6 | Provide evidence (images and video) so officers can verify quickly |
| G7 | Reduce manual camera watching and the number of unnecessary notifications |
| G8 | Provide a web dashboard for live monitoring, alerts and incident search |

## 4. Non-Goals (Version 1)

- Automatic reporting to police or external authorities
- Mobile application
- Cross-site multi-camera person re-identification
- Edge-device deployment (planned future work)
- Training new models from scratch

## 5. Users and Personas

| Persona | Needs |
|---------|-------|
| Security Officer | See live status, receive alerts, verify with evidence, act quickly |
| Security Administrator | Configure cameras, zones, limits and known-person database; manage users |
| Auditor / Reviewer | Search past incidents by keyword, date, camera, threat level |

## 6. Use Cases

| ID | Use case |
|----|----------|
| UC1 | Officer watches live camera views with detected people, confidence scores and Track IDs |
| UC2 | System raises a high-severity alert when a person enters a restricted area |
| UC3 | System raises a medium-severity alert when the number of people exceeds the occupancy limit |
| UC4 | System correlates restricted entry + long stay + abandoned bag + unknown face into one incident |
| UC5 | Officer opens an incident and reviews person image, face image, object image and video clip |
| UC6 | Officer searches past incidents using keywords |
| UC7 | Admin enrols a known person with face images |
| UC8 | Admin draws restricted zones, lines and occupancy limits on a camera view |
| UC9 | Officer processes a recorded video file for investigation |

## 7. Functional Requirements

### 7.1 Video Input
- FR-1: Accept live camera streams (RTSP / webcam) and recorded video files.
- FR-2: Split video into frames and resize to the model input size.
- FR-3: Support multiple cameras, each identified by a camera ID (e.g. CAM02, CAM03).

### 7.2 Object Detection
- FR-4: Detect people, vehicles and luggage in every processed frame.
- FR-5: Output class, bounding box coordinates and confidence score per detection.

### 7.3 Object Tracking
- FR-6: Assign a unique Track ID to each object and keep it stable across frames.
- FR-7: Record movement path and time spent in each zone per track.

### 7.4 Person Identification
- FR-8: Run face recognition on detected persons.
- FR-9: Classify each person as Known (with name/ID) or Unknown.
- FR-10: Attach the identity to the person's Track ID for the rest of the track.

### 7.5 Activity Analysis
- FR-11: Analyse sequences of frames with CNN-LSTM.
- FR-12: Recognise fighting, running, loitering and absence of objects.

### 7.6 Security Event Detection
- FR-13: Detect restricted-zone entry.
- FR-14: Detect virtual line crossing.
- FR-15: Detect prolonged stay (dwell time above threshold).
- FR-16: Detect loitering.
- FR-17: Detect abandoned / lost objects.
- FR-18: Detect occupancy limit exceeded (e.g. 5 people against a limit of 4).
- FR-19: Each event carries type, time, location and severity.

### 7.7 Event Correlation and Threat Analysis
- FR-20: Correlate events using Track ID, person ID, object ID, activity, time, location and event type.
- FR-21: Merge events that occur close together in time and space into one incident.
- FR-22: Compute threat level and threat score per incident.

### 7.8 Incident Generation
- FR-23: Each incident contains: incident number, person/object involved, status, place, time, suspicious activities, threat level and score.

### 7.9 Evidence Collection
- FR-24: Save person image, face image, object image, incident video clip, time and location.

### 7.10 Alerts and Dashboard
- FR-25: Notify officers of serious incidents with evidence attached.
- FR-26: Dashboard shows live footage, tracked objects, current activity and threat level.
- FR-27: Dashboard shows camera status, FPS, processing time per frame, people count and active alerts (as in the runtime table of the paper).
- FR-28: Incidents are stored in a database and searchable by keyword.
- FR-29: Incident status can be updated (New, Acknowledged, Resolved, False Alarm).

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Processing speed of about 18–22 FPS per stream at 1080p input |
| NFR-2 | Per-frame processing time between about 20 ms and 45 ms |
| NFR-3 | Alert shown on dashboard within a few seconds of the triggering event |
| NFR-4 | Evidence is retrievable for every incident |
| NFR-5 | Only authorised users can view incidents and evidence |
| NFR-6 | System continues running if one camera fails |
| NFR-7 | Zones, limits and thresholds are configurable without code changes |

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Sustained FPS per stream | 18 or higher |
| Detection and tracking stability | Track ID persists through short occlusions |
| Alert reduction | Fewer notifications than one-alert-per-detection baseline |
| Activity classification | Precision, recall and F1 reported on labelled ground truth |
| Incident usefulness | Every incident has complete evidence set |

## 10. Assumptions and Constraints

- Development hardware is a single laptop with an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
- Pretrained models are used wherever possible. Only the activity model is fine-tuned.
- Known-person database is provided by the administrator.
- Activity classification evaluation against labelled ground truth is required for the final report.

## 11. Future Scope

Advanced deep learning for more complex unusual behaviour, multi-camera correlation to follow a person or object over a large area, edge computing for faster processing, improved face recognition, mobile application, improved notification system, and automated reporting of security violations.
