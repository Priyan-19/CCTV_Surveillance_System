# 09 — UI / UX Specification

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Design Goals

- The officer understands the situation at a glance.
- Alerts show evidence immediately so verification takes seconds.
- Severity is obvious through colour and label, not colour alone.
- Live monitoring never blocks on slow operations.

## 2. Navigation

Left menu: **Live Monitor, Incidents, Cameras, Zones, Known Persons, Settings, Users** (Admin-only items hidden for other roles).

## 3. Screens

### 3.1 Login
Username, password, sign-in. Error message on failure.

### 3.2 Live Monitor (main screen)

| Area | Content |
|------|---------|
| Camera grid / main view | Live annotated video with boxes, confidence scores and Track IDs, identity label (name or Unknown) |
| Alert banner on a camera | Shows alert type and severity, e.g. "Restricted presence (High)" |
| Camera status panel | Camera name and ID, online/offline, FPS, processing time (ms), people count, alert state |
| Current tracks panel | List of active tracked objects with ID, class, identity, zone, dwell time, current activity |
| Active incidents panel | Newest incidents with level, place, time |
| Threat indicator | Current highest threat level |

Camera runtime table layout (matches the reference dashboard):

| Camera | People | FPS | Proc. (ms) | Alert |
|--------|--------|-----|------------|-------|
| Production floor | 3 | 18.22 | 27.4 | None |
| Main entrance | 2 | 22.37 | 44.8 | Restricted presence (High) |
| Parking area | 1 | 18.94 | 21.6 | None |
| Warehouse aisle | 5 | 18.01 | 31.4 | Occupancy limit (Medium) |

### 3.3 Incident List
- Table: incident number, time, camera/place, subject, threat level, score, status.
- Filters: date range, camera, threat level, status.
- Keyword search box.
- Sort by time or score.

### 3.4 Incident Detail
| Section | Content |
|---------|---------|
| Header | Incident number, threat level badge, score, status |
| Summary | Person/object involved, place, start and last time, suspicious activities |
| Timeline | Ordered events that formed the incident |
| Evidence | Person image, face image, object image, video player for clip |
| Actions | Acknowledge, Resolve, Mark false alarm, Add note |
| History | Who changed what and when |

### 3.5 Cameras
List, add/edit/remove, source URL, start/stop processing, status.

### 3.6 Zones Editor (Admin)
- Camera snapshot as canvas.
- Draw polygons (restricted, occupancy) and lines (virtual lines with direction arrow).
- Set name, dwell threshold, occupancy limit, high-risk flag.
- Save and apply without restart.

### 3.7 Known Persons (Admin)
- List, add person, upload 3–5 face photos, set authorised zones, delete.

### 3.8 Settings (Admin)
- Threat weights, level cut-offs, correlation window.
- Detection and face-match thresholds.
- Notification channel setup.
- Retention period.

### 3.9 Users (Admin)
Create users, set role, reset password.

## 4. Alert Presentation

| Level | Colour | Behaviour |
|-------|--------|-----------|
| Low | Grey | Logged in list only |
| Medium | Amber | Banner on camera + entry in Active Incidents |
| High | Red | Banner, toast with evidence thumbnail, notification sent |
| Critical | Dark red | Persistent toast, sound option, notification sent |

Every alert shows: level, place, time, short description, evidence thumbnail, "Open incident" button.

## 5. Interaction Rules

- New incidents appear without page refresh (WebSocket).
- Clicking an alert opens Incident Detail.
- Status changes update all open clients.
- Video clip plays inline.
- Large evidence images open in a viewer.

## 6. Accessibility and Responsiveness

- Colour accompanied by text labels and icons.
- Keyboard operable controls.
- Desktop-first; usable on tablet width.

## 7. Empty and Error States

- No incidents: "No incidents in this period."
- Camera offline: grey tile with "Camera offline" and last seen time.
- Evidence missing: placeholder with reason.
- API failure: inline message with retry.
