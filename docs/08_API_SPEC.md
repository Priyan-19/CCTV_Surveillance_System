# 08 — API Specification

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection
**Base URL:** `/api/v1`
**Format:** JSON. Authentication by bearer token. Role checks on every endpoint.

## 1. Authentication

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| POST | `/auth/login` | Username and password, returns token | all |
| POST | `/auth/logout` | Invalidate token | all |
| GET | `/auth/me` | Current user and role | all |

## 2. Cameras

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/cameras` | List cameras with status | all |
| POST | `/cameras` | Add camera | admin |
| PUT | `/cameras/{id}` | Update camera | admin |
| DELETE | `/cameras/{id}` | Remove camera | admin |
| POST | `/cameras/{id}/start` | Start processing | admin |
| POST | `/cameras/{id}/stop` | Stop processing | admin |
| GET | `/cameras/{id}/stream` | Live annotated MJPEG stream | officer, admin |
| GET | `/cameras/{id}/metrics` | FPS, processing ms, people count, alert state | all |

## 3. Zones

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/cameras/{id}/zones` | List zones | all |
| POST | `/cameras/{id}/zones` | Create zone (polygon, line, occupancy) | admin |
| PUT | `/zones/{zone_id}` | Update | admin |
| DELETE | `/zones/{zone_id}` | Delete | admin |

Zone body example:
```json
{
  "name": "Server room",
  "type": "restricted",
  "geometry": [[120,80],[560,80],[560,420],[120,420]],
  "dwell_threshold_s": 30,
  "high_risk": true
}
```

## 4. Persons and Face Enrolment

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/persons` | List known persons | admin |
| POST | `/persons` | Create person | admin |
| POST | `/persons/{id}/faces` | Upload face images, create embeddings | admin |
| PUT | `/persons/{id}` | Update details and authorised zones | admin |
| DELETE | `/persons/{id}` | Delete person and embeddings | admin |

## 5. Incidents

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/incidents` | List with filters | all |
| GET | `/incidents/{id}` | Full incident with events and evidence | all |
| PATCH | `/incidents/{id}/status` | Set acknowledged / resolved / false_alarm | officer, admin |
| POST | `/incidents/{id}/notes` | Add note | officer, admin |
| GET | `/incidents/search?q=` | Keyword search | all |

**List filters:** `camera_id`, `threat_level`, `status`, `from`, `to`, `q`, `page`, `page_size`.

Incident response example:
```json
{
  "incident_no": "INC-000045",
  "camera_id": "CAM03",
  "place": "Main entrance - Restricted area",
  "subject": {"track_id": 12, "identity": "unknown", "objects": ["bag"]},
  "status": "new",
  "activities": ["Restricted zone entry", "Prolonged stay", "Abandoned object"],
  "threat_level": "critical",
  "threat_score": 100,
  "start_time": "2026-01-01T10:15:02Z",
  "last_event_time": "2026-01-01T10:18:40Z",
  "events": [],
  "evidence": [
    {"type": "person_image", "url": "/api/v1/evidence/101"},
    {"type": "face_image", "url": "/api/v1/evidence/102"},
    {"type": "object_image", "url": "/api/v1/evidence/103"},
    {"type": "video_clip", "url": "/api/v1/evidence/104"}
  ]
}
```

## 6. Evidence

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/evidence/{id}` | Download image or clip | officer, admin, auditor |

## 7. Settings

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/settings/threat` | Weights, levels, window | admin |
| PUT | `/settings/threat` | Update weights, levels, window | admin |
| GET | `/settings/models` | Current model versions and thresholds | admin |
| PUT | `/settings/models` | Update thresholds | admin |

## 8. Users

| Method | Endpoint | Description | Roles |
|--------|----------|-------------|-------|
| GET | `/users` | List | admin |
| POST | `/users` | Create | admin |
| PUT | `/users/{id}` | Update role | admin |
| DELETE | `/users/{id}` | Delete | admin |

## 9. WebSocket

**Endpoint:** `/ws/live` (token required)

Client may subscribe to cameras: `{"action":"subscribe","camera_ids":["CAM02","CAM03"]}`

| Message type | Payload |
|--------------|---------|
| `camera_status` | `{camera_id, fps, processing_ms, people_count, alert_state, status}` |
| `detections` | `{camera_id, timestamp, objects:[{track_id, class, bbox, confidence, identity}]}` |
| `incident_created` | Incident summary |
| `incident_updated` | Changed fields |

## 10. Error Format

```json
{"error": {"code": "NOT_FOUND", "message": "Incident not found"}}
```

| HTTP | Meaning |
|------|---------|
| 400 | Invalid input |
| 401 | Not authenticated |
| 403 | Role not allowed |
| 404 | Not found |
| 409 | Conflict |
| 500 | Server error |
