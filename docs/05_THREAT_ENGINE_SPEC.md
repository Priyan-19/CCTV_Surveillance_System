# 05 — Threat Engine Specification

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

The threat engine is the central intelligence of the system. It collects events from every module and merges them into incidents instead of sending alerts for every individual detection.

## 1. Inputs

| Source | Data |
|--------|------|
| Tracker | Track ID, bbox, path, zone membership, dwell time |
| Face module | Known / Unknown, person ID |
| Activity module | Fighting, running, loitering, absence of objects |
| Rule engine | Restricted entry, line crossing, prolonged stay, abandoned object, occupancy |
| Config | Zone map, thresholds, weights, time window |

## 2. Event Types

| Event type | Detection logic | Default severity |
|------------|-----------------|------------------|
| RESTRICTED_ZONE_ENTRY | Person centre point inside a restricted polygon | High |
| LINE_CROSSING | Track path crosses a virtual line in the forbidden direction | Medium |
| PROLONGED_STAY | Dwell time in a zone exceeds threshold | Medium |
| LOITERING | Rule-based low movement in area OR activity model label | Medium |
| ABANDONED_OBJECT | Bag/luggage stationary beyond threshold and owner track no longer nearby | High |
| OCCUPANCY_EXCEEDED | People count in zone exceeds limit (example: 5 detected against limit of 4) | Medium |
| FIGHTING | CNN-LSTM label with confidence above threshold | High |
| RUNNING | CNN-LSTM label with confidence above threshold | Low |
| UNKNOWN_PERSON | Face not matched to known database | Low (modifier) |

## 3. Event Record

```
event_id, camera_id, track_id, person_id, object_id,
event_type, timestamp, location (zone id + coordinates),
severity, confidence, details
```

## 4. Correlation Logic

Events are grouped into one incident when they share enough context.

**Correlation keys:** Track ID, person ID, object ID, action/behaviour, time of event, location, event type.

**Rules:**

1. Open a new incident when an event arrives that is not linked to any open incident.
2. Link a new event to an open incident if **any** of the following is true, and it falls inside the correlation time window:
   - same Track ID or person ID;
   - same object ID (for example, an abandoned bag linked to the person who carried it);
   - same zone/location within the time window.
3. A bag detected near a person track and later left behind links the object to that person.
4. An incident stays open while new related events arrive. It closes after a quiet period with no new events.
5. Events outside the window start a separate incident.

**Default window:** 120 seconds between related events (configurable). Quiet-period close: 300 seconds (configurable).

## 5. Threat Scoring

Starting values are defaults. They are configuration, to be tuned during testing.

### 5.1 Event weights

| Event | Weight |
|-------|--------|
| FIGHTING | 50 |
| RESTRICTED_ZONE_ENTRY | 40 |
| ABANDONED_OBJECT | 30 |
| LINE_CROSSING | 20 |
| LOITERING | 20 |
| PROLONGED_STAY | 15 |
| OCCUPANCY_EXCEEDED | 15 |
| RUNNING | 10 |

### 5.2 Modifiers

| Condition | Adjustment |
|-----------|------------|
| Subject is Unknown person | +15 |
| Subject is Known and authorised for the zone | −20 |
| Three or more distinct event types in one incident | +10 |
| Event happens in a high-risk zone | +10 |
| Repeated same event type | +25% of that weight, once |

### 5.3 Formula

```
score = clamp( sum(event weights, unique types) + modifiers , 0, 100 )
```

### 5.4 Levels

| Score | Level | Action |
|-------|-------|--------|
| 0–29 | Low | Log only, visible in dashboard |
| 30–59 | Medium | Dashboard alert |
| 60–79 | High | Dashboard alert + notification with evidence |
| 80–100 | Critical | Priority alert + notification with evidence |

### 5.5 Worked Example

Unknown person enters server room, stays, leaves a bag.

| Item | Value |
|------|-------|
| RESTRICTED_ZONE_ENTRY | 40 |
| PROLONGED_STAY | 15 |
| ABANDONED_OBJECT | 30 |
| Unknown person modifier | +15 |
| 3+ distinct events modifier | +10 |
| **Score** | **100 → Critical** |

## 6. Incident Output

| Field | Description |
|-------|-------------|
| incident_no | Unique, sequential (e.g. INC-000045) |
| subject | Person (Track ID, identity) and/or object involved |
| status | New / Acknowledged / Resolved / False Alarm |
| place | Camera ID and zone |
| time | Start and last-event time |
| activities | List of suspicious activities and events |
| threat_level | Low / Medium / High / Critical |
| threat_score | 0–100 |
| evidence | Person image, face image, object image, video clip |

## 7. Alerting Rules

- Alert once per incident creation and again only when the level increases.
- Include evidence links in every alert.
- Single low-weight events do not create notifications.
- Medium alerts stay on dashboard; High and Critical also notify.

## 8. Tuning and Validation

- Weights, thresholds and window are config values.
- Replay recorded test videos and compare incident output against expected outcomes.
- Measure alert reduction versus one-alert-per-event baseline.

## 9. Design Refinements (Phase 4 Updates)

1. **Repeated Event Modifier (Clarification a):** +25% of the highest-weight repeated event type, applied **once per incident** (not compounding with successive repetitions).
2. **Stratified Unknown Identity Modifier (Clarification b):**
   - Verified stranger against enrolled gallery: **+15**
   - Overhead / wide-angle / unconstrained blurred face (`no_usable_face`): **+5**
   - Biometric status pending: **0**
3. **Authorized Person Deduction (Clarification c):** -20 deduction is applied **only if every person event** in the incident occurred in an authorized zone. Any restricted zone entry immediately voids the deduction.
4. **Audit Trace & History (Clarification d):** Every scoring decision appends a timestamped explanation and modifier breakdown to `incident.score_history` and `incident.details`.
5. **Evidence Collection Deduplication:** Snapshot crops are collected once per incident upon detection. Video clip re-encoding is performed only upon alert escalation and incident closure to eliminate frame-rate degradation.
