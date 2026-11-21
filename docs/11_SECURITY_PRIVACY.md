# 11 — Security and Privacy

**Project:** An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## 1. Why This Matters

The system processes video of people and stores face data and incident evidence. This is sensitive and must be protected and used responsibly.

## 2. Access Control

| Role | Permissions |
|------|-------------|
| Admin | Everything: cameras, zones, persons, settings, users, incidents |
| Officer | Live view, view incidents and evidence, change incident status, add notes |
| Auditor | Read-only access to incidents, evidence and search |

- Token-based authentication on all API and WebSocket access.
- Role check on every endpoint.
- Session expiry and logout.

## 3. Data Protection

| Data | Protection |
|------|-----------|
| Passwords | Salted hash (bcrypt or argon2) |
| Face embeddings | Stored as vectors; access limited to admin and the pipeline |
| Enrolment images | Restricted folder, admin only |
| Evidence files | Served only through authorised API, no public URLs |
| Database | Local access only; backups encrypted |
| Transport | HTTPS between browser and server in deployment |
| Secrets (camera URLs, tokens) | Environment variables or secret file, not in code |

## 4. Privacy Principles

1. **Local processing:** video is analysed on the local machine and is not sent to external services.
2. **Minimal storage:** only incident evidence is kept; continuous video is not stored.
3. **Purpose limitation:** face data is used only to decide Known or Unknown for security.
4. **Consent for enrolment:** persons are enrolled only with their knowledge and consent.
5. **Retention limit:** evidence deleted after the configured period (default 90 days).
6. **Right to remove:** an enrolled person can be deleted together with all embeddings and images.
7. **Optional blurring:** faces of unknown, non-incident people are not saved; only incident face images are stored.

## 5. Audit and Accountability

- Log logins, enrolment, setting changes, evidence access and status changes with user and time.
- Incident actions are recorded in `incident_actions`.

## 6. Responsible Use

- The system supports human decision-making; alerts are for verification by an officer, not automatic punishment.
- Show confidence and evidence, not just a verdict.
- Allow marking False Alarm and use these to review thresholds.
- Monitor and document false-positive and false-negative behaviour, including differences in face recognition performance across conditions.
- Comply with applicable local laws and notify people that CCTV analytics are in use.

## 6.1 Notifications

Notification messages include incident number, level, place, time and a link. Face images are not sent through external messaging channels unless the administrator explicitly enables this.

## 7. Threats and Mitigations

| Threat | Mitigation |
|--------|-----------|
| Unauthorised access to evidence | Role checks, authenticated evidence endpoint |
| Credential theft | Hashed passwords, expiry, HTTPS |
| Face database leak | Restricted access, encrypted backups |
| Camera feed tampering | Source whitelisting, status monitoring |
| Denial of service on pipeline | Bounded queues, per-camera isolation |
| Spoofed or misleading alerts | Alerts generated only by server pipeline, signed internal calls |
| Excessive false alarms | Correlation, thresholds, feedback loop |
