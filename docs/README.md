# An Intelligent CCTV Surveillance System Using YOLO, ByteTrack, CNN-LSTM and Face Recognition for Automated Threat Detection

## Documentation Index

This system converts raw CCTV video into security incidents. It detects objects, tracks them across frames, identifies faces, analyses behaviour over time, correlates all findings in a threat engine, and presents a single evidence-backed incident on a web dashboard.

## Document Set

| # | File | Purpose |
|---|------|---------|
| 01 | [01_PRD.md](01_PRD.md) | Product Requirements: problem, goals, users, features, success metrics |
| 02 | [02_TRD.md](02_TRD.md) | Technical Requirements: stack, hardware, performance, constraints |
| 03 | [03_SYSTEM_ARCHITECTURE.md](03_SYSTEM_ARCHITECTURE.md) | Layers, modules, deployment, component responsibilities |
| 04 | [04_DATA_FLOW.md](04_DATA_FLOW.md) | End-to-end data flow, DFD levels, sequence of one incident |
| 05 | [05_THREAT_ENGINE_SPEC.md](05_THREAT_ENGINE_SPEC.md) | Event types, correlation rules, threat score and level logic |
| 06 | [06_ML_MODEL_PLAN.md](06_ML_MODEL_PLAN.md) | Models, pretrained vs fine-tuned, datasets, training and evaluation plan |
| 07 | [07_DATABASE_SCHEMA.md](07_DATABASE_SCHEMA.md) | Tables, fields, relationships, evidence storage layout |
| 08 | [08_API_SPEC.md](08_API_SPEC.md) | REST and WebSocket interfaces |
| 09 | [09_UI_UX_SPEC.md](09_UI_UX_SPEC.md) | Dashboard screens, components, alert presentation |
| 10 | [10_TEST_PLAN.md](10_TEST_PLAN.md) | Test levels, cases, acceptance criteria |
| 11 | [11_SECURITY_PRIVACY.md](11_SECURITY_PRIVACY.md) | Access control, face-data protection, retention |
| 12 | [12_PROJECT_PLAN.md](12_PROJECT_PLAN.md) | Phases, milestones, folder structure, risks |

## Glossary

| Term | Meaning |
|------|---------|
| Detection | A single object found in a single frame (class, bounding box, confidence) |
| Track | The same object followed across frames, carrying one Track ID |
| Event | One rule-triggered or model-triggered security observation (e.g. restricted zone entry) |
| Incident | A group of correlated events about the same subject, place and time window |
| Threat level / score | Severity label and numeric value assigned to an incident |
| Evidence | Person image, face image, object image and video clip attached to an incident |
| Zone | A polygon drawn on a camera view (restricted area, line, occupancy area) |
