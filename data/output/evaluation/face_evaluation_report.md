# Phase 3: Open-Set Face Recognition Evaluation Report

**Generated:** 2026-11-24 12:28:08
**Benchmark Dataset:** ChokePoint (`Portal 1`)
**Model:** InsightFace (`buffalo_s` RetinaFace + ArcFace mobile backbone on RTX 4050 GPU)
**Operating Point:** Similarity Threshold = `0.35`, Ambiguity Margin = `0.04`

## 1. Metric Summary & Statistical Confidence Bounds

| Metric | Measured Rate | Target Spec | 95% Confidence Interval (Clopper-Pearson) | Rule of Three (95% Upper Bound) | Statistical Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **False Accept Rate (FAR)** | **0.00%** (0/13) | < 2.0% | [0.00%, 24.71%] | 23.08% | No false accept errors observed in n=13 stranger encounters |
| **False Reject Rate (FRR)** | **0.00%** (0/12) | < 5.0% | [0.00%, 26.46%] | 25.00% | No false reject errors observed in n=12 known encounters |
| **Overall Accuracy** | **100.00%** (25/25) | > 95.0% | [86.28%, 100.00%] | — | 100.0% accuracy observed in n=25 encounters |
| **Pending Rate** | **0.00%** (0/25) | 0.0% | — | — | 0 pending tracks at encounter exit |
| **Decision Latency** | **0.39 s** | < 2.0 s | p50: 0.33s, p95: 0.67s | Max: 0.67s | Rapid convergence before subject exits portal |

## 2. Confusion Matrix (Encounters)

| Actual Class \ Predicted Class | Predicted KNOWN | Predicted UNKNOWN |
| :--- | :---: | :---: |
| **Actual KNOWN (12)** | **12** (True Accept) | **0** (False Reject) |
| **Actual STRANGER (13)** | **0** (False Accept) | **13** (True Reject) |

## 3. Strict 3-Session Dataset Split & Quarantine Protocol

- **Total Subjects:** 25
- **Enrolled / Known Subjects (12):** `['0007', '0009', '0012', '0013', '0014', '0015', '0018', '0019', '0020', '0021', '0022', '0024']`
- **Stranger Subjects (13):** `['0001', '0003', '0004', '0005', '0006', '0010', '0011', '0016', '0017', '0023', '0025', '0026', '0027']`
- **Enrolment Sequence:** `P1E_S1_C1` (Session 1)
- **Tuning Sequence:** `P1E_S2_C1` (Session 2)
- **Validation Sequence:** `P1E_S3_C1` (Session 3)
- **Session Disjointness:** Confirmed 100% session-disjoint across Enrolment, Tuning, and Validation (Zero cross-session contamination).
- **Quarantined Enrolment Frames:** 48 frames excluded from test scoring.
- **Ground Truth Linkage:** 25 / 25 tracks successfully linked to XML ground truth (0 unlinked tracks).

## 4. Tuning Sequence Parameter Sweep Grid (Session 2)

| Similarity Threshold | Ambiguity Margin | FAR (%) | FRR (%) | Accuracy (%) | Mean Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.35 | 0.04 | 0.0% | 0.0% | 100.0% | 0.34 s |
| 0.35 | 0.06 | 0.0% | 0.0% | 100.0% | 0.34 s |
| 0.35 | 0.08 | 0.0% | 0.0% | 100.0% | 0.34 s |
| 0.35 | 0.10 | 0.0% | 0.0% | 100.0% | 0.34 s |
| 0.40 | 0.04 | 0.0% | 0.0% | 100.0% | 0.40 s |
| 0.40 | 0.06 | 0.0% | 0.0% | 100.0% | 0.40 s |
| 0.40 | 0.08 | 0.0% | 0.0% | 100.0% | 0.40 s |
| 0.40 | 0.10 | 0.0% | 0.0% | 100.0% | 0.40 s |
| 0.45 | 0.04 | 0.0% | 8.3% | 96.0% | 0.40 s |
| 0.45 | 0.06 | 0.0% | 8.3% | 96.0% | 0.40 s |
| 0.45 | 0.08 | 0.0% | 8.3% | 96.0% | 0.40 s |
| 0.45 | 0.10 | 0.0% | 8.3% | 96.0% | 0.40 s |
| 0.50 | 0.04 | 0.0% | 25.0% | 88.0% | 0.44 s |
| 0.50 | 0.06 | 0.0% | 25.0% | 88.0% | 0.44 s |
| 0.50 | 0.08 | 0.0% | 25.0% | 88.0% | 0.44 s |
| 0.50 | 0.10 | 0.0% | 25.0% | 88.0% | 0.44 s |
| 0.55 | 0.04 | 0.0% | 41.7% | 80.0% | 0.43 s |
| 0.55 | 0.06 | 0.0% | 41.7% | 80.0% | 0.43 s |
| 0.55 | 0.08 | 0.0% | 41.7% | 80.0% | 0.43 s |
| 0.55 | 0.10 | 0.0% | 41.7% | 80.0% | 0.43 s |

## 5. Per-Frame Quality Filtering & Rejection Diagnostics

| Metric / Stage | Frame Count | Percentage of Inspected |
| :--- | :---: | :---: |
| **Total Appearance Frames Inspected** | 701 | 100.0% |
| **RetinaFace Face Detections** | 137 | 19.5% |
| **Rejected: Undersized Face (< 36px)** | 8 | 1.1% |
| **Rejected: Motion Blur (Laplacian var < 25.0)** | 0 | 0.0% |
| **Rejected: Low Confidence (< 0.55)** | 0 | 0.0% |
| **Passed Quality Filter (Submitted to ArcFace)** | 129 | 18.4% |

### Face Size & Sharpness Distribution (Validation Sequence)

| Dimension Range | Crop Count | | Blur Variance Range | Crop Count |
| :--- | :---: | :--- | :--- | :---: |
| `< 36 px` (Rejected) | 8 | | `< 25.0` (Rejected) | 0 |
| `36 - 50 px` | 35 | | `25.0 - 50.0` | 1 |
| `50 - 80 px` | 72 | | `50.0 - 100.0` | 5 |
| `> 80 px` | 22 | | `> 100.0` | 131 |

## 6. Detailed Validation Results per Encounter

| Subject ID | Ground Truth | Predicted Status | Name | Similarity | Margin | Latency (s) | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0015` | KNOWN | KNOWN | Subject_0015 | 0.5223 | 0.3355 | 0.67 | True Accept (TA) |
| `0005` | STRANGER | UNKNOWN | (Unknown) | 0.1681 | 0.0270 | 0.33 | True Reject (TR) |
| `0003` | STRANGER | UNKNOWN | (Unknown) | 0.1675 | 0.0728 | 0.33 | True Reject (TR) |
| `0019` | KNOWN | KNOWN | Subject_0019 | 0.6517 | 0.5066 | 0.40 | True Accept (TA) |
| `0021` | KNOWN | KNOWN | Subject_0021 | 0.7342 | 0.6115 | 0.33 | True Accept (TA) |
| `0017` | STRANGER | UNKNOWN | (Unknown) | 0.1416 | 0.0094 | 0.33 | True Reject (TR) |
| `0016` | STRANGER | UNKNOWN | (Unknown) | 0.3260 | 0.2110 | 0.33 | True Reject (TR) |
| `0013` | KNOWN | KNOWN | Subject_0013 | 0.6912 | 0.5738 | 0.33 | True Accept (TA) |
| `0009` | KNOWN | KNOWN | Subject_0009 | 0.6893 | 0.5943 | 0.33 | True Accept (TA) |
| `0026` | STRANGER | UNKNOWN | (Unknown) | 0.3807 | 0.2325 | 0.33 | True Reject (TR) |
| `0027` | STRANGER | UNKNOWN | (Unknown) | 0.1480 | 0.0067 | 0.33 | True Reject (TR) |
| `0014` | KNOWN | KNOWN | Subject_0014 | 0.6945 | 0.5486 | 0.33 | True Accept (TA) |
| `0011` | STRANGER | UNKNOWN | (Unknown) | 0.1616 | 0.0398 | 0.33 | True Reject (TR) |
| `0001` | STRANGER | UNKNOWN | (Unknown) | 0.2220 | 0.0868 | 0.40 | True Reject (TR) |
| `0010` | STRANGER | UNKNOWN | (Unknown) | 0.2207 | 0.0270 | 0.33 | True Reject (TR) |
| `0012` | KNOWN | KNOWN | Subject_0012 | 0.7094 | 0.5739 | 0.33 | True Accept (TA) |
| `0020` | KNOWN | KNOWN | Subject_0020 | 0.6237 | 0.4169 | 0.40 | True Accept (TA) |
| `0024` | KNOWN | KNOWN | Subject_0024 | 0.6873 | 0.5879 | 0.67 | True Accept (TA) |
| `0023` | STRANGER | UNKNOWN | (Unknown) | 0.1880 | 0.0005 | 0.40 | True Reject (TR) |
| `0022` | KNOWN | KNOWN | Subject_0022 | 0.7817 | 0.5813 | 0.33 | True Accept (TA) |
| `0025` | STRANGER | UNKNOWN | (Unknown) | 0.1377 | 0.0207 | 0.33 | True Reject (TR) |
| `0018` | KNOWN | KNOWN | Subject_0018 | 0.6879 | 0.6002 | 0.67 | True Accept (TA) |
| `0006` | STRANGER | UNKNOWN | (Unknown) | 0.1923 | 0.0556 | 0.33 | True Reject (TR) |
| `0004` | STRANGER | UNKNOWN | (Unknown) | 0.1542 | 0.0136 | 0.47 | True Reject (TR) |
| `0007` | KNOWN | KNOWN | Subject_0007 | 0.6541 | 0.4323 | 0.47 | True Accept (TA) |
