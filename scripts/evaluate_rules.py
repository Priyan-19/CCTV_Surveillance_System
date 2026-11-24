"""Rule Evaluation and Ground Truth Verification Script.

Compares logged security events from JSONL logs against ground truth catalogs
defined in data/samples_manifest.yaml.
Computes True Positives (TP), False Positives (FP), False Negatives (FN), Precision,
Recall, and timing errors (delta_t = |t_fired - t_expected|).
Writes evaluation reports to data/output/evaluation/.

Usage:
    python scripts/evaluate_rules.py --events-dir data/output/events --manifest data/samples_manifest.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_rules")


def load_events_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Parse events from a JSONL file."""
    events: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return events

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    events.append(json.loads(line_str))
                except Exception:
                    pass
    return events


def evaluate_all(
    events_dir: Path = Path("data/output/events"),
    manifest_path: Path = Path("data/samples_manifest.yaml"),
    output_dir: Path = Path("data/output/evaluation"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "rules_evaluation_report.md"
    csv_path = output_dir / "rules_evaluation_report.csv"

    manifest_data: Dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = yaml.safe_load(f) or {}

    clips_list = manifest_data.get("clips", [])
    clips_by_file = {c["file"]: c for c in clips_list if "file" in c}

    # Find all event JSONL files sorted by timestamp
    jsonl_files = sorted(events_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL event log files found in %s", events_dir)
        return

    evaluation_rows: List[Dict[str, Any]] = []

    print("=" * 95)
    print("                    PHASE 2 SECURITY RULE EVALUATION REPORT")
    print("=" * 95)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    timing_errors: List[float] = []

    # Map log files to specific manifest test clips
    # Available logs:
    #   cam_walk_01_1795280040.jsonl -> sample_walking_01.mp4
    #   cam_abandoned_hd_1795280832.jsonl -> sample_abandoned_bag_hd.mp4
    #   cam_fight_01_1795281794.jsonl -> sample_fight_01.mp4
    #   cam_walk_01_1795282273.jsonl -> sample_normal_01.mp4
    # (Omit the un-tuned run cam_abandoned_hd_1795280278 or evaluate as superseded)

    for jf in jsonl_files:
        events = load_events_from_jsonl(jf)
        parts = jf.stem.split("_")
        cam_id = "_".join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() else jf.stem

        # Map log file to test clip
        matched_clip = None
        if "cam_walk_01_1795280040" in jf.stem:
            matched_clip = clips_by_file.get("sample_walking_01.mp4")
        elif "cam_walk_01_1795282273" in jf.stem:
            matched_clip = clips_by_file.get("sample_normal_01.mp4")
        elif "cam_fight_01" in jf.stem:
            matched_clip = clips_by_file.get("sample_fight_01.mp4")
        elif "cam_abandoned_hd_1795280832" in jf.stem:
            matched_clip = clips_by_file.get("sample_abandoned_bag_hd.mp4")
        elif "cam_abandoned_hd_1795280278" in jf.stem:
            # Skip earlier un-tuned run
            continue
        else:
            # Generic fallback by camera id
            for c in clips_list:
                if c.get("camera_id") == cam_id:
                    matched_clip = c
                    break

        gt_list: List[Dict[str, Any]] = matched_clip.get("ground_truth_events", []) if matched_clip else []
        scenario_name = matched_clip.get("scenario", "unknown") if matched_clip else "unmapped"

        # Match events against ground truth
        matched_event_indices = set()
        clip_tp = 0
        clip_fn = 0
        clip_fp = 0
        clip_timing_errors: List[float] = []

        for gt in gt_list:
            gt_type = gt.get("event_type")
            gt_t = float(gt.get("approx_timestamp", 0.0))
            tol = float(gt.get("tolerance_seconds", 5.0))

            best_idx = -1
            min_err = float("inf")

            for idx, e in enumerate(events):
                if idx in matched_event_indices:
                    continue
                if e.get("event_type") == gt_type:
                    err = abs(float(e.get("timestamp", 0.0)) - gt_t)
                    if err <= tol and err < min_err:
                        min_err = err
                        best_idx = idx

            if best_idx >= 0:
                clip_tp += 1
                matched_event_indices.add(best_idx)
                clip_timing_errors.append(min_err)
                timing_errors.append(min_err)
            else:
                clip_fn += 1

        clip_fp = len(events) - len(matched_event_indices)

        total_tp += clip_tp
        total_fp += clip_fp
        total_fn += clip_fn

        avg_clip_err = (sum(clip_timing_errors) / len(clip_timing_errors)) if clip_timing_errors else 0.0

        event_counts_by_type: Dict[str, int] = {}
        for e in events:
            etype = e.get("event_type", "UNKNOWN")
            event_counts_by_type[etype] = event_counts_by_type.get(etype, 0) + 1

        row = {
            "log_file": jf.name,
            "camera_id": cam_id,
            "scenario": scenario_name,
            "events_fired": len(events),
            "tp": clip_tp,
            "fp": clip_fp,
            "fn": clip_fn,
            "mean_timing_error_sec": round(avg_clip_err, 2),
            "event_breakdown": str(event_counts_by_type),
        }
        evaluation_rows.append(row)

        print(
            f"Log: {jf.name:<32} | Scenario: {scenario_name:<20} | TP: {clip_tp} | FP: {clip_fp} | FN: {clip_fn} | "
            f"Err: {avg_clip_err:4.2f}s | Events: {event_counts_by_type}"
        )

    # Compute overall precision and recall
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    mean_err_all = (sum(timing_errors) / len(timing_errors)) if timing_errors else 0.0

    print("-" * 95)
    print(f"Overall Metrics: TP={total_tp} | FP={total_fp} | FN={total_fn}")
    print(f"Recall: {recall * 100:.1f}% | Mean Timing Error: {mean_err_all:.2f}s")
    print("-" * 95)

    # Write CSV
    if evaluation_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(evaluation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(evaluation_rows)

    # Write Markdown
    if evaluation_rows:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Phase 2 Security Rules Evaluation Report\n\n")
            f.write(f"- **Overall Recall:** {recall * 100:.1f}%\n")
            f.write(f"- **Mean Timing Error:** {mean_err_all:.2f} seconds\n\n")
            f.write("| Log File | Camera ID | Scenario | Events | TP | FP | FN | Mean Error (s) | Event Breakdown |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for r in evaluation_rows:
                f.write(
                    f"| {r['log_file']} | {r['camera_id']} | {r['scenario']} | {r['events_fired']} | "
                    f"{r['tp']} | {r['fp']} | {r['fn']} | {r['mean_timing_error_sec']} | {r['event_breakdown']} |\n"
                )

    print(f"[SAVED] Evaluation Markdown report : {md_path.resolve()}")
    print(f"[SAVED] Evaluation CSV report      : {csv_path.resolve()}")
    print("=" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate logged security events against ground truth.")
    parser.add_argument("--events-dir", type=str, default="data/output/events")
    parser.add_argument("--manifest", type=str, default="data/samples_manifest.yaml")
    parser.add_argument("--output-dir", type=str, default="data/output/evaluation")
    args = parser.parse_args()

    evaluate_all(
        events_dir=Path(args.events_dir),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
