"""Security Event Timeline Visualization Script.

Parses a JSONL event log file and renders a visual scatter timeline of
all security events against video timestamp, color-coded by severity.

Usage:
    python scripts/event_timeline.py --event-log data/output/events/cam_walk_01_xxxx.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

SEVERITY_COLORS = {
    "Low": "#00bcd4",
    "Medium": "#ffb300",
    "High": "#ff6f00",
    "Critical": "#d32f2f",
}


def plot_event_timeline(jsonl_path: Path, output_path: Optional[Path] = None) -> None:
    if not jsonl_path.exists():
        print(f"[ERROR] Event log '{jsonl_path}' not found.")
        return

    events: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    events.append(json.loads(line_str))
                except Exception:
                    pass

    if not events:
        print("[WARNING] Zero events found in log file.")
        return

    timestamps = [e.get("timestamp", 0.0) for e in events]
    event_types = [e.get("event_type", "UNKNOWN") for e in events]
    severities = [e.get("severity", "Medium") for e in events]

    unique_types = sorted(list(set(event_types)))
    type_to_y = {t: idx for idx, t in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(12, max(4, len(unique_types) * 1.2)))

    for t_val, etype, sev in zip(timestamps, event_types, severities):
        y_val = type_to_y[etype]
        color = SEVERITY_COLORS.get(sev, "#ff9800")
        ax.scatter(t_val, y_val, color=color, s=120, edgecolors="black", zorder=3)
        ax.annotate(
            f"{sev}",
            (t_val, y_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    ax.set_yticks(list(type_to_y.values()))
    ax.set_yticklabels(list(type_to_y.keys()), fontsize=11, fontweight="bold")
    ax.set_xlabel("Video Timeline (Seconds)", fontsize=11)
    ax.set_title(f"Security Event Incident Timeline: {jsonl_path.name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()

    out_file = output_path or jsonl_path.with_suffix(".png")
    plt.savefig(out_file, dpi=140)
    plt.close()
    print(f"[SAVED] Event Timeline plot saved to: {out_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render security event timeline.")
    parser.add_argument("--event-log", type=str, required=True, help="Path to .jsonl event file.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
    args = parser.parse_args()

    plot_event_timeline(Path(args.event_log), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
