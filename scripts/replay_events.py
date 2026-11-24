"""Threat Engine Scenario Replay & Verification Tool.

Replays recorded or synthetic security event sequences through the Threat Engine,
verifying event correlation, score computations, audit explanations, and alert dispatching.

Conforms to docs/05_THREAT_ENGINE_SPEC.md and docs/10_TEST_PLAN.md.

Usage:
    python scripts/replay_events.py scenarios/server_room_intrusion.yaml
    python scripts/replay_events.py --all
    python scripts/replay_events.py --all --db-path data/cctv.db
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.types import Event, EventType, Severity
from src.storage.repository import StorageRepository
from src.threat.incident_manager import IncidentManager


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_scenario(
    scenario_path: Path,
    threat_config: Dict[str, Any],
    alerts_config: Dict[str, Any],
    repository: Optional[StorageRepository] = None,
) -> Dict[str, Any]:
    """Execute a single scenario file through the IncidentManager."""
    data = load_yaml(scenario_path)
    name = data.get("name", scenario_path.stem)
    desc = data.get("description", "")
    default_cam = data.get("camera_id", "CAM-01")
    default_place = data.get("place", "Main Area")
    events_raw = data.get("events", [])

    manager = IncidentManager(
        threat_config=threat_config,
        alerts_config=alerts_config,
        repository=repository,
    )

    dispatched_alerts = []
    last_timestamp = 0.0

    print("=" * 80)
    print(f" SCENARIO: {name.upper()}")
    print(f" Source   : {scenario_path.relative_to(PROJECT_ROOT)}")
    if desc:
        print(f" Goal     : {desc}")
    print("-" * 80)

    for idx, raw_evt in enumerate(events_raw, start=1):
        # Fill defaults
        raw_evt.setdefault("camera_id", default_cam)
        loc = raw_evt.setdefault("location", {})
        if not loc.get("zone_id"):
            loc["zone_id"] = "zone_main"
        if not loc.get("zone_name"):
            loc["zone_name"] = default_place

        ts = float(raw_evt.get("timestamp", 0.0))
        last_timestamp = max(last_timestamp, ts)

        inc, alert = manager.process_event(raw_evt)

        evt_type = raw_evt.get("event_type")
        track_id = raw_evt.get("track_id")
        score_str = f"Score: {inc.threat_score:3d} [{inc.threat_level:<8}]" if inc else "Buffered Context"
        inc_str = inc.incident_no if inc else "        "

        alert_mark = f"-> [DISPATCHED {alert.threat_level.upper()}]" if alert else ""
        print(f"  [{idx:2d}] t={ts:5.1f}s | {evt_type:<22} (trk #{str(track_id):<2}) | {inc_str} | {score_str} {alert_mark}")

        if alert:
            dispatched_alerts.append(alert)

    # Flush quiet period at the end
    manager.flush_quiet_period(last_timestamp + 350.0)

    all_incidents = list(manager.open_incidents.values()) + list(manager.closed_incidents.values())
    # Sort by incident_no
    all_incidents.sort(key=lambda x: x.incident_no)

    print("-" * 80)
    print(f" INCIDENTS CREATED ({len(all_incidents)}):")
    for inc in all_incidents:
        print(f"\n  * {inc.incident_no} | {inc.threat_level.upper():<8} (Score: {inc.threat_score}/100) | Place: {inc.place}")
        print(f"    Activities     : {', '.join(inc.activities)}")
        subj = inc.subject or {}
        subj_str = f"Track #{subj.get('track_id')}"
        if subj.get("person_name"):
            subj_str += f" ({subj.get('person_name')})"
        if subj.get("identity_status"):
            subj_str += f" | Status: {subj.get('identity_status')}"
        if subj.get("reason"):
            subj_str += f" ({subj.get('reason')})"
        print(f"    Subject        : {subj_str}")
        print(f"    Timeline       : {inc.start_time:.1f}s -> {inc.last_event_time:.1f}s (Closed at: {inc.closed_at}s)")
        print(f"    Audit Trace    :")
        for line in inc.explanation:
            print(f"      - {line}")

    # Metrics Summary
    raw_event_count = len([e for e in events_raw if e.get("event_type") != "UNKNOWN_PERSON"])
    alert_count = len(dispatched_alerts)
    reduction_pct = ((raw_event_count - alert_count) / raw_event_count * 100.0) if raw_event_count > 0 else 0.0

    print("-" * 80)
    print(f" SCENARIO SUMMARY & ALERT REDUCTION:")
    print(f"  - Total Ingested Events : {len(events_raw)}")
    print(f"  - Security Rule Events  : {raw_event_count}")
    print(f"  - Correlated Incidents  : {len(all_incidents)}")
    print(f"  - Dispatched Alerts     : {alert_count}")
    print(f"  - Alert Reduction Rate  : {reduction_pct:5.1f}% (vs 1-alert-per-event baseline)")
    print("=" * 80 + "\n")

    return {
        "scenario": name,
        "raw_events": len(events_raw),
        "incidents": len(all_incidents),
        "alerts": alert_count,
        "reduction_pct": reduction_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay security scenarios through the Threat Engine.")
    parser.add_argument("scenarios", nargs="*", help="Path to scenario YAML file(s).")
    parser.add_argument("--all", action="store_true", help="Replay all scenarios in scenarios/ directory.")
    parser.add_argument("--db-path", type=str, default=None, help="Optional SQLite database path for persistence.")
    parser.add_argument("--threat-config", type=str, default="configs/threat.yaml", help="Threat engine configuration.")
    parser.add_argument("--alerts-config", type=str, default="configs/alerts.yaml", help="Alerts configuration.")
    args = parser.parse_args()

    threat_cfg = load_yaml(Path(args.threat_config))
    alerts_cfg = load_yaml(Path(args.alerts_config))

    # Optional Repository
    repo = None
    if args.db_path:
        repo = StorageRepository(db_path=args.db_path)
        print(f"[INFO] Storage Repository active: {Path(args.db_path).resolve()}")

    # Collect scenario files
    scenario_files: List[Path] = []
    if args.all or not args.scenarios:
        scenarios_dir = PROJECT_ROOT / "scenarios"
        if scenarios_dir.exists():
            scenario_files = sorted(list(scenarios_dir.glob("*.yaml")))
    else:
        for p in args.scenarios:
            path = Path(p)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                scenario_files.append(path)

    if not scenario_files:
        print("[ERROR] No scenario files found to replay.")
        sys.exit(1)

    print("\n" + "#" * 80)
    print("       INTELLIGENT CCTV SURVEILLANCE: THREAT ENGINE SCENARIO REPLAY")
    print("#" * 80)
    print(f"Total Scenarios Found : {len(scenario_files)}")
    print(f"Correlation Window    : {threat_cfg.get('correlation', {}).get('time_window_seconds', 120.0)}s")
    print(f"Quiet Period Window   : {threat_cfg.get('correlation', {}).get('quiet_period_seconds', 300.0)}s")
    print("#" * 80 + "\n")

    results = []
    for sc_path in scenario_files:
        res = run_scenario(sc_path, threat_cfg, alerts_cfg, repository=repo)
        results.append(res)

    # Master Table
    print("\n" + "=" * 80)
    print("                   ALL SCENARIOS AGGREGATED BENCHMARK")
    print("=" * 80)
    print(f"{'Scenario Name':<38} | {'Events':<6} | {'Incs':<4} | {'Alerts':<6} | {'Reduction %':<11}")
    print("-" * 80)
    tot_events = sum(r["raw_events"] for r in results)
    tot_incs = sum(r["incidents"] for r in results)
    tot_alerts = sum(r["alerts"] for r in results)
    for r in results:
        print(f"{r['scenario']:<38} | {r['raw_events']:<6} | {r['incidents']:<4} | {r['alerts']:<6} | {r['reduction_pct']:>9.1f}%")
    print("-" * 80)
    agg_red = ((tot_events - tot_alerts) / tot_events * 100.0) if tot_events > 0 else 0.0
    print(f"{'TOTAL / AGGREGATE':<38} | {tot_events:<6} | {tot_incs:<4} | {tot_alerts:<6} | {agg_red:>9.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
