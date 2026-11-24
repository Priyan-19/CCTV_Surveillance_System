"""Incident Report Tool — Phase 4 CCTV Surveillance System.

Queries the SQLite database for a specific incident and prints a structured
report including: incident details, correlated events, scoring audit trace,
evidence files (with existence check), and human-operator review prompt.

Usage:
    python scripts/incident_report.py --incident-no INC-000006
    python scripts/incident_report.py --list-recent 10
    python scripts/incident_report.py --incident-no INC-000006 --db-path data/cctv.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.repository import StorageRepository

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("incident_report")

# ANSI colour codes (Windows 10+ terminal supports these)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_ORANGE = "\033[33m"
_YELLOW = "\033[93m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_GRAY   = "\033[90m"

SEVERITY_COLOR = {
    "low":      _GRAY,
    "medium":   _YELLOW,
    "high":     _ORANGE,
    "critical": _RED,
}


def _conn(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")
    return con


def list_recent(db_path: str, n: int = 10) -> None:
    with _conn(db_path) as con:
        rows = con.execute(
            """
            SELECT incident_no, threat_level, threat_score, place, status,
                   start_time, closed_at,
                   (SELECT COUNT(*) FROM events WHERE incident_id = incidents.id) AS event_count,
                   (SELECT COUNT(*) FROM evidence WHERE incident_id = incidents.id) AS evidence_count
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

    if not rows:
        print("No incidents found in database.")
        return

    print(f"\n{_BOLD}{'Incident':12} {'Level':8} {'Score':5} {'Status':8} {'Events':6} {'Evid':4}  Place{_RESET}")
    print("-" * 90)
    for r in rows:
        col = SEVERITY_COLOR.get(str(r["threat_level"]).lower(), "")
        print(
            f"{col}{r['incident_no']:12} {str(r['threat_level']).upper():8} "
            f"{r['threat_score']:5}  {str(r['status']):8} {r['event_count']:6} {r['evidence_count']:4}  "
            f"{r['place'] or '—'}{_RESET}"
        )


def show_incident(db_path: str, incident_no: str) -> None:
    repo = StorageRepository(db_path=db_path)
    incident = repo.get_incident(incident_no)
    if incident is None:
        print(f"[ERROR] Incident {incident_no!r} not found in {db_path}.")
        sys.exit(1)

    # --- Header ----------------------------------------------------------------
    sev = str(incident.threat_level).lower()
    col = SEVERITY_COLOR.get(sev, "")
    score = incident.threat_score

    print("\n" + "=" * 80)
    print(f"{col}{_BOLD}INCIDENT REPORT — {incident.incident_no}{_RESET}")
    print("=" * 80)
    print(f"{'Threat Level':22}: {col}{_BOLD}{str(incident.threat_level).upper()}{_RESET}  "
          f"(Score {score}/100)")
    print(f"{'Camera / Place':22}: {incident.camera_id}  ({incident.place or 'Unknown Location'})")
    print(f"{'Status':22}: {incident.status}")
    print(f"{'Start Time':22}: {incident.start_time}")
    print(f"{'Last Event':22}: {incident.last_event_time}")
    print(f"{'Closed':22}: {incident.closed_at or '— (open)'}")

    # Activities
    print(f"{'Activities':22}: {', '.join(incident.activities) or '—'}")

    # Subject
    subject = incident.subject or {}
    if subject:
        tid = subject.get("track_id", subject.get("primary_track_id", "—"))
        istat = subject.get("identity_status", subject.get("identity", {}).get("status", "—"))
        pname = subject.get("person_name", "—")
        print(f"{'Subject Track ID':22}: {tid}")
        print(f"{'Identity Status':22}: {istat}  ({pname})")

    # --- Scoring Audit Trace ---------------------------------------------------
    print(f"\n{_BOLD}{_CYAN}── Threat Score Audit Trace ──{_RESET}")
    for line in incident.explanation:
        print(f"  {_GRAY}{line}{_RESET}")

    # --- Correlated Events -----------------------------------------------------
    print(f"\n{_BOLD}{_CYAN}── Correlated Events ({len(incident.events)}) ──{_RESET}")
    if not incident.events:
        print("  No events found.")
    for ev in incident.events:
        loc = ev.location if isinstance(ev.location, dict) else {}
        zone = loc.get("zone_name", loc.get("zone_id", "—"))
        det = ev.details if isinstance(ev.details, dict) else {}
        istat = det.get("identity_status", "—")
        print(
            f"  [{ev.timestamp}]  {str(ev.event_type.value if hasattr(ev.event_type, 'value') else ev.event_type):28} "
            f"sev={str(ev.severity.value if hasattr(ev.severity, 'value') else ev.severity):6}  conf={ev.confidence:.2f}  "
            f"track={ev.track_id}  zone={zone}  id={istat}"
        )

    # --- Evidence Items --------------------------------------------------------
    print(f"\n{_BOLD}{_CYAN}── Evidence Package ({len(incident.evidence)} item(s)) ──{_RESET}")
    if not incident.evidence:
        print("  No evidence collected for this incident.")
    missing_count = 0
    for ev in incident.evidence:
        fp = ev.file_path or ""
        fp_path = Path(fp.replace("/", "\\")) if fp else None
        if fp_path and fp_path.exists():
            exists_tag = f"{_GREEN}[FILE OK]{_RESET}"
            size_kb = fp_path.stat().st_size // 1024
            exists_tag += f" ({size_kb} KB)"
        elif fp_path:
            exists_tag = f"{_RED}[FILE MISSING]{_RESET}"
            missing_count += 1
        else:
            exists_tag = f"{_GRAY}[no path]{_RESET}"
            missing_count += 1
        print(f"  [{ev.type:20}]  {fp or '—'}  {exists_tag}")

    if missing_count > 0:
        print(f"\n  {_ORANGE}WARNING: {missing_count} evidence file(s) are missing from disk.{_RESET}")

    # --- Verdict Prompt --------------------------------------------------------
    print(f"\n{_BOLD}{'─' * 78}{_RESET}")
    print(f"{_BOLD}Human Operator Review:{_RESET}")
    print("  Was the incident correctly identified? Options:")
    print("  [C] Correct — grouping, score, and activities are accurate.")
    print("  [W] Wrong grouping — events belong to separate incidents.")
    print("  [S] Wrong score — activities are correct but score level is off.")
    print("  [E] Bad evidence — wrong person/object crop, or clip is unplayable.")
    print(f"{'─' * 78}")
    print("  Record your verdict in docs/PHASE4_NOTES.md Section 7 (Real-Footage Review).")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Incident Report Tool.")
    parser.add_argument("--incident-no", type=str, default=None,
                        help="Incident number to report (e.g. INC-000006).")
    parser.add_argument("--list-recent", type=int, default=None, metavar="N",
                        help="List the N most recent incidents and exit.")
    parser.add_argument("--db-path", type=str, default="data/cctv.db",
                        help="Path to SQLite database.")
    args = parser.parse_args()

    db = str(PROJECT_ROOT / args.db_path) if not Path(args.db_path).is_absolute() else args.db_path
    if not Path(db).exists():
        print(f"[ERROR] Database not found: {db}")
        sys.exit(1)

    if args.list_recent is not None:
        list_recent(db, args.list_recent)
    elif args.incident_no:
        show_incident(db, args.incident_no)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
