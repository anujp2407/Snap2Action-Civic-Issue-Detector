"""
database.py
-----------
SQLite storage for Snap2Action Workstation reports.
Supports reports ledger, status tracking (Pending Review, Reported, Resolved),
locations, severity, and candidates audit JSON.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "db" / "snap2action.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    detected_image_path TEXT,
    mode TEXT NOT NULL,               -- 'road' or 'wall'
    issue_type TEXT NOT NULL,         -- 'pothole' | 'crack' | 'unknown'
    confidence REAL NOT NULL,
    bbox TEXT,                        -- "x1,y1,x2,y2" or NULL
    area INTEGER,
    roi_confidence REAL,
    severity TEXT,
    severity_score REAL,
    candidates_json TEXT,             -- all kept candidates, for analysis
    notes TEXT,
    location TEXT DEFAULT '5th Ave & Pine St (Ward 4)',
    status TEXT DEFAULT 'Pending Review',
    created_at TEXT NOT NULL
);
"""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    try:
        conn.execute(SCHEMA)
        # Migrate table if existing schema didn't have location/status
        cols = [c[1] for c in conn.execute("PRAGMA table_info(reports)").fetchall()]
        if "location" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN location TEXT DEFAULT '5th Ave & Pine St (Ward 4)'")
        if "status" not in cols:
            conn.execute("ALTER TABLE reports ADD COLUMN status TEXT DEFAULT 'Pending Review'")
        conn.commit()
    finally:
        conn.close()


def insert_report(image_path, detected_image_path, mode, detection, severity, notes="", location="5th Ave & Pine St (Ward 4)", status="Pending Review"):
    """
    Insert a detection report into SQLite.
    """
    conn = _connect()
    try:
        bbox = detection.get("bbox")
        bbox_str = ",".join(str(v) for v in bbox) if bbox else None
        candidates = detection.get("all_boxes") or detection.get("candidates") or []

        cursor = conn.execute(
            """INSERT INTO reports
               (image_path, detected_image_path, mode, issue_type, confidence,
                bbox, area, roi_confidence, severity, severity_score,
                candidates_json, notes, location, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(image_path),
                str(detected_image_path) if detected_image_path else None,
                mode,
                detection.get("issue_type", "unknown"),
                detection.get("confidence", 0.0),
                bbox_str,
                detection.get("area"),
                detection.get("roi_confidence"),
                severity.get("severity", "Unknown"),
                severity.get("score", 0.0),
                json.dumps(candidates),
                notes,
                location,
                status,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_report_status(report_id, new_status):
    conn = _connect()
    try:
        conn.execute("UPDATE reports SET status = ? WHERE id = ?", (new_status, report_id))
        conn.commit()
    finally:
        conn.close()


def update_report_notes(report_id, notes):
    conn = _connect()
    try:
        conn.execute("UPDATE reports SET notes = ? WHERE id = ?", (notes, report_id))
        conn.commit()
    finally:
        conn.close()


def fetch_history(limit=200):
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_report(report_id):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_report(report_id):
    conn = _connect()
    try:
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
    finally:
        conn.close()


def dashboard_stats():
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        by_type = conn.execute(
            "SELECT issue_type, COUNT(*) c FROM reports GROUP BY issue_type"
        ).fetchall()
        by_severity = conn.execute(
            "SELECT severity, COUNT(*) c FROM reports GROUP BY severity"
        ).fetchall()
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM reports WHERE issue_type != 'unknown'"
        ).fetchone()[0]
        critical_count = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE severity IN ('High', 'Critical')"
        ).fetchone()[0]
        return {
            "total": total,
            "by_type": {r["issue_type"]: r["c"] for r in by_type},
            "by_severity": {r["severity"]: r["c"] for r in by_severity},
            "avg_confidence": round(avg_conf, 3) if avg_conf else 0.0,
            "critical_count": critical_count,
        }
    finally:
        conn.close()
