"""
AQUAVISION: AI-Based Oil Spill Detection
SQLite Incident Storage & Persistence Engine (Phase 15)

Stores real inference detections with architectural hooks for future
geospatial and AIS correlation layers without fabricating fake coordinates.
"""

import os
import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

def _resolve_db_path() -> Path:
    # Use /tmp on Vercel or read-only filesystems
    if os.environ.get("VERCEL") or not os.access("backend", os.W_OK):
        return Path(tempfile.gettempdir()) / "aquavision.db"
    return Path("backend/aquavision.db")

DB_PATH = _resolve_db_path()


def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row factories."""
    global DB_PATH
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
    except (sqlite3.OperationalError, OSError, PermissionError):
        # Fallback to tempfile if parent directory is read-only
        DB_PATH = Path(tempfile.gettempdir()) / "aquavision.db"
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if not present."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_uuid TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                oil_probability REAL NOT NULL,
                no_oil_probability REAL NOT NULL,
                explanation TEXT,
                original_image_url TEXT NOT NULL,
                gradcam_image_url TEXT,
                overlay_image_url TEXT,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                vessel_attribution_status TEXT DEFAULT 'Pending Phase 2 AIS Correlation'
            );
        """)
        conn.commit()


def save_incident(incident_data: Dict[str, Any]) -> Dict[str, Any]:
    """Persists a new incident record and returns the created row."""
    inc_uuid = incident_data.get("incident_uuid", str(uuid.uuid4()))
    now_iso = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incidents (
                incident_uuid, filename, prediction, class_id, confidence,
                oil_probability, no_oil_probability, explanation,
                original_image_url, gradcam_image_url, overlay_image_url,
                model_version, created_at, latitude, longitude, vessel_attribution_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inc_uuid,
            incident_data["filename"],
            incident_data["prediction"],
            incident_data["class_id"],
            float(incident_data["confidence"]),
            float(incident_data["probabilities"]["oil_spill"]),
            float(incident_data["probabilities"]["no_oil"]),
            incident_data.get("explanation", ""),
            incident_data["original_image_url"],
            incident_data.get("gradcam_image_url"),
            incident_data.get("overlay_image_url"),
            incident_data.get("model_version", "efficientnet_b0-v1.0"),
            now_iso,
            incident_data.get("latitude"), # None if unavailable
            incident_data.get("longitude"), # None if unavailable
            incident_data.get("vessel_attribution_status", "Awaiting future AIS module")
        ))
        conn.commit()
        new_id = cursor.lastrowid

    return get_incident_by_id(new_id)


def get_all_incidents(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves previous incidents sorted by recency."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_incident_by_id(incident_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves an incident by its numeric ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


# Auto-initialize database on import
init_db()
