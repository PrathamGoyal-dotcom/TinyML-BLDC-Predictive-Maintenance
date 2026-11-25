"""SQLite storage layer for raw telemetry windows, 28 features, and inference logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.schemas import FEATURE_NAMES, InferenceResult, RecordingRecord


class RecordingsRepository:
    """Thread-safe SQLite repository using WAL mode for concurrent reads/writes."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self) -> None:
        """Create tables and indexes if they do not already exist."""
        feature_cols = ",\n    ".join(f"{name} REAL NOT NULL" for name in FEATURE_NAMES)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id TEXT NOT NULL UNIQUE,
            machine_id TEXT NOT NULL,
            motor_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            sample_rate_hz REAL NOT NULL,
            rpm REAL NOT NULL,
            load_pct REAL NOT NULL,
            temperature_c REAL NOT NULL,
            label TEXT,
            is_synthetic INTEGER NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL,
            vibration_g_json TEXT NOT NULL,
            current_a_json TEXT,
            {feature_cols},
            health_tier INTEGER NOT NULL,
            health_status TEXT NOT NULL,
            predicted_fault TEXT NOT NULL,
            confidence REAL NOT NULL,
            anomaly_score REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_recordings_machine_ts
            ON recordings(machine_id, id DESC);
        CREATE INDEX IF NOT EXISTS idx_recordings_label
            ON recordings(label);
        CREATE INDEX IF NOT EXISTS idx_recordings_fault
            ON recordings(predicted_fault);
        """
        with self._connect() as conn:
            conn.executescript(ddl)

    def insert_recording(
        self,
        *,
        recording_id: str,
        machine_id: str,
        motor_id: str,
        timestamp: str,
        sample_rate_hz: float,
        rpm: float,
        load_pct: float,
        temperature_c: float,
        label: Optional[str],
        is_synthetic: bool,
        vibration_g: list[float],
        current_a: Optional[list[float]],
        features: dict[str, float],
        inference: InferenceResult,
    ) -> None:
        """Persist a recording window, its 28 features, and inference results."""
        columns = [
            "recording_id",
            "machine_id",
            "motor_id",
            "timestamp",
            "sample_rate_hz",
            "rpm",
            "load_pct",
            "temperature_c",
            "label",
            "is_synthetic",
            "sample_count",
            "vibration_g_json",
            "current_a_json",
            *FEATURE_NAMES,
            "health_tier",
            "health_status",
            "predicted_fault",
            "confidence",
            "anomaly_score",
        ]
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(columns)
        values: list[Any] = [
            recording_id,
            machine_id,
            motor_id,
            timestamp,
            float(sample_rate_hz),
            float(rpm),
            float(load_pct),
            float(temperature_c),
            label,
            1 if is_synthetic else 0,
            len(vibration_g),
            json.dumps(vibration_g, separators=(",", ":")),
            json.dumps(current_a, separators=(",", ":")) if current_a is not None else None,
            *[float(features[name]) for name in FEATURE_NAMES],
            int(inference.health_tier),
            inference.health_status,
            inference.predicted_fault,
            float(inference.confidence),
            float(inference.anomaly_score),
        ]
        sql = f"INSERT INTO recordings ({col_list}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, values)
            conn.commit()

    def _row_to_record(
        self, row: sqlite3.Row, include_raw: bool = True
    ) -> RecordingRecord:
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        vib = json.loads(row["vibration_g_json"]) if include_raw else None
        curr = (
            json.loads(row["current_a_json"])
            if (include_raw and row["current_a_json"] is not None)
            else None
        )
        return RecordingRecord(
            recording_id=row["recording_id"],
            machine_id=row["machine_id"],
            motor_id=row["motor_id"],
            timestamp=row["timestamp"],
            sample_rate_hz=float(row["sample_rate_hz"]),
            rpm=float(row["rpm"]),
            load_pct=float(row["load_pct"]),
            temperature_c=float(row["temperature_c"]),
            label=row["label"],
            is_synthetic=bool(row["is_synthetic"]),
            sample_count=int(row["sample_count"]),
            features=features,
            inference=InferenceResult(
                health_tier=int(row["health_tier"]),
                health_status=row["health_status"],
                predicted_fault=row["predicted_fault"],
                confidence=float(row["confidence"]),
                anomaly_score=float(row["anomaly_score"]),
            ),
            vibration_g=vib,
            current_a=curr,
        )

    def _build_where(
        self,
        machine_id: Optional[str] = None,
        motor_id: Optional[str] = None,
        label: Optional[str] = None,
        predicted_fault: Optional[str] = None,
        health_tier: Optional[int] = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if machine_id is not None:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if motor_id is not None:
            clauses.append("motor_id = ?")
            params.append(motor_id)
        if label is not None:
            clauses.append("label = ?")
            params.append(label)
        if predicted_fault is not None:
            clauses.append("predicted_fault = ?")
            params.append(predicted_fault)
        if health_tier is not None:
            clauses.append("health_tier = ?")
            params.append(health_tier)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where_sql, params

    def list_recordings(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        machine_id: Optional[str] = None,
        motor_id: Optional[str] = None,
        label: Optional[str] = None,
        predicted_fault: Optional[str] = None,
        health_tier: Optional[int] = None,
        include_raw: bool = True,
    ) -> tuple[int, list[RecordingRecord]]:
        """Query stored recordings with optional filters and pagination."""
        where_sql, params = self._build_where(
            machine_id=machine_id,
            motor_id=motor_id,
            label=label,
            predicted_fault=predicted_fault,
            health_tier=health_tier,
        )
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM recordings {where_sql}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM recordings
                {where_sql}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return int(total), [
            self._row_to_record(r, include_raw=include_raw) for r in rows
        ]

    def iter_export_rows(
        self,
        *,
        machine_id: Optional[str] = None,
        label: Optional[str] = None,
        include_synthetic: bool = True,
    ) -> list[sqlite3.Row]:
        """Return rows ordered chronologically for CSV training export."""
        clauses: list[str] = []
        params: list[Any] = []
        if machine_id is not None:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if label is not None:
            clauses.append("label = ?")
            params.append(label)
        if not include_synthetic:
            clauses.append("is_synthetic = 0")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM recordings {where_sql} ORDER BY id ASC", params
            ).fetchall()

    def get_latest_by_machine(self) -> list[dict[str, Any]]:
        """Return the most recent recording per machine_id to hydrate dashboard state."""
        sql = """
        SELECT r.*
        FROM recordings r
        INNER JOIN (
            SELECT machine_id, MAX(id) AS max_id
            FROM recordings
            GROUP BY machine_id
        ) latest ON r.machine_id = latest.machine_id AND r.id = latest.max_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "machine_id": row["machine_id"],
                    "tier": int(row["health_tier"]),
                    "status": row["health_status"],
                    "fault": row["predicted_fault"],
                    "confidence": float(row["confidence"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "rpm": float(row["rpm"]),
                    "temperature_c": float(row["temperature_c"]),
                    "updated_at": row["timestamp"],
                }
            )
        return results
