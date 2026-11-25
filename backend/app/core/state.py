"""Live per-machine dashboard state store and async SSE pub/sub broker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
from typing import Any, Optional

from app.core.localization import TIER_TO_STATUS, get_localized_copy
from app.schemas import DashboardResponse, InferenceResult


def utc_now_iso() -> str:
    """Return compact UTC ISO-8601 timestamp (second precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_frontend_metrics(
    *,
    vib_std: float,
    curr_rms: float,
    anomaly_score: float,
    sound_db: Optional[float] = None,
) -> dict[str, float]:
    """Map raw sensor features into frontend units (mm/s, A, dB, 0-100 degradation)."""
    vibration_mms = round(min(25.0, max(0.5, 1.8 + float(vib_std) * 12.0)), 2)
    current_a = round(max(0.0, float(curr_rms)), 2)
    sound = (
        round(float(sound_db), 1)
        if sound_db is not None
        else round(58.0 + 32.0 * float(anomaly_score), 1)
    )
    degradation = round(max(0.0, min(100.0, float(anomaly_score) * 100.0)), 1)
    return {
        "vibration": vibration_mms,
        "current": current_a,
        "sound": sound,
        "degradation": degradation,
    }


class DashboardStateManager:
    """Maintains latest machine health snapshots and fans out live SSE events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, dict[str, Any]] = {}
        self._latest_machine_id: Optional[str] = None
        self._subscribers: dict[
            str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]]
        ] = {}

    def hydrate(self, records: list[dict[str, Any]]) -> None:
        """Load initial machine states from persisted SQLite snapshots."""
        with self._lock:
            for rec in records:
                self._states[rec["machine_id"]] = dict(rec)
                self._latest_machine_id = rec["machine_id"]

    def clear(self) -> None:
        """Reset in-memory state (used during test isolation)."""
        with self._lock:
            self._states.clear()
            self._latest_machine_id = None
            self._subscribers.clear()

    def build_dashboard_payload(
        self, snapshot: dict[str, Any], language: str = "en"
    ) -> DashboardResponse:
        """Format a compact, localized DashboardResponse from an internal snapshot."""
        tier = int(snapshot["tier"])
        fault = snapshot["fault"]
        lang, headline, action_hint = get_localized_copy(language, tier, fault)
        return DashboardResponse(
            machine_id=snapshot["machine_id"],
            tier=tier,
            status=TIER_TO_STATUS[tier],
            fault=fault,
            confidence=round(float(snapshot["confidence"]), 3),
            anomaly_score=round(float(snapshot["anomaly_score"]), 3),
            vibration=round(float(snapshot.get("vibration", 0.0)), 2),
            current=round(float(snapshot.get("current", 0.0)), 2),
            sound=round(float(snapshot.get("sound", 0.0)), 1),
            degradation=round(float(snapshot.get("degradation", 0.0)), 1),
            rpm=round(float(snapshot["rpm"]), 1),
            temperature_c=round(float(snapshot["temperature_c"]), 1),
            language=lang,
            headline=headline,
            action_hint=action_hint,
            updated_at=snapshot["updated_at"],
        )

    def update_state(
        self,
        *,
        machine_id: str,
        inference: InferenceResult,
        rpm: float,
        temperature_c: float,
        vib_std: float = 0.0,
        curr_rms: float = 0.0,
        sound_db: Optional[float] = None,
        updated_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update live state for a machine and push to active SSE subscribers."""
        metrics = compute_frontend_metrics(
            vib_std=vib_std,
            curr_rms=curr_rms,
            anomaly_score=float(inference.anomaly_score),
            sound_db=sound_db,
        )
        snapshot: dict[str, Any] = {
            "machine_id": machine_id,
            "tier": int(inference.health_tier),
            "status": inference.health_status,
            "fault": inference.predicted_fault,
            "confidence": float(inference.confidence),
            "anomaly_score": float(inference.anomaly_score),
            **metrics,
            "rpm": float(rpm),
            "temperature_c": float(temperature_c),
            "updated_at": updated_at or utc_now_iso(),
        }
        with self._lock:
            self._states[machine_id] = snapshot
            self._latest_machine_id = machine_id
            subs = list(self._subscribers.get(machine_id, []))

        for loop, queue in subs:
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(self._enqueue_drop_oldest, queue, dict(snapshot))
            except RuntimeError:
                pass
        return snapshot

    @staticmethod
    def _enqueue_drop_oldest(
        queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]
    ) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def get_snapshot(self, machine_id: str) -> Optional[dict[str, Any]]:
        """Retrieve latest raw state snapshot for a machine."""
        with self._lock:
            snap = self._states.get(machine_id)
            return dict(snap) if snap is not None else None

    def get_latest_snapshot(self) -> Optional[dict[str, Any]]:
        """Retrieve the most recently updated machine snapshot."""
        with self._lock:
            if self._latest_machine_id is None:
                return None
            snap = self._states.get(self._latest_machine_id)
            return dict(snap) if snap is not None else None

    def list_machine_ids(self) -> list[str]:
        """Return all known machine IDs sorted alphabetically."""
        with self._lock:
            return sorted(self._states.keys())

    def subscribe(
        self, machine_id: str, loop: asyncio.AbstractEventLoop
    ) -> asyncio.Queue[dict[str, Any]]:
        """Register an async queue for live updates on machine_id."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)
        with self._lock:
            self._subscribers.setdefault(machine_id, []).append((loop, queue))
        return queue

    def unsubscribe(
        self, machine_id: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        """Remove a subscriber queue when an SSE client disconnects."""
        with self._lock:
            subs = self._subscribers.get(machine_id, [])
            self._subscribers[machine_id] = [
                (lp, q) for lp, q in subs if q is not queue
            ]
            if not self._subscribers[machine_id]:
                self._subscribers.pop(machine_id, None)
