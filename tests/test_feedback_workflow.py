from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from output.alert_manager import AlertManager
from output.clinical_dashboard import AlertFeedbackRequest, create_dashboard_app
from utils.schemas import RiskEvent


def _manager(tmp_path: Path) -> AlertManager:
    return AlertManager(
        json_log_path=str(tmp_path / "alerts.jsonl"),
        feedback_log_path=str(tmp_path / "feedback.jsonl"),
    )


def _event(track_id: int, level: str, timestamp: float) -> RiskEvent:
    event_name = "fall_detected" if level in {"HIGH", "CRITICAL"} else "stable"
    return RiskEvent(
        track_id=track_id,
        risk_level=level,
        confidence=0.9 if level in {"HIGH", "CRITICAL"} else 0.1,
        timestamp=timestamp,
        event=event_name,
        reasons=["sudden_vertical_drop"] if event_name == "fall_detected" else [],
    )


def test_feedback_annotations_are_append_only_jsonl(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    first = manager.record_feedback("d435i", 12, 100.5, "confirmed_fall")
    second = manager.record_feedback("d435i", 12, 100.5, "false_alarm")

    rows = [json.loads(line) for line in (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0] == first
    assert rows[1] == second
    assert rows[0]["label"] == "confirmed_fall"
    assert rows[1]["label"] == "false_alarm"
    assert rows[1]["annotated_at"] >= rows[0]["annotated_at"]


def test_feedback_endpoint_validates_labels_and_dashboard_has_controls(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    app = create_dashboard_app(manager)
    feedback_endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/feedback")
    dashboard_endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/dashboard")

    response = feedback_endpoint(
        AlertFeedbackRequest.model_validate({
            "stream_id": "d435i",
            "track_id": 4,
            "timestamp": 200.25,
            "label": "unclear",
        })
    )
    assert response["ok"] is True
    assert response["annotation"]["label"] == "unclear"

    normal_response = feedback_endpoint(
        AlertFeedbackRequest.model_validate({
            "stream_id": "d435i",
            "track_id": -1,
            "timestamp": 201.0,
            "label": "non_fall_activity",
        })
    )
    assert normal_response["ok"] is True

    with pytest.raises(ValidationError):
        AlertFeedbackRequest.model_validate({
                "stream_id": "d435i",
                "track_id": 4,
                "timestamp": 200.25,
                "label": "maybe",
            })

    html = dashboard_endpoint()
    assert "Confirm Fall" in html
    assert "False Alarm" in html
    assert "Mark current activity normal" in html
    assert "No active person to mark" in html
    assert "eventTs" in html


def test_open_alerts_and_summary_use_latest_state_per_track(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.emit("d435i", _event(track_id=1, level="HIGH", timestamp=10.0))
    manager.emit("d435i", _event(track_id=1, level="HIGH", timestamp=11.0))
    manager.emit("d435i", _event(track_id=2, level="CRITICAL", timestamp=12.0))
    manager.emit("d435i", _event(track_id=2, level="LOW", timestamp=13.0))

    open_alerts = manager.get_open_alerts()
    assert len(open_alerts) == 1
    assert open_alerts[0]["event"]["track_id"] == 1
    assert open_alerts[0]["event"]["timestamp"] == 11.0

    summary = manager.get_summary()
    assert summary["active_tracks"] == 2
    assert summary["high_alerts"] == 1
    assert summary["critical_alerts"] == 0
    assert summary["open_high_priority"] == 1

    manager.ack_track("d435i", 1)
    assert manager.get_open_alerts() == []
    assert manager.get_summary()["open_high_priority"] == 0
