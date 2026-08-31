from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ingestion.rtsp_stream import build_jetson_rtsp_pipeline
from models.inference_engine import SynchronizedInferenceEngine
from output.alert_manager import AlertManager
from pose.pose_estimator import MockPoseEngine, PoseEstimator
from run import load_config
from tracking.tracker import ByteTrackLikeTracker
from utils.async_jsonl import BoundedJSONLWriter
from utils.performance import ModuleLatency
from utils.schemas import Detection, RiskEvent


def test_jetson_rtsp_pipeline_uses_nvdec_and_latest_frame_sink() -> None:
    pipeline = build_jetson_rtsp_pipeline(
        "rtsp://camera.local/live?token=a&b=c",
        codec="h265",
        latency_ms=80,
        transport="tcp",
    )
    assert "rtph265depay" in pipeline
    assert "nvv4l2decoder" in pipeline
    assert "appsink drop=true max-buffers=1 sync=false" in pipeline
    assert "latency=80" in pipeline


def test_bounded_jsonl_writer_flushes_before_close(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = BoundedJSONLWriter(path, queue_size=4, batch_size=3, flush_interval_sec=0.01)
    for idx in range(7):
        writer.write(json.dumps({"idx": idx}))
    writer.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["idx"] for row in rows] == list(range(7))


def test_shared_inference_backend_warms_once() -> None:
    class _Backend:
        def __init__(self) -> None:
            self.warmups = 0
            self.calls = 0

        def warmup(self) -> None:
            self.warmups += 1

        def predict(self, inputs):
            self.calls += 1
            return inputs

    backend = _Backend()
    shared = SynchronizedInferenceEngine(backend)  # type: ignore[arg-type]
    shared.warmup()
    shared.warmup()
    assert shared.predict("frame") == "frame"
    assert backend.warmups == 1
    assert backend.calls == 1


def test_latency_tracker_reports_tail_percentiles() -> None:
    metric = ModuleLatency()
    for value in range(1, 101):
        metric.update(float(value))
    assert metric.percentile_ms(50) == 50.0
    assert metric.percentile_ms(95) == 95.0
    assert metric.percentile_ms(99) == 99.0


def test_alert_manager_drains_async_log_on_close(tmp_path: Path) -> None:
    path = tmp_path / "alerts.jsonl"
    manager = AlertManager(str(path), enable_api=False)
    manager.start()
    manager.emit(
        "s0",
        RiskEvent(
            track_id=3,
            risk_level="HIGH",
            confidence=0.9,
            timestamp=10.0,
            event="instability_risk",
            reasons=["lean_instability"],
        ),
    )
    manager.close()
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["stream_id"] == "s0"
    assert payload["event"]["risk_level"] == "HIGH"


def test_alert_manager_health_tracks_safe_stream_metrics(tmp_path: Path) -> None:
    class _Source:
        frames_captured = 20
        frames_dropped = 2
        reconnect_count = 1

    manager = AlertManager(str(tmp_path / "health.jsonl"), enable_api=False)
    manager.register_stream("ward-a")
    manager.update_stream_health("ward-a", fps=19.5, source=_Source())
    health = manager.get_health()
    manager.close()

    assert health["status"] == "ok"
    assert health["streams"]["ward-a"]["frames_captured"] == 20
    assert health["streams"]["ward-a"]["reconnect_count"] == 1


def test_runtime_privacy_toggle_is_explicit_and_resets_enabled(tmp_path: Path) -> None:
    manager = AlertManager(str(tmp_path / "privacy.jsonl"), enable_api=False, allow_privacy_toggle=True)
    manager.register_stream("ward-a", privacy_mode="person_pixelate")

    initial = manager.get_privacy_status("ward-a")
    disabled = manager.set_privacy_enabled("ward-a", False)
    restored = manager.set_privacy_enabled("ward-a", True)
    manager.close()

    assert initial["effective_mode"] == "person_pixelate"
    assert disabled["effective_mode"] == "none"
    assert restored["effective_mode"] == "person_pixelate"


def test_tracker_reports_ids_removed_after_misses() -> None:
    tracker = ByteTrackLikeTracker(iou_threshold=0.2, max_misses=0)
    estimator = PoseEstimator(MockPoseEngine())
    det = Detection(bbox=(10.0, 10.0, 60.0, 100.0), confidence=0.9)
    poses = estimator.predict(np.zeros((120, 160, 3), dtype=np.uint8), [det.bbox])
    tracks = tracker.update([det], poses, timestamp=1.0)
    assert len(tracks) == 1
    track_id = tracks[0].track_id

    tracker.update([], [], timestamp=2.0)
    assert tracker.last_removed_track_ids == [track_id]


def test_jetson_config_requires_accelerator_and_resolves_engine_path(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AH_RTSP_URL", "rtsp://camera.local/live")
    cfg = load_config(str(root / "config" / "jetson_orin_nx_rtsp.yaml"))
    assert cfg["pipeline"]["device"] == "cuda:0"
    assert cfg["pipeline"]["require_accelerator"] is True
    assert cfg["pipeline"]["allow_cpu_fallback"] is False
    assert cfg["detection"]["backend"] == "none"
    assert Path(cfg["pose"]["model_path"]).is_absolute()
    assert cfg["ingestion"]["rtsp_use_jetson_gstreamer"] is True
    assert cfg["streams"][0]["source"] == "rtsp://camera.local/live"
    assert cfg["output"]["training_log_path"] is None
