from __future__ import annotations

import signal
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from detection.yolo_detector import MockDetectionEngine, UltralyticsYOLOEngine, YOLOPersonDetector
from features.feature_extractor import FeatureExtractor
from ingestion.realsense_source import create_realsense_source
from ingestion.rtsp_stream import create_rtsp_source
from ingestion.video_loader import create_video_source
from models.inference_engine import InferenceEngine, SynchronizedInferenceEngine
from output.alert_manager import AlertManager
from output.training_data_logger import TrainingDataLogger
from output.visualizer import VisualizationConfig, Visualizer
from pose.pose_estimator import MockPoseEngine, MoveNetTorchEngine, PoseEstimator, RTMOMMPoseEngine, UltralyticsPoseEngine
from risk.risk_scoring import RiskScorer
from temporal.rule_engine import RuleEngine
from temporal.temporal_model import (
    HeuristicTemporalEngine,
    NullTemporalEngine,
    TemporalRiskModel,
    TorchGRUInferenceEngine,
    TorchSTGCNInferenceEngine,
    TorchTransformerLiteInferenceEngine,
)
from tracking.tracker import ByteTrackLikeTracker, PoseAwareKalmanTracker
from utils.logger import setup_logger
from utils.performance import FPSMonitor, PerformanceTracker
from utils.schemas import Detection


class StreamConfig(BaseModel):
    stream_id: str
    type: str = Field(pattern="^(webcam|video|rtsp|realsense|gstreamer|dummy)$")
    source: str | int


class PipelineConfig(BaseModel):
    device: str = "auto"
    fps: float = 15.0
    buffer_size: int = 4
    max_frames: int | None = None
    metrics_interval_sec: float = 3.0
    sequence_len: int = 16
    require_accelerator: bool = False
    allow_cpu_fallback: bool = False
    warmup_models: bool = True
    reject_duplicate_inference: bool = False
    share_inference_backends: bool = False


class SharedInferenceBackends:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: dict[tuple[Any, ...], SynchronizedInferenceEngine] = {}

    def get_or_create(self, key: tuple[Any, ...], factory) -> InferenceEngine:
        with self._lock:
            backend = self._backends.get(key)
            if backend is None:
                backend = SynchronizedInferenceEngine(factory())
                self._backends[key] = backend
            return backend


class RiskDetectionPipeline:
    def __init__(
        self,
        stream: StreamConfig,
        cfg: dict[str, Any],
        alert_manager: AlertManager | None = None,
        inference_backends: SharedInferenceBackends | None = None,
    ) -> None:
        self.stream = stream
        self.cfg = cfg
        self.logger = setup_logger(name=f"pipeline.{stream.stream_id}", level=cfg["logging"]["level"])
        self.pipeline_cfg = PipelineConfig(**cfg["pipeline"])
        self._inference_backends = inference_backends
        self.device = self._resolve_device(self.pipeline_cfg.device)
        self._validate_runtime_configuration()

        self.source = self._build_source()
        self.detector = self._build_detector()
        self.pose_estimator = self._build_pose_estimator()
        self.tracker = self._build_tracker()

        bed_zones = [tuple(zone) for zone in cfg["features"].get("bed_zones", [])]
        self._bed_zones = bed_zones
        self.feature_extractor = FeatureExtractor(
            bed_zones=bed_zones,
            min_kpt_conf=cfg["features"]["min_keypoint_conf"],
        )

        self.rule_engine = RuleEngine(**cfg["rules"])
        self.temporal_model = self._build_temporal_model()
        self.risk_scorer = RiskScorer(**cfg["risk"])

        alert_cfg = cfg["output"]
        live_stream_cfg = alert_cfg.get("live_stream", {})
        suppress_cfg = alert_cfg.get("alert_suppression", {})
        writer_cfg = alert_cfg.get("async_writer", {})
        self._live_stream_enabled = bool(live_stream_cfg.get("enabled", False))
        publish_fps = max(0.1, float(live_stream_cfg.get("fps", 8.0)))
        self._live_stream_interval = 1.0 / publish_fps
        self._last_live_stream_ts = 0.0
        if alert_manager is None:
            self.alert_manager = AlertManager(
                json_log_path=alert_cfg["json_log_path"],
                enable_api=alert_cfg["enable_rest_api"],
                api_host=alert_cfg["rest_api_host"],
                api_port=alert_cfg["rest_api_port"],
                frame_jpeg_quality=int(live_stream_cfg.get("jpeg_quality", 80)),
                dedupe_window_sec=float(suppress_cfg.get("dedupe_window_sec", 1.0)),
                emit_on_level_change_only=bool(suppress_cfg.get("emit_on_level_change_only", True)),
                logger_name=f"alerts.{stream.stream_id}",
                log_queue_size=int(writer_cfg.get("alert_queue_size", 2048)),
                log_batch_size=int(writer_cfg.get("alert_batch_size", 64)),
                frame_queue_size=int(writer_cfg.get("frame_queue_size", 2)),
            )
            self._owns_alert_manager = True
        else:
            self.alert_manager = alert_manager
            self._owns_alert_manager = False
        self.alert_manager.register_stream(stream.stream_id)
        vis_cfg = VisualizationConfig(**alert_cfg.get("visualization", {}))
        self.visualizer = Visualizer(stream_id=stream.stream_id, cfg=vis_cfg)
        self.training_logger = TrainingDataLogger(
            alert_cfg.get("training_log_path"),
            queue_size=int(writer_cfg.get("training_queue_size", 4096)),
            batch_size=int(writer_cfg.get("training_batch_size", 128)),
        )

        self._stop_event = threading.Event()
        self._perf = PerformanceTracker()
        self._fps = FPSMonitor()
        self._seq: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.pipeline_cfg.sequence_len))

    def _shared_backend(self, key: tuple[Any, ...], factory) -> InferenceEngine:
        if self._inference_backends is None:
            return factory()
        return self._inference_backends.get_or_create(key, factory)

    def _validate_runtime_configuration(self) -> None:
        det_backend = str(self.cfg["detection"].get("backend", "none"))
        pose_backend = str(self.cfg["pose"].get("backend", "mock"))
        temporal_backend = str(self.cfg["temporal_model"].get("backend", "none"))
        if self.pipeline_cfg.reject_duplicate_inference and det_backend == "ultralytics" and pose_backend == "ultralytics_pose":
            raise RuntimeError(
                "This profile rejects duplicate full-frame inference: use detection.backend=none with one-stage Ultralytics pose"
            )

        required_paths: list[tuple[str, str]] = []
        if det_backend == "ultralytics":
            required_paths.append(("detection", self.cfg["detection"]["model_path"]))
        if pose_backend in {"ultralytics_pose", "movenet_torch"}:
            required_paths.append(("pose", self.cfg["pose"]["model_path"]))
        if temporal_backend in {"torch_gru", "torch_transformer_lite", "torch_transformer", "torch_stgcn", "stgcn"}:
            required_paths.append(("temporal_model", self.cfg["temporal_model"]["model_path"]))

        for section, raw_path in required_paths:
            model_path = Path(raw_path)
            if not model_path.exists():
                raise FileNotFoundError(f"{section} model not found: {model_path}")
            if model_path.suffix.lower() == ".engine" and not self.device.startswith("cuda"):
                raise RuntimeError(f"TensorRT engine requires CUDA, resolved device={self.device}: {model_path}")

    def _build_source(self):
        fps = self.pipeline_cfg.fps
        buffer_size = self.pipeline_cfg.buffer_size
        if self.stream.type == "rtsp":
            ingestion_cfg = self.cfg["ingestion"]
            return create_rtsp_source(
                rtsp_url=str(self.stream.source),
                fps=fps,
                buffer_size=buffer_size,
                reconnect_interval_sec=ingestion_cfg["rtsp_reconnect_sec"],
                max_retries=ingestion_cfg["rtsp_max_retries"],
                use_jetson_gstreamer=bool(ingestion_cfg.get("rtsp_use_jetson_gstreamer", False)),
                codec=str(ingestion_cfg.get("rtsp_codec", "h264")),
                latency_ms=int(ingestion_cfg.get("rtsp_latency_ms", 120)),
                transport=str(ingestion_cfg.get("rtsp_transport", "tcp")),
                runtime_reconnect=bool(ingestion_cfg.get("rtsp_runtime_reconnect", True)),
                runtime_max_retries=ingestion_cfg.get("rtsp_runtime_max_retries"),
            )
        if self.stream.type == "realsense":
            return create_realsense_source(
                source=self.stream.source,
                options=self.cfg["ingestion"].get("realsense", {}),
                buffer_size=buffer_size,
            )
        webcam_options = {
            "requested_fps": self.cfg["ingestion"].get("webcam_requested_fps"),
            "width": self.cfg["ingestion"].get("webcam_width"),
            "height": self.cfg["ingestion"].get("webcam_height"),
            "backend": self.cfg["ingestion"].get("webcam_backend", "auto"),
        }
        return create_video_source(
            self.stream.type,
            self.stream.source,
            fps=fps,
            buffer_size=buffer_size,
            webcam_options=webcam_options,
        )

    def _resolve_device(self, requested: str) -> str:
        normalized = str(requested).lower()
        if normalized == "cpu":
            if self.pipeline_cfg.require_accelerator:
                raise RuntimeError("pipeline.require_accelerator=true is incompatible with pipeline.device=cpu")
            return "cpu"
        try:
            import torch
        except ImportError as exc:
            if normalized.startswith("cuda") or self.pipeline_cfg.require_accelerator:
                raise RuntimeError("CUDA was requested but PyTorch is not installed") from exc
            return "cpu"
        if torch.cuda.is_available():
            return requested if normalized.startswith("cuda") else "cuda:0"
        if normalized.startswith("cuda") or self.pipeline_cfg.require_accelerator:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cpu"

    def _build_detector(self) -> YOLOPersonDetector | None:
        det_cfg = self.cfg["detection"]
        backend_type = det_cfg["backend"]
        if backend_type == "none":
            return None
        if backend_type == "ultralytics":
            key = (
                "ultralytics_detection",
                det_cfg["model_path"],
                self.device,
                str(det_cfg.get("input_size", 640)),
                float(det_cfg["conf_threshold"]),
                int(det_cfg["person_class_id"]),
            )
            backend = self._shared_backend(
                key,
                lambda: UltralyticsYOLOEngine(
                    model_path=det_cfg["model_path"],
                    device=self.device,
                    input_size=det_cfg.get("input_size", 640),
                    conf_threshold=det_cfg["conf_threshold"],
                    person_class_id=det_cfg["person_class_id"],
                    allow_cpu_fallback=self.pipeline_cfg.allow_cpu_fallback,
                ),
            )
        else:
            backend = MockDetectionEngine()
        return YOLOPersonDetector(
            backend=backend,
            conf_threshold=det_cfg["conf_threshold"],
            person_class_id=det_cfg["person_class_id"],
        )

    def _build_pose_estimator(self) -> PoseEstimator:
        pose_cfg = self.cfg["pose"]
        backend_type = pose_cfg["backend"]
        if backend_type == "movenet_torch":
            backend = MoveNetTorchEngine(
                model_path=pose_cfg["model_path"],
                device=self.device,
                input_size=pose_cfg["input_size"],
            )
        elif backend_type == "ultralytics_pose":
            pose_conf = pose_cfg.get("conf_threshold", self.cfg["detection"].get("conf_threshold", 0.25))
            key = (
                "ultralytics_pose",
                pose_cfg["model_path"],
                self.device,
                str(pose_cfg.get("input_size", 256)),
                float(pose_conf),
                float(pose_cfg.get("match_iou_threshold", 0.1)),
            )
            backend = self._shared_backend(
                key,
                lambda: UltralyticsPoseEngine(
                    model_path=pose_cfg["model_path"],
                    device=self.device,
                    min_match_iou=pose_cfg.get("match_iou_threshold", 0.1),
                    input_size=pose_cfg.get("input_size", 256),
                    conf_threshold=pose_conf,
                    allow_cpu_fallback=self.pipeline_cfg.allow_cpu_fallback,
                ),
            )
        elif backend_type in {"rtmo", "rtmo_mmpose"}:
            backend = RTMOMMPoseEngine(
                model_alias=pose_cfg.get("model_alias", "rtmo-m"),
                device=self.device,
                bbox_thr=pose_cfg.get("bbox_thr", 0.2),
            )
        else:
            backend = MockPoseEngine()
        return PoseEstimator(backend=backend)

    def _build_tracker(self):
        tracker_cfg = self.cfg["tracking"]
        backend_type = str(tracker_cfg.get("backend", "legacy_iou")).lower()
        if backend_type in {"pose_kalman", "pose_aware", "pose_hungarian"}:
            return PoseAwareKalmanTracker(
                iou_threshold=float(tracker_cfg.get("iou_threshold", 0.15)),
                max_misses=int(tracker_cfg.get("max_misses", 20)),
                track_high_thresh=float(tracker_cfg.get("track_high_thresh", 0.35)),
                track_low_thresh=float(tracker_cfg.get("track_low_thresh", 0.10)),
                new_track_thresh=float(tracker_cfg.get("new_track_thresh", 0.40)),
                pose_similarity_threshold=float(tracker_cfg.get("pose_similarity_threshold", 0.20)),
                max_center_distance=float(tracker_cfg.get("max_center_distance", 1.25)),
                iou_weight=float(tracker_cfg.get("iou_weight", 0.50)),
                pose_weight=float(tracker_cfg.get("pose_weight", 0.40)),
                motion_weight=float(tracker_cfg.get("motion_weight", 0.10)),
                keypoint_ema_alpha=float(tracker_cfg.get("keypoint_ema_alpha", 0.70)),
            )
        if backend_type not in {"legacy_iou", "bytetrack_like", "iou"}:
            raise ValueError(f"Unsupported tracking backend: {backend_type}")
        return ByteTrackLikeTracker(
            iou_threshold=float(tracker_cfg["iou_threshold"]),
            max_misses=int(tracker_cfg["max_misses"]),
        )

    def _build_temporal_model(self) -> TemporalRiskModel:
        tm_cfg = self.cfg["temporal_model"]
        backend_type = tm_cfg["backend"]
        if backend_type == "torch_gru":
            backend = TorchGRUInferenceEngine(
                model_path=tm_cfg["model_path"],
                device=self.device,
            )
        elif backend_type in {"torch_transformer_lite", "torch_transformer"}:
            backend = TorchTransformerLiteInferenceEngine(
                model_path=tm_cfg["model_path"],
                device=self.device,
            )
        elif backend_type in {"torch_stgcn", "stgcn"}:
            backend = TorchSTGCNInferenceEngine(
                model_path=tm_cfg["model_path"],
                device=self.device,
            )
        elif backend_type == "none":
            backend = NullTemporalEngine()
        else:
            backend = HeuristicTemporalEngine()
        return TemporalRiskModel(
            backend=backend,
            sequence_len=self.pipeline_cfg.sequence_len,
            infer_interval=int(tm_cfg.get("infer_interval", 1)),
            min_infer_steps=int(tm_cfg.get("min_infer_steps", 2)),
            min_pose_quality=float(tm_cfg.get("min_pose_quality", 0.20)),
        )

    def _warmup_models(self) -> None:
        if not self.pipeline_cfg.warmup_models:
            return
        warmed: set[int] = set()
        for backend in [getattr(self.detector, "backend", None), getattr(self.pose_estimator, "backend", None)]:
            if backend is None or id(backend) in warmed:
                continue
            warmup = getattr(backend, "warmup", None)
            if callable(warmup):
                warmup()
                warmed.add(id(backend))

    def _cleanup_track_state(self, track_ids: list[int]) -> None:
        for track_id in track_ids:
            self._seq.pop(track_id, None)
            self.feature_extractor.remove_track(track_id)
            self.rule_engine.remove_track(track_id)
            self.temporal_model.remove_track(track_id)
            self.risk_scorer.remove_track(track_id)
            self.alert_manager.retire_track(self.stream.stream_id, track_id)

    def _detections_from_poses(self, poses) -> list[Detection]:
        det_cfg = self.cfg["detection"]
        threshold = float(det_cfg.get("conf_threshold", 0.0))
        detections: list[Detection] = []
        for pose in poses:
            score = float(pose.confidence)
            if score < threshold:
                continue
            detections.append(
                Detection(
                    bbox=pose.bbox,
                    confidence=score,
                    class_id=det_cfg.get("person_class_id", 0),
                    class_name="person",
                )
            )
        return detections

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._warmup_models()
        self.source.start()
        if self._owns_alert_manager:
            self.alert_manager.start()
        self.logger.info("stream=%s started device=%s", self.stream.stream_id, self.device)

        frame_count = 0
        last_metric_ts = time.time()

        try:
            while not self._stop_event.is_set():
                packet = self.source.read(timeout=0.25)
                if packet is None:
                    running = bool(getattr(self.source, "is_running", True))
                    if not running:
                        break
                    continue
                if packet.frame is None:
                    continue

                frame = packet.frame
                ts = packet.timestamp

                if self.detector is None:
                    with self._perf.track("pose"):
                        poses = self.pose_estimator.predict_full_frame(frame)
                    with self._perf.track("detection"):
                        detections = self._detections_from_poses(poses)
                else:
                    with self._perf.track("detection"):
                        detections = self.detector.detect(frame)
                    bboxes = [d.bbox for d in detections]
                    with self._perf.track("pose"):
                        poses = self.pose_estimator.predict(frame, bboxes)

                with self._perf.track("tracking"):
                    tracks = self.tracker.update(detections, poses, timestamp=ts)
                self._cleanup_track_state(self.tracker.last_removed_track_ids)
                risk_events: dict[int, Any] = {}

                for track in tracks:
                    with self._perf.track("features"):
                        feature = self.feature_extractor.extract(track)
                    self._seq[track.track_id].append(feature)

                    with self._perf.track("rules"):
                        rule_decision = self.rule_engine.evaluate(feature)
                    with self._perf.track("temporal_ml"):
                        ml_prob = self.temporal_model.predict(
                            list(self._seq[track.track_id]),
                            track_id=track.track_id,
                        )
                    with self._perf.track("risk"):
                        event = self.risk_scorer.score(rule_decision, ml_prob)
                    with self._perf.track("output"):
                        self.alert_manager.emit(self.stream.stream_id, event)
                        self.training_logger.emit(self.stream.stream_id, feature, event)
                    risk_events[track.track_id] = event

                frame_count += 1
                fps = self._fps.tick()
                self.alert_manager.update_stream_health(self.stream.stream_id, fps=fps, source=self.source)
                needs_composed_frame = self.visualizer.cfg.enabled or self._live_stream_enabled
                if needs_composed_frame:
                    with self._perf.track("visualization"):
                        keep_running = self.visualizer.render(
                            frame=frame,
                            detections=detections,
                            tracks=tracks,
                            risk_events=risk_events,
                            fps=fps,
                            bed_zones=self._bed_zones,
                        )
                else:
                    keep_running = True
                publish_now = time.monotonic()
                if self._live_stream_enabled and publish_now - self._last_live_stream_ts >= self._live_stream_interval:
                    output_frame = self.visualizer.get_last_output_frame()
                    with self._perf.track("ui_stream"):
                        self.alert_manager.publish_frame(
                            self.stream.stream_id,
                            output_frame if output_frame is not None else frame,
                        )
                    self._last_live_stream_ts = publish_now
                packet.frame = None
                del frame
                del detections
                del poses
                if not keep_running:
                    self.stop()
                    break

                now = time.time()
                if now - last_metric_ts >= self.pipeline_cfg.metrics_interval_sec:
                    self.logger.info(
                        "stream=%s fps=%.2f memory_mb=%.2f captured=%d dropped=%d reconnects=%d latency=%s",
                        self.stream.stream_id,
                        fps,
                        self._perf.memory_usage_mb(),
                        int(getattr(self.source, "frames_captured", 0)),
                        int(getattr(self.source, "frames_dropped", 0)),
                        int(getattr(self.source, "reconnect_count", 0)),
                        self._perf.summary(),
                    )
                    last_metric_ts = now

                if self.pipeline_cfg.max_frames is not None and frame_count >= self.pipeline_cfg.max_frames:
                    break

        finally:
            self.source.stop()
            self.visualizer.close()
            self.training_logger.close()
            if self._owns_alert_manager:
                self.alert_manager.close()
            self.logger.info("stream=%s stopped after %d frames", self.stream.stream_id, frame_count)


class MultiStreamRunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        stream_cfgs = [StreamConfig(**item) for item in cfg["streams"]]
        output_cfg = cfg["output"]
        live_stream_cfg = output_cfg.get("live_stream", {})
        suppress_cfg = output_cfg.get("alert_suppression", {})
        writer_cfg = output_cfg.get("async_writer", {})
        self.alert_manager = AlertManager(
            json_log_path=output_cfg["json_log_path"],
            enable_api=output_cfg["enable_rest_api"],
            api_host=output_cfg["rest_api_host"],
            api_port=output_cfg["rest_api_port"],
            frame_jpeg_quality=int(live_stream_cfg.get("jpeg_quality", 80)),
            dedupe_window_sec=float(suppress_cfg.get("dedupe_window_sec", 1.0)),
            emit_on_level_change_only=bool(suppress_cfg.get("emit_on_level_change_only", True)),
            logger_name="alerts",
            log_queue_size=int(writer_cfg.get("alert_queue_size", 2048)),
            log_batch_size=int(writer_cfg.get("alert_batch_size", 64)),
            frame_queue_size=int(writer_cfg.get("frame_queue_size", 2)),
        )
        pipeline_cfg = PipelineConfig(**cfg["pipeline"])
        inference_backends = SharedInferenceBackends() if pipeline_cfg.share_inference_backends else None
        self.pipelines = [
            RiskDetectionPipeline(
                stream=s,
                cfg=cfg,
                alert_manager=self.alert_manager,
                inference_backends=inference_backends,
            )
            for s in stream_cfgs
        ]

    def stop(self) -> None:
        for p in self.pipelines:
            p.stop()

    def run(self) -> None:
        if not self.pipelines:
            return
        self.alert_manager.start()

        def _sig_handler(signum, frame):
            _ = (signum, frame)
            self.stop()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        try:
            with ThreadPoolExecutor(max_workers=len(self.pipelines)) as ex:
                futures = [ex.submit(p.run) for p in self.pipelines]
                for f in futures:
                    f.result()
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()
            self.alert_manager.close()
