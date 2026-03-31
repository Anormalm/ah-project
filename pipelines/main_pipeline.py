from __future__ import annotations

import signal
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from detection.yolo_detector import MockDetectionEngine, UltralyticsYOLOEngine, YOLOPersonDetector
from features.feature_extractor import FeatureExtractor
from ingestion.rtsp_stream import create_rtsp_source
from ingestion.video_loader import create_video_source
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
    TorchTransformerLiteInferenceEngine,
)
from tracking.tracker import ByteTrackLikeTracker
from utils.logger import setup_logger
from utils.performance import FPSMonitor, PerformanceTracker
from utils.schemas import Detection


class StreamConfig(BaseModel):
    stream_id: str
    type: str = Field(pattern="^(webcam|video|rtsp|dummy)$")
    source: str | int


class PipelineConfig(BaseModel):
    device: str = "auto"
    fps: float = 15.0
    buffer_size: int = 4
    max_frames: int | None = None
    metrics_interval_sec: float = 3.0
    sequence_len: int = 16


class RiskDetectionPipeline:
    def __init__(self, stream: StreamConfig, cfg: dict[str, Any], alert_manager: AlertManager | None = None) -> None:
        self.stream = stream
        self.cfg = cfg
        self.logger = setup_logger(name=f"pipeline.{stream.stream_id}", level=cfg["logging"]["level"])
        self.pipeline_cfg = PipelineConfig(**cfg["pipeline"])
        self.device = self._resolve_device(self.pipeline_cfg.device)

        self.source = self._build_source()
        self.detector = self._build_detector()
        self.pose_estimator = self._build_pose_estimator()
        self.tracker = ByteTrackLikeTracker(
            iou_threshold=cfg["tracking"]["iou_threshold"],
            max_misses=cfg["tracking"]["max_misses"],
        )

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
        self._live_stream_enabled = bool(live_stream_cfg.get("enabled", False))
        if alert_manager is None:
            self.alert_manager = AlertManager(
                json_log_path=alert_cfg["json_log_path"],
                enable_api=alert_cfg["enable_rest_api"],
                api_host=alert_cfg["rest_api_host"],
                api_port=alert_cfg["rest_api_port"],
                frame_jpeg_quality=int(live_stream_cfg.get("jpeg_quality", 80)),
                logger_name=f"alerts.{stream.stream_id}",
            )
            self._owns_alert_manager = True
        else:
            self.alert_manager = alert_manager
            self._owns_alert_manager = False
        self.alert_manager.register_stream(stream.stream_id)
        vis_cfg = VisualizationConfig(**alert_cfg.get("visualization", {}))
        self.visualizer = Visualizer(stream_id=stream.stream_id, cfg=vis_cfg)
        self.training_logger = TrainingDataLogger(alert_cfg.get("training_log_path"))

        self._stop_event = threading.Event()
        self._perf = PerformanceTracker()
        self._fps = FPSMonitor()
        self._seq: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.pipeline_cfg.sequence_len))

    def _build_source(self):
        fps = self.pipeline_cfg.fps
        buffer_size = self.pipeline_cfg.buffer_size
        if self.stream.type == "rtsp":
            return create_rtsp_source(
                rtsp_url=str(self.stream.source),
                fps=fps,
                buffer_size=buffer_size,
                reconnect_interval_sec=self.cfg["ingestion"]["rtsp_reconnect_sec"],
                max_retries=self.cfg["ingestion"]["rtsp_max_retries"],
            )
        webcam_options = {
            "requested_fps": self.cfg["ingestion"].get("webcam_requested_fps"),
            "width": self.cfg["ingestion"].get("webcam_width"),
            "height": self.cfg["ingestion"].get("webcam_height"),
        }
        return create_video_source(
            self.stream.type,
            self.stream.source,
            fps=fps,
            buffer_size=buffer_size,
            webcam_options=webcam_options,
        )

    def _resolve_device(self, requested: str) -> str:
        if requested not in {"auto", "cpu", "cuda"}:
            return requested
        if requested == "cpu":
            return "cpu"
        try:
            import torch
            import torchvision
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            try:
                boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
                scores = torch.tensor([0.9], device="cuda")
                torchvision.ops.nms(boxes, scores, 0.5)
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"

    def _build_detector(self) -> YOLOPersonDetector | None:
        det_cfg = self.cfg["detection"]
        backend_type = det_cfg["backend"]
        if backend_type == "none":
            return None
        if backend_type == "ultralytics":
            backend = UltralyticsYOLOEngine(
                model_path=det_cfg["model_path"],
                device=self.device,
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
            backend = UltralyticsPoseEngine(
                model_path=pose_cfg["model_path"],
                device=self.device,
                min_match_iou=pose_cfg.get("match_iou_threshold", 0.1),
                input_size=pose_cfg.get("input_size", 256),
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
        elif backend_type == "none":
            backend = NullTemporalEngine()
        else:
            backend = HeuristicTemporalEngine()
        return TemporalRiskModel(
            backend=backend,
            sequence_len=self.pipeline_cfg.sequence_len,
            infer_interval=int(tm_cfg.get("infer_interval", 1)),
            min_infer_steps=int(tm_cfg.get("min_infer_steps", 2)),
        )

    def _detections_from_poses(self, poses) -> list[Detection]:
        det_cfg = self.cfg["detection"]
        threshold = float(det_cfg.get("conf_threshold", 0.0))
        detections: list[Detection] = []
        for pose in poses:
            confs = [k[2] for k in pose.keypoints]
            score = float(sum(confs) / max(len(confs), 1))
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
                with self._perf.track("visualization"):
                    keep_running = self.visualizer.render(
                        frame=frame,
                        detections=detections,
                        tracks=tracks,
                        risk_events=risk_events,
                        fps=fps,
                        bed_zones=self._bed_zones,
                    )
                if self._live_stream_enabled:
                    with self._perf.track("ui_stream"):
                        self.alert_manager.publish_frame(self.stream.stream_id, frame)
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
                        "stream=%s fps=%.2f memory_mb=%.2f latency=%s",
                        self.stream.stream_id,
                        fps,
                        self._perf.memory_usage_mb(),
                        self._perf.summary(),
                    )
                    last_metric_ts = now

                if self.pipeline_cfg.max_frames is not None and frame_count >= self.pipeline_cfg.max_frames:
                    break

        finally:
            self.source.stop()
            self.visualizer.close()
            self.logger.info("stream=%s stopped after %d frames", self.stream.stream_id, frame_count)


class MultiStreamRunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        stream_cfgs = [StreamConfig(**item) for item in cfg["streams"]]
        output_cfg = cfg["output"]
        live_stream_cfg = output_cfg.get("live_stream", {})
        self.alert_manager = AlertManager(
            json_log_path=output_cfg["json_log_path"],
            enable_api=output_cfg["enable_rest_api"],
            api_host=output_cfg["rest_api_host"],
            api_port=output_cfg["rest_api_port"],
            frame_jpeg_quality=int(live_stream_cfg.get("jpeg_quality", 80)),
            logger_name="alerts",
        )
        self.pipelines = [RiskDetectionPipeline(stream=s, cfg=cfg, alert_manager=self.alert_manager) for s in stream_cfgs]

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

