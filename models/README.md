# Models Directory

Place model artifacts here:
- `yolov8n.pt` for Ultralytics YOLO detection
- `movenet_lightning.torchscript` for pose estimation
- `temporal_gru.pt` for temporal model weights

Future deployment artifacts:
- ONNX exports (`*.onnx`)
- TensorRT engines (`*.engine`)

Jetson Orin NX engines must be built on the target device. The default Jetson
profile expects `yolo11n-pose-fp16-384.engine`; create it with
`scripts/export_jetson_tensorrt.py`. The adjacent `.engine.json` manifest records
the source/engine hashes and build environment.
