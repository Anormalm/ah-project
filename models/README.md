# Models Directory

Place model artifacts here:
- `yolo26s-pose.pt` for the balanced edge-SOTA pose profile
- `yolo26n-pose.pt` for the multi-stream pose profile
- `movenet_lightning.torchscript` for pose estimation
- `temporal_gru.pt` for temporal model weights
- `temporal_stgcn.pt` for the optional skeleton graph-temporal model

Future deployment artifacts:
- ONNX exports (`*.onnx`)
- TensorRT engines (`*.engine`)

Jetson Orin NX engines must be built on the target device. The balanced SOTA
profile expects `yolo26s-pose-fp16-640.engine`; the throughput profile expects
`yolo26n-pose-fp16-512.engine`. Create them with
`scripts/export_jetson_tensorrt.py`. The adjacent `.engine.json` manifest records
the source/engine hashes and build environment. See `SOTA_MODELS.md` before
selecting or training a production model.
