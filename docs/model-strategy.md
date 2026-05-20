# Model Strategy

## Recommended production stack

This repository now supports a staged rollout:

1. `balanced-cpu-coco`
   - Use on CPU-only machines.
   - Model: `yolov8n.pt`
   - Targets: `person`, `car`, `truck`, `bus`
   - Goal: fast validation and operator testing.

2. `balanced-gpu-custom`
   - Use on NVIDIA GPU machines.
   - Recommended model family: `YOLO11m` or `YOLO11s`
   - Targets: `person`, `dump_truck`, `tanker_truck`, `truck`
   - Goal: real deployment with better small-object recall.

3. `precision-gpu-custom`
   - Use when recall matters more than latency.
   - Recommended model family: `YOLO11l`
   - Pair with tighter ROI filtering or reduced frame stride.

4. `next-gen-gpu-custom`
   - Use for the next upgrade cycle on a GPU workstation.
   - Recommended model family: `YOLO26m` or `YOLO26s`
   - Pair with temporal tracking and hard-negative retraining.

## Why this stack

- YOLO-family detectors are easier to fine-tune and deploy than heavier transformer-only stacks.
- Drone footage usually fails on generic COCO weights because people and vehicles are too small.
- Custom training matters more than choosing a larger generic model.
- Tracking should be added after detection quality is acceptable.

## Recommended tracker choices

1. `ByteTrack`
   - Best default for stable multi-object tracking.
   - Good fit for person and vehicle continuity.

2. `BoT-SORT`
   - Use if camera motion compensation and identity stability matter more.

## Immediate recommendation for this project

- Detection model for training: `YOLO11m` today, `YOLO26m` when your training machine is ready
- Edge/CPU fallback model: `YOLO11s` or `YOLOv8n`
- Tracker: `ByteTrack`
- Class set: `person`, `dump_truck`, `tanker_truck`, `truck`

## New data tooling recommendation

- Auto-label bootstrap: `Grounding DINO + SAM 3`
- Human review: mandatory before training
- Hard-negative refresh loop: every major field test

## Runtime switch points

- Edit `configs/runtime_detector.json` to point `gpu.model_path` to your trained weights.
- Keep `cpu.model_path` on a lightweight fallback model for machines without CUDA.
- If small objects are still missed, first reduce `frame_stride`, then raise `target_width`.
