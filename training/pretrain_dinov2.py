"""
Phase 5: self-supervised pretraining with DINOv2 on unlabeled drone footage.

DINOv2 weights act as a strong, domain-adapted feature backbone that the
downstream detector can initialize from. The script is intentionally thin —
it shells out to the official `dinov2` training entrypoint with our config —
because re-implementing distillation/teacher-student is out of scope and
DINOv2 already exposes a configurable training loop.

Prerequisites:
    pip install dinov2  # or clone https://github.com/facebookresearch/dinov2
    # collect 50k+ unlabeled drone frames into datasets/unlabeled/

Workflow:
    1. Collect raw drone footage (no labels needed).
    2. Run this script — produces a checkpoint at runs/dinov2/checkpoint.pth.
    3. Convert the backbone weights and load them into the detector via
       `finetune_detector.py --model <converted.pt>`.

This is a long-running job (days on a single A100); usually run on a cluster.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 self-supervised pretraining.")
    parser.add_argument("--data-root", default="datasets/unlabeled", help="Directory with raw drone frames.")
    parser.add_argument("--output", default="runs/dinov2", help="Where to write the checkpoint.")
    parser.add_argument("--arch", default="vit_base_patch14", help="DINOv2 architecture variant.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--config", default=None, help="Override DINOv2 config yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without launching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists() or not any(data_root.iterdir()):
        print(f"[pretrain] aborting — no images in {data_root}", file=sys.stderr)
        return 1

    if shutil.which("dinov2") is None and shutil.which("torchrun") is None:
        print(
            "[pretrain] neither `dinov2` CLI nor `torchrun` is on PATH.\n"
            "  Install: pip install dinov2  (or clone the official repo)",
            file=sys.stderr,
        )
        return 2

    Path(args.output).mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        "-m", "dinov2.train.train",
        "--config-file", args.config or "dinov2/configs/train/vitb14.yaml",
        f"--output-dir={args.output}",
        f"train.dataset_path=ImageNet:split=TRAIN:root={data_root}:extra={data_root}",
        f"train.batch_size_per_gpu={args.batch_size}",
        f"optim.epochs={args.epochs}",
        f"student.arch={args.arch}",
    ]
    print("[pretrain] launching DINOv2:", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
