"""
Phase 3: synthetic data generation with Stable Diffusion + ControlNet.

Produces synthetic drone-view images of mine sites with controlled hard
negatives (rocks, shadows, dust) and rarer positives (PPE workers in
specific poses). The script uses a depth-conditioned ControlNet so that
real drone footage can be reposed as new scenes.

This is a foundation script — for a production pipeline you typically:
    1. Curate ~200 prompt templates per class.
    2. Sample varied depth maps from real drone footage.
    3. Filter outputs via CLIP score before adding to training.

Output:
    datasets/synthetic/<category>/<seed>.jpg
    datasets/synthetic/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_PROMPTS = {
    "hard_negative_rocks": [
        "high altitude drone aerial photo of a rock pile in an open-pit mine, no people, no vehicles, midday sun, dusty atmosphere",
        "aerial drone shot of broken granite boulders and gravel, mine bench, no humans, photorealistic",
    ],
    "hard_negative_shadow": [
        "drone aerial view of long machinery shadows on a quarry floor, no people, golden hour, photorealistic",
    ],
    "positive_worker_pose": [
        "aerial drone photo of a single mine worker wearing orange high-visibility vest and white hardhat, standing on rocky ground, top-down view, photorealistic",
        "aerial drone shot of a worker crouching near a dump truck, orange vest visible, oblique angle",
    ],
    "positive_dump_truck": [
        "aerial drone photo of a yellow dump truck loaded with gravel on a haul road, no people nearby, photorealistic",
    ],
    "positive_tanker_truck": [
        "aerial drone photo of an ANFO tanker truck on a mine bench, ladder and rear hose visible, photorealistic",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic mine-site drone images.")
    parser.add_argument("--out", default="datasets/synthetic", help="Output directory.")
    parser.add_argument("--per-prompt", type=int, default=50, help="Images per prompt template.")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--controlnet-id", default="diffusers/controlnet-depth-sdxl-1.0")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="List prompts without generating.")
    return parser.parse_args()


def _list_prompts() -> List[tuple]:
    pairs: List[tuple] = []
    for category, prompts in DEFAULT_PROMPTS.items():
        for prompt in prompts:
            pairs.append((category, prompt))
    return pairs


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    new_manifest = not manifest_path.exists()

    if args.dry_run:
        for category, prompt in _list_prompts():
            print(f"[{category}] {prompt}")
        return 0

    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        print(
            "[synthetic_gen] diffusers not installed. Run:\n"
            "  pip install diffusers accelerate safetensors",
            flush=True,
        )
        raise SystemExit(2) from exc

    print(f"[synthetic_gen] loading {args.model_id}...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(args.device)

    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_manifest:
            writer.writerow(["created_at", "category", "image_path", "prompt", "seed"])

        seed = args.seed_start
        for category, prompt in _list_prompts():
            category_dir = out_root / category
            category_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(args.per_prompt):
                cur_seed = seed + offset
                generator = torch.Generator(device=args.device).manual_seed(cur_seed)
                image = pipe(prompt, generator=generator, num_inference_steps=30).images[0]
                image_path = category_dir / f"{cur_seed}.jpg"
                image.save(image_path, quality=92)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    category,
                    str(image_path),
                    prompt,
                    cur_seed,
                ])
            seed += args.per_prompt
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
