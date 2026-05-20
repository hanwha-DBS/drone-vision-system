import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap auto-labeling settings for Grounding DINO + SAM review workflow."
    )
    parser.add_argument(
        "--config",
        default="configs/auto_labeling.json",
        help="Path to auto-labeling config json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [
            path
            for path in input_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    manifest = {
        "image_count": len(images),
        "prompts": config["prompts"],
        "grounding_dino": config["grounding_dino"],
        "sam3": config["sam3"],
        "label_map": config["label_map"],
        "images": [str(path) for path in images],
    }

    manifest_path = output_dir / "bootstrap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"bootstrap manifest written: {manifest_path}")
    print("next step:")
    print("  1. run Grounding DINO on the listed images with the configured prompts")
    print("  2. optionally refine masks/tracks with SAM 3")
    print("  3. review all pseudo labels before training")


if __name__ == "__main__":
    main()
