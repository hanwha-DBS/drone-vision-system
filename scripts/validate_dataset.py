import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure.")
    parser.add_argument(
        "--data",
        default="configs/person_dump_tanker_dataset.yaml",
        help="Path to dataset yaml.",
    )
    return parser.parse_args()


def count_files(path: Path, suffixes):
    return sum(1 for item in path.glob("*") if item.suffix.lower() in suffixes)


def main():
    args = parse_args()
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(config["path"])

    print(f"dataset root: {root.resolve()}")
    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        print(f"[{split}] images={image_dir} labels={label_dir}")

        if not image_dir.exists():
            print("  - missing image directory")
            continue
        if not label_dir.exists():
            print("  - missing label directory")
            continue

        image_count = count_files(image_dir, {".jpg", ".jpeg", ".png", ".bmp"})
        label_count = count_files(label_dir, {".txt"})
        print(f"  - image files: {image_count}")
        print(f"  - label files: {label_count}")

        if image_count != label_count:
            print("  - warning: image/label counts differ")

    print("class mapping:")
    for key, value in config.get("names", {}).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
