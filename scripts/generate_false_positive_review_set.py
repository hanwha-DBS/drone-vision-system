import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a review manifest for likely false-positive frames."
    )
    parser.add_argument(
        "--frames-dir",
        default="datasets/review_candidates/frames",
        help="Directory containing extracted review frames.",
    )
    parser.add_argument(
        "--output",
        default="datasets/review_candidates/false_positive_review.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in frames_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "review_status", "notes"])
        for image_path in image_paths:
            writer.writerow([str(image_path), "pending", "check rocks/shadows/front-back workers"])

    print(f"review csv written: {output_path}")
    print(f"frames queued: {len(image_paths)}")


if __name__ == "__main__":
    main()
