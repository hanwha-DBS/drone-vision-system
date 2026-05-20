# Auto Labeling Guide

## Purpose

Use Grounding DINO and SAM-style segmentation/tracking tools to bootstrap labels faster before custom YOLO training.

## Recommended modern workflow

1. Grounding DINO for phrase-based coarse detection
2. SAM 3 for mask refinement when available
3. Human review in a labeling tool
4. Export reviewed labels back into YOLO format

## Good prompt set for this project

- `person`
- `dump truck`
- `tanker truck`
- `truck`

Optional support prompts for review-only passes:

- `bus`
- `excavator`
- `loader`

## Important caveat

Auto labels are only a bootstrap.
They are especially risky for:

- tiny front/back-view workers
- heavily occluded workers
- trucks partly hidden by rocks
- bright gravel textures that resemble safety gear

Always review before training.

## Files in this repo

- `configs/auto_labeling.json`
- `scripts/auto_label_bootstrap.py`
- `scripts/generate_false_positive_review_set.py`

## Suggested review loop

1. Auto-label raw images
2. Review and correct in a labeling tool
3. Merge corrected labels into `datasets/person_dump_tanker`
4. Retrain
5. Run detector on new footage
6. Export false positives for the next hard-negative round
