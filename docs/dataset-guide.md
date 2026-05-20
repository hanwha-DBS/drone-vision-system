# Dataset Guide

## Goal

Train a detector for:

- `person`
- `dump_truck`
- `tanker_truck`
- `truck`

## Folder layout

```text
datasets/person_dump_tanker/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## Label format

Use YOLO text labels:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `0..1`.

## Class mapping

```text
0 person
1 dump_truck
2 tanker_truck
3 truck
```

## Data collection notes

- Sample both nadir and oblique drone views.
- Keep many examples where people are tiny.
- Include different altitudes, weather, and ground colors.
- Capture empty scenes too, so the model learns restraint.
- For vehicles, label by visible type, not by guesswork.
- If a tanker is ambiguous, label it as `truck` instead of forcing `tanker_truck`.

## Split recommendation

- `train`: 70%
- `val`: 20%
- `test`: 10%

Keep scenes separated by flight or location where possible to reduce leakage.

## Minimum useful dataset

- `person`: 2,000+ boxes
- `dump_truck`: 800+ boxes
- `tanker_truck`: 800+ boxes
- `truck`: 1,000+ boxes

More diverse small-object examples will help more than simply adding many near-duplicate frames.

## Annotation quality rules

- Draw tight boxes.
- Do not include long shadows.
- Do not merge nearby people into one box.
- Do not relabel the same vehicle type inconsistently across frames.
