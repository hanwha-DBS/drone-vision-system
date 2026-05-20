# Hard Negative Guide

## Why this matters

Drone mining footage produces false positives because the detector sees many person-like textures:

- rock piles
- shadows
- excavator arms
- orange debris
- white gravel patches
- equipment edges

If these negatives are not labeled into the workflow, the model keeps learning shortcuts.

## What to collect

Create a hard-negative pool with:

- empty scenes with no people
- scenes with only rocks and rubble
- scenes with high-visibility vests missing the worker
- excavators, dumpers, loaders, hoses, rails, barriers
- backlit and low-contrast areas
- front-view and back-view workers at very small scale

## Recommended dataset mix

For every 10 positive images, aim for:

- 3 to 5 empty hard-negative images
- 2 to 3 difficult mixed images with tiny people

## Annotation rules

- If there is no target object, keep the label file empty.
- Do not draw boxes on rocks or shadows.
- Label ambiguous tanker-like vehicles as `truck` unless confirmed.
- Keep front-view, back-view, crouched, and partially occluded workers.

## Training strategy

1. Run the current detector on fresh drone footage.
2. Save false positives as review candidates.
3. Move the reviewed frames into the hard-negative image pool.
4. Retrain every time the hard-negative pool grows materially.

## Expected outcome

This will usually reduce:

- rock-as-person false positives
- machinery-edge false positives
- unstable one-frame detections
