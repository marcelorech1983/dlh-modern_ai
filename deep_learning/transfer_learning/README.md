# Transfer Learning for Computer Vision

Transfer learning pipeline with Keras/TensorFlow on CIFAR-10 and Caltech 101: building a frozen MobileNetV2 feature extractor, attaching a classification head, fine-tuning by selectively unfreezing backbone layers, applying data augmentation, and training a full two-phase (feature extraction + fine-tuning) classifier on Caltech 101.

## Tasks

| # | Task | File | Status |
|---|---|---|---|
| 0 | Frozen Feature Extractor | `0-frozen_extractor.py` | Not started |
| 1 | Classification Head | `1-classification_head.py` | Not started |
| 2 | Unfreezing Layers | `2-unfreeze_top.py` | Not started |
| 3 | Data Augmentation | `3-data_aug.py` | Not started |
| 4 | Knowledge Transfer: Taming the 101 | `4-transfer_101.py` | Not started |

0 of 5 tasks complete.

## Dataset

- `keras.datasets.cifar10` — CIFAR-10 images (32x32, 10 classes), used for tasks 0–3
- Caltech 101 — 101 object classes + background, used for task 4 (target: ≥85% validation accuracy)

## Requirements

- Python 3.11, pycodestyle 2.14.0
- numpy 2.0.2, pandas 2.2.2, tensorflow 2.18.0, matplotlib 3.10.0
- Every module and function documented
- Files start with `#!/usr/bin/env python3`, end with a newline, and are executable

## Author

Marcelo Rech — [github.com/marcelorech1983](https://github.com/marcelorech1983)
