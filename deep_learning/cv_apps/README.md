# Computer Vision Applications

Object detection pipeline with YOLOv8, Albumentations, and OpenCV on a Pascal VOC 2012 subset (person, car, bicycle): preparing the dataset in YOLO format, applying basic and custom augmentation pipelines, training with augmentation, hyperparameter tuning, and tuning inference confidence/IoU thresholds.

## Tasks

| # | Task | File | Status |
|---|---|---|---|
| 0 | Download & Organize | `0-prep_data.py` | Done |
| 1 | Basic Transformations | `1-basic_aug.py` | Done |
| 2 | Albumentations custom Transformations | `2-custom_aug.py` | Done |
| 3 | Train & Augment | `3-train_aug.py` | Done |
| 4 | Hyperparameter Tuning | `4-tune_train.py` | Not started |
| 5 | Inference Tuning | `5-tune_inference.py` | Not started |

4 of 6 tasks complete.

## Dataset

- Pascal VOC 2012 (subset: `person`, `car`, `bicycle`), reorganized into YOLOv8 format under `datasets/detection/`, used across all tasks

## Requirements

- Python 3.11, pycodestyle 2.14.0
- numpy 2.0.2, matplotlib 3.10.0, opencv-python 4.12.0.88, torch 2.8.0, albumentations 2.0.8, ultralytics 8.4.7
- Every module and function documented
- Files start with `#!/usr/bin/env python3`, end with a newline, and are executable

## Author

Marcelo Rech — [github.com/marcelorech1983](https://github.com/marcelorech1983)
