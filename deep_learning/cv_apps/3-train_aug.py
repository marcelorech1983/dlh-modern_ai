#!/usr/bin/env python3
"""Computer Vision Applications.
This module provides tools and functions for object detection, image
segmentation, and dataset preprocessing using modern deep learning frameworks.
"""
from ultralytics import YOLO


def train_with_augmentation(
        data, model_path="yolov8n.pt", epochs=50, imgsz=640, batch=16,
        augmentation=True, yolo_aug_params=None,
        albumentations_transforms=None, save=True, plots=True,
        verbose=True):
    """Trains a YOLO model using configurable native
    or custom Albumentations augmentations. Executes model training with
    specified hyperparameters and returns the model and results."""
    yolo_model = YOLO(model_path)

    train_args = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }
    if not augmentation:
        train_args.update(NO_AUG)
    if yolo_aug_params:
        train_args.update(yolo_aug_params)
    if albumentations_transforms:
        train_args["augmentations"] = albumentations_transforms

    results = yolo_model.train(**train_args)
    return yolo_model, results
