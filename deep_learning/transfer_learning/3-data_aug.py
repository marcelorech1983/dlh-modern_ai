#!/usr/bin/env python3
"""Transfer Learning for Computer Vision.
This module provides functions for feature extraction, model fine-tuning,
and building transfer learning pipelines using pretrained CNNs in Keras."""
from tensorflow import keras


def build_data_augmentation():
    """Creates a Keras Sequential model
    with seeded image augmentation layers."""
    model = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(0.15, seed=42),
            keras.layers.RandomZoom(0.15, seed=42),
            keras.layers.RandomContrast(0.1, seed=42),
        ]
    )
    return model
