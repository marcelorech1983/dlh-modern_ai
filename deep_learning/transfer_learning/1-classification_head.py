#!/usr/bin/env python3
"""Transfer Learning for Computer Vision.
This module provides functions for feature extraction, model fine-tuning,
and building transfer learning pipelines using pretrained CNNs in Keras."""

from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """Attaches a custom classification head
    with a dense layer to a feature extractor."""
    x = base_model.output
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(base_model.input, outputs)
