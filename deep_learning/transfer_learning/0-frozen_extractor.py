#!/usr/bin/env python3
"""Transfer Learning for Computer Vision.
This module provides functions for feature extraction, model fine-tuning,
and building transfer learning pipelines using pretrained CNNs in Keras."""

from tensorflow import keras


def build_feature_extractor():
    """Loads a pretrained MobileNetV2 base model
    and freezes its weights for feature extraction."""
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, outputs)
