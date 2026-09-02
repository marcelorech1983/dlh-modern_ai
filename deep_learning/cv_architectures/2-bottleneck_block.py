#!/usr/bin/env python3
"""Computer Vision Architectures.
This module provides functions for building, customizing, and training
convolutional neural network (CNN) architectures in Keras."""
from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """BBuilds a ResNet bottleneck residual block:
    1x1 reduce -> 3x3 -> 1x1 expand, with a residual connection."""
    shortcut = x  # save original input for the skip connection

    # 1x1 conv: shrink channels
    y = keras.layers.Conv2D(filters, kernel_size=1, strides=stride,
                            padding='same')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    # 3x3 conv: main spatial pattern detection
    y = keras.layers.Conv2D(filters, kernel_size=3, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    # 1x1 conv: expand channels back out by factor of 4
    y = keras.layers.Conv2D(filters * 4, kernel_size=1, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    # reshape shortcut if shape wouldn't otherwise match y
    if downsample:
        shortcut = keras.layers.Conv2D(filters * 4, kernel_size=1,
                                       strides=stride,
                                       padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    out = keras.layers.Add()([y, shortcut])  # residual connection
    out = keras.layers.ReLU()(out)  # final activation

    return out
