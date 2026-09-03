#!/usr/bin/env python3
"""Computer Vision Architectures.
This module provides functions for building, customizing, and training
convolutional neural network (CNN) architectures in Keras."""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Builds a MobileNetV1 depthwise separable convolution block."""
    x = keras.layers.DepthwiseConv2D(
        3, strides=stride, padding='same', use_bias=False)(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x
