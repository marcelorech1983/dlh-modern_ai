#!/usr/bin/env python3
"""Computer Vision Architectures.
This module provides functions for building, customizing, and training
convolutional neural network (CNN) architectures in Keras."""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Builds a MobileNetV1 depthwise separable convolution block."""
    x = keras.layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                     use_bias=False)(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters, 1, padding='same',
                            use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x


def mobilenet_backbone(inputs):
    """Builds the MobileNetV1 feature extraction backbone."""
    x = keras.layers.Conv2D(32, 3, strides=2, padding='same',
                            use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = depthwise_separable_conv(x, 64, stride=1)
    x = depthwise_separable_conv(x, 128, stride=2)
    x = depthwise_separable_conv(x, 128, stride=1)
    x = depthwise_separable_conv(x, 256, stride=2)
    x = depthwise_separable_conv(x, 256, stride=1)
    x = depthwise_separable_conv(x, 512, stride=2)

    for _ in range(5):
        x = depthwise_separable_conv(x, 512, stride=1)

    x = depthwise_separable_conv(x, 1024, stride=2)
    x = depthwise_separable_conv(x, 1024, stride=1)

    return x
