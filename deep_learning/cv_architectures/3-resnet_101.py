#!/usr/bin/env python3
"""Computer Vision Architectures.
This module provides functions for building, customizing, and training
convolutional neural network (CNN) architectures in Keras."""
from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Builds a ResNet bottleneck residual block:
    1x1 reduce -> 3x3 -> 1x1 expand, with a residual connection."""
    shortcut = x

    y = keras.layers.Conv2D(filters, 1, strides=stride,
                            padding='same')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    y = keras.layers.Conv2D(filters, 3, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    y = keras.layers.Conv2D(filters * 4, 1, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    if downsample:
        shortcut = keras.layers.Conv2D(filters * 4, 1, strides=stride,
                                       padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    out = keras.layers.Add()([y, shortcut])
    out = keras.layers.ReLU()(out)

    return out


def make_layer(x, blocks, filters, stride=1, name=None):
    """Stacks one ResNet stage: a first block with a projection
    shortcut, then identity blocks."""
    x = bottleneck_block(x, filters, stride=stride, downsample=True,
                         name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(x, filters, stride=1, downsample=False,
                             name=f'{name}_block{i+1}')
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """Builds the ResNet-101 architecture."""
    inputs = keras.Input(shape=input_shape)

    # conv1
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # max pool
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # the four stages
    x = make_layer(x, 3, 64, stride=1, name='conv2')
    x = make_layer(x, 4, 128, stride=2, name='conv3')
    x = make_layer(x, 23, 256, stride=2, name='conv4')
    x = make_layer(x, 3, 512, stride=2, name='conv5')

    # head
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)
