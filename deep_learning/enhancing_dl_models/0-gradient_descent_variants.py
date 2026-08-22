#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def train_with_gradient_descent_variant(
        variant, learning_rate, x_train, batch_size):
    """Returns an optimizer and batch size based
    on the gradient descent variant."""
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    if variant == "batch":
        bs = len(x_train)
    elif variant == "stochastic":
        bs = 1
    elif variant == "mini_batch":
        bs = batch_size

    return optimizer, bs
