#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """Returns an SGD optimizer configured with optional
    momentum or Nesterov acceleration."""

    if name == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    elif name == "SGD+Momentum":
        optimizer = keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum)
    elif name == "SGD+Momentum+Nesterov":
        optimizer = keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum, nesterov=nesterov)

    return optimizer
