#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """Configures and returns a Keras SGD, Adam, or RMSprop optimizer.
    Applies the relevant parameters based on the selected optimizer name."""

    if name == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)
    elif name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(
            learning_rate=learning_rate, rho=rho)

    return optimizer
