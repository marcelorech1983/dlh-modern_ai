#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def get_optimizer_SGD_with_schedule(
        schedule_type, initial_lr, decay_steps, decay_rate, momentum):
    """Returns an SGD optimizer configured with momentum and
    a learning rate decay schedule."""

    if schedule_type == "exponential":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
    elif schedule_type == "inverse_time":
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=momentum)

    return optimizer, lr_schedule
