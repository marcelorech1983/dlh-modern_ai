#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """Returns a Keras EarlyStopping callback configured to monitor
    performance and restore best weights."""
    callback = keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=verbose)
    return callback
