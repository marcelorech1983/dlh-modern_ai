#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """Logs Keras model training metrics and weights to TensorBoard."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_log_dir = log_dir + "/" + timestamp
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=full_log_dir, histogram_freq=1)

    model.fit(X, Y,
              epochs=epochs,
              verbose=verbose,
              callbacks=[tensorboard])
    return None
