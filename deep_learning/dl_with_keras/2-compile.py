#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """Compiles a Keras model with SGD optimizer
    and binary cross-entropy loss."""
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return None
