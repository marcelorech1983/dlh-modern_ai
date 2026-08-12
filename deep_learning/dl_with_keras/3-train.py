#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""


def train_model(model, X, Y, epochs, verbose=1):
    """Trains a Keras model using input features and target labels."""
    model.fit(X, Y, epochs=epochs, verbose=verbose)
    return None
