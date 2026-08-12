#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""


def evaluate_model(model, X, Y, verbose=0):
    """Evaluates the performance of a
    trained Keras model on test data."""
    results = model.evaluate(X, Y, verbose=verbose)
    return results
