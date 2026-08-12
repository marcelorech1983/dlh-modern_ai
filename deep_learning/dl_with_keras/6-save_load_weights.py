#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def save_model_weights(model, filepath):
    """Saves only the weights of a Keras model to a file path."""
    model.save_weights(filepath)
    return None


def load_model_weights(model, filepath):
    """Loads saved weights into an existing Keras model."""
    model.load_weights(filepath)
    return None
