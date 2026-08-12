#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def save_model(model, filepath):
    """Saves a Keras model to the specified file path."""
    model.save(filepath)
    return None


def load_model(filepath):
    """Loads a saved Keras model from the specified file path."""
    model = keras.models.load_model(filepath)
    return model
