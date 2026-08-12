#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Builds a sequential classification model.
    Args:
        input_dim: Number of input features.
        neurons_h: Number of hidden neurons.
    Returns:
        The compiled Keras model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(neurons_h, activation="sigmoid"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model
