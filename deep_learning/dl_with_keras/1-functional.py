#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """Builds a functional Keras model with
    one hidden layer for classification."""
    inputs = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Dense(neurons_h, activation="sigmoid")(inputs)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
