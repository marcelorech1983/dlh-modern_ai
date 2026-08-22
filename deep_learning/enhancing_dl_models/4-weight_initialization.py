#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def build_model_initializer_by_activation(
        input_dim, hidden_units, activation):
    """Builds a Keras model using weight initializers
    matched to the activation function."""
    if activation == "sigmoid" or activation == "tanh":
        initializer = keras.initializers.GlorotUniform()
    elif activation == "relu" or activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
    inputs = keras.Input(shape=(input_dim,))
    if activation == "leaky_relu":
        x = keras.layers.Dense(hidden_units,
                               kernel_initializer=initializer)(inputs)
        x = keras.layers.LeakyReLU()(x)
    else:
        x = keras.layers.Dense(hidden_units, kernel_initializer=initializer,
                               activation=activation)(inputs)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
