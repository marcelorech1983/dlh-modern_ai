#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def build_model_initializer_by_activation(
        input_dim, hidden_units, activation):
    """Builds a Keras model using weight initializers
    matched to the activation function."""
    glorot_init = keras.initializers.GlorotUniform()
    he_init = keras.initializers.HeNormal()
    if activation == "sigmoid" or activation == "tanh":
        initializer = glorot_init
    elif activation == "relu" or activation == "leaky_relu":
        initializer = he_init
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units, kernel_initializer=initializer,
                           activation=activation),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model
