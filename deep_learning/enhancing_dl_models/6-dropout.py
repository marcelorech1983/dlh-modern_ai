#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def build_model_with_dropout(input_dim, hidden_units, n_layers,
                             dropout_rate_input, dropout_rate_hidden):
    """Builds a Keras model with dropout regularization
    applied after input and hidden layers."""
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dropout(rate=dropout_rate_input)(inputs)
    for i in range(n_layers):
        x = keras.layers.Dense(
            hidden_units,
            activation="relu")(x)
        x = keras.layers.Dropout(rate=dropout_rate_hidden)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
