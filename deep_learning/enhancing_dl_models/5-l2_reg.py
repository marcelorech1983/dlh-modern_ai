#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def build_model_with_L2_regularization(
        input_dim, hidden_units, n_layers, lambda_l2):
    """Builds a Keras neural network with L2 kernel weight
    regularization across hidden layers."""
    regularizer = keras.regularizers.L2(lambda_l2)
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(hidden_units, kernel_regularizer=regularizer,
                           activation="relu")(inputs)
    for i in range(n_layers - 1):
        x = keras.layers.Dense(
            hidden_units,
            kernel_regularizer=regularizer, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
