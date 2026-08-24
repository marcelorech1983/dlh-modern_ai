#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
from tensorflow import keras


def build_model(hp):
    """Builds a compiled Keras model with tunable
    architecture and learning rate for KerasTuner search."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    num_layers = hp.Int('num_layers', min_value=1, max_value=2)
    units = hp.Int('units', min_value=4, max_value=12, step=4)
    activation = hp.Choice('activation', values=['relu', 'sigmoid'])

    for i in range(num_layers):
        model.add(keras.layers.Dense(units=units, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
