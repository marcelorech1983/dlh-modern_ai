#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """Creates a deep sequential neural network for classification."""
    layers_list = [keras.layers.Input(shape=(input_dim,))]
    for neurons in hidden_layers:
        layers_list.append(keras.layers.Dense(neurons, activation="relu"))
    layers_list.append(keras.layers.Dense(10, activation="softmax"))
    model = keras.Sequential(layers_list)
    return model
