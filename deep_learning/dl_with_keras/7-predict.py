#!/usr/bin/env python3
"""Introduction to Deep Learning with Keras.
This module contains simple functions to build, train, and test
neural networks using the Keras library."""
import tensorflow as tf


def predict(model, X, verbose=0):
    """Generates predicted class labels for
    input data using a Keras model."""
    prob = model.predict(X, verbose=verbose)
    predictions = tf.argmax(prob, axis=1)
    return predictions
