#!/usr/bin/env python3
"""Computer Vision Architectures.
This module provides functions for building, customizing, and training
convolutional neural network (CNN) architectures in Keras."""
from tensorflow import keras


def compile_and_train_cnn(model, epochs, batch_size,
                          x_train, y_train, x_val, y_val,
                          optimizer_name='adam', optimizer_params=None):
    """Compiles and trains a CNN model,
    returning the model and its training history."""

    # Choose the optimizer
    if optimizer_params is None:
        optimizer_params = {}

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(**optimizer_params)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(**optimizer_params)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(**optimizer_params)
    elif optimizer_name == 'adagrad':
        optimizer = keras.optimizers.Adagrad(**optimizer_params)
    elif optimizer_name == 'adamax':
        optimizer = keras.optimizers.Adamax(**optimizer_params)
    elif optimizer_name == 'nadam':
        optimizer = keras.optimizers.Nadam(**optimizer_params)
    else:
        optimizer = keras.optimizers.Adam()

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val)
    )

    # Return the trained model and the history
    return model, history
