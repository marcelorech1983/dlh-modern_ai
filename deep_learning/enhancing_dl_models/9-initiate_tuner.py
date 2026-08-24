#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
import keras_tuner


def initiate_tuner(tuner_type, build_model, seed, hyperband_iterations,
                   max_trials, objective, overwrite=True):
    """Initializes and returns a Keras Tuner object of the requested
    type, ready to search for the best hyperparameters."""
    if tuner_type == 'Hyperband':
        tuner = keras_tuner.Hyperband(
            build_model,
            objective=objective,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            overwrite=overwrite)
    elif tuner_type == 'RandomSearch':
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=overwrite)
    elif tuner_type == 'BayesianOptimization':
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=overwrite)

    return tuner
