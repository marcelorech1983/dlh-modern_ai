#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Set up and return an untrained Support Vector Machine
    (SVM) classifier configured with your chosen kernel function."""
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
