#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import tree
train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(X_train, y_train, X_test, y_test,
                             ccp_alphas, random_state, min_samples_leaf,
                             min_samples_split):
    """Train several decision trees using different pruning alpha values
    and compare how well they score on training and testing data."""
    # Define pre-pruning
    clfs = []
    train_scores = []
    test_scores = []

    for alpha in ccp_alphas:
        # Initialize classifier with pre-pruning params, ccp_alpha, and seed
        clf = tree.DecisionTreeClassifier(
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=alpha,
            random_state=random_state
        )
        # Train the model using the imported train_tree function
        train_tree(clf, X_train, y_train)
        # Collect trained model instance and accuracy scores
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    return clfs, train_scores, test_scores
