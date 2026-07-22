#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Create and return a boosting model based on a given name,
    setting up its tree count and random seed."""
    model = {
        "adaboost": ensemble.AdaBoostClassifier,
        "gradientboosting": ensemble.GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "lightgbm": lgb.LGBMClassifier
    }
    if name not in model:
        raise ValueError(f"Unknown model name {name}")
    # Get the actual class from the dictionary
    ClassifierClass = model[name]
    # Initialize it
    if name == "lightgbm":
        untrained = ClassifierClass(
            n_estimators=n_estimators, random_state=random_state, verbose=-1
        )
    else:
        untrained = ClassifierClass(
            n_estimators=n_estimators, random_state=random_state
        )
    return untrained
