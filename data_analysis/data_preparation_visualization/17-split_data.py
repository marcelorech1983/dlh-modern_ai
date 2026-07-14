#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """Split the DataFrame into training and testing sets using
    stratified sampling to preserve the target class distribution."""
    # X gets all columns EXCEPT the target
    X = df.drop(columns=[target])
    # y gets ONLY the target column
    y = df[target]
    # Just shorter names
    ts = test_size
    rs = random_state
    # Split X and y together, stratify keeps the same Yes/No
    # ratio in both train and test sets as the original data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=ts,
        stratify=y,
        random_state=rs)
    return (X_train, X_test, y_train, y_test)
