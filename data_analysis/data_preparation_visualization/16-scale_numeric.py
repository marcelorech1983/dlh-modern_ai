#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
from sklearn import preprocessing


def scale_numeric(df):
    """Standardize 'MonthlyCharges' and 'TotalCharges'
    using StandardScaler to achieve a mean of 0 and a
    standard deviation of 1."""
    # Initialize the scaler tool
    scaler = preprocessing.StandardScaler()
    # Create a variable to store both columns
    scaling_cols = ['MonthlyCharges', 'TotalCharges']
    # Scaling whit the method fit_transform()
    df[scaling_cols] = scaler.fit_transform(df[scaling_cols])
    return df
