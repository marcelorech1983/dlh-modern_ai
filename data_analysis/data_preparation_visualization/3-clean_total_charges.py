#!/usr/bin/env python3
"""This module provides functions for cleaning and
preprocessing telecom dataset features."""
import pandas as pd


def clean_total_charges(df, method='drop'):
    """Clean missing TotalCharges using 'drop' (remove rows),
    'median' (fill with median), or 'impute' (MonthlyCharges*tenure)."""
    df = df.copy()

    if method == 'drop':
        df = df.dropna(subset=["TotalCharges"])
    elif method == 'median':
        median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median)
    elif method == 'impute':
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"])

    return df
