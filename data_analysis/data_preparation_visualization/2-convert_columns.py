#!/usr/bin/env python3
"""This module provides functions for cleaning and converting data structures,
specifically targeting column type casting and mapping for data analysis
pipelines."""
import pandas as pd


def convert_columns(df):
    """Convert TotalCharges to numeric (errors to NaN)
    and SeniorCitizen to 'No'/'Yes'."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
    return df
