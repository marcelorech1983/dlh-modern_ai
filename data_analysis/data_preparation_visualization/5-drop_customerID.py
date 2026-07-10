#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""


def drop_customerID(df):
    """Drop the customerID column from the DataFrame to remove
    unique identifiers without predictive value."""
    df = df.drop(columns=["customerID"])
    return df
