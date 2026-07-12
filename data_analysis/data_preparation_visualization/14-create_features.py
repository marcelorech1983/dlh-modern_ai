#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import pandas as pd


def create_features(df):
    """Build 'NumServices' and 'TenureGroup' from the raw columns,
    then drop the columns that were used to create them."""
    # Get all Yes/No service column and sum all yes through axis=1 rows
    services_count = (df[['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV',
                          'StreamingMovies']] == 'Yes').sum(axis=1)
    # Handle the column InternetService store 1 for every row that is not a No
    has_internet = (df['InternetService'] != 'No') * 1
    # Create the new feature adding services_count and has_internet
    df['NumServices'] = services_count + has_internet
    # Cutoff points for each tenure group. Last one is infinity
    # so anything above 60 still gets captured, no matter what.
    edges = [0, 12, 24, 48, 60, float('inf')]
    # Labels, same order as the edges above
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    # pd.cut splits tenure into the ranges above, then replaces
    # each value with the matching label
    df['TenureGroup'] = pd.cut(df['tenure'], bins=edges, labels=labels)
    # Drops the original columns that were used to create the new ones
    df = df.drop(columns=['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV',
                          'StreamingMovies', 'InternetService', 'tenure'])
    # Returns the modified DataFrame
    return df
