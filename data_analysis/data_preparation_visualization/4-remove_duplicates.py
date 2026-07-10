#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""


def remove_duplicates(df):
    """Remove all duplicate rows from the DataFrame
    and return the deduplicated result."""
    df = df.drop_duplicates()
    return df
