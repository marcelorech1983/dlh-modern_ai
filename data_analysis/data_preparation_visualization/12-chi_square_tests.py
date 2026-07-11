#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Perform Chi-square independence tests between each categorical
    feature and 'Churn', returning a dictionary of feature names
    and their corresponding p-values."""
    # Create a new table from df including just category columns
    # Droping the Churn column
    table = (
        df.select_dtypes(include=['object', 'category'])
        .drop(columns='Churn').columns.tolist()
    )
    # Store all the results in a dictionary variable
    result = {}
    # Loop through the columns of the table
    for i in range(len(table)):
        # Calculate the croos table
        c_tab = pd.crosstab(df[table[i]], df["Churn"])
        # Use the cross table to calculate the chi
        calc = stats.chi2_contingency(c_tab)
        # The calculation of chi gives you 4 results
        # You need p-value that is the second one
        p = calc[1]
        # Store the result in the result variable
        result[table[i]] = float(p)
    return result
