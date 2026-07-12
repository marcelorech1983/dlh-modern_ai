#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
from scipy import stats


def ttest_numeric(df):
    """Perform Welch's t-tests comparing numeric feature
    means between Churn=Yes and Churn=No groups, returning
    a dictionary of feature names and p-values."""
    table = (df.select_dtypes(include=['number']).columns.tolist())
    # Store all the results in a dictionary variable
    result = {}
    # Loop through the columns of the table
    for i in range(len(table)):
        # Store the data into two groups Yes and No
        yes = df[df['Churn'] == 'Yes'][table[i]]
        no = df[df['Churn'] == 'No'][table[i]]
        # Calculate t-test and p-value from those two groups
        # Welch's t-test (equal_var=False) - my two groups probably
        # don't have the same spread, so this is the safer pick
        t_stat, p_value = stats.ttest_ind(yes, no, equal_var=False)
        # The calculation of t-test gives you 2 results
        # You need p-value that is the second one
        p = p_value
        # Store the result in the result variable
        result[table[i]] = p
    return result
