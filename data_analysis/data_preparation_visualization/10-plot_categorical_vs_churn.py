#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Display a (12, 8) bar plot showing the churn
    rate (Yes proportion) per category for a given
    column with 45° rotated x-labels."""
    plt.figure(figsize=(12, 8))
    # Group by category, get % of Yes/No within each group
    churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True)
    # Filter just the data whit yes churn
    churn_rate_yes = churn_rate[:, 'Yes']
    # Define x and y
    x = churn_rate_yes.index
    y = churn_rate_yes.values
    plt.bar(x, y,)
    # column with 45° rotated x-labels.
    plt.xticks(rotation=45)
    # Give a name for y label
    plt.ylabel("Churn Rate")
    # Give a name for the title
    plt.title(f"Churn Rate by {col}")
    plt.show()
