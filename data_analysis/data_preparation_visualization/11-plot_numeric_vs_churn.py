#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Display a (12, 8) plot with overlapping 30-bin
    histograms comparing a numeric feature's distribution
    for Churn=Yes and Churn=No, including a titled legend."""
    plt.figure(figsize=(12, 8))
    # Split the numeric column into two groups by Churn
    churn_no = df[df['Churn'] == 'No'][col]
    churn_yes = df[df['Churn'] == 'Yes'][col]

    # Overlapping histograms, one per group
    plt.hist(churn_no, bins=30, alpha=0.7, label='No')
    plt.hist(churn_yes, bins=30, alpha=0.7, label='Yes')

    # Give a name for x label
    plt.xlabel(f"{col}")
    # Give a name for the title
    plt.title(f"{col} Distribution by Churn")
    # Create the legend
    plt.legend(title='Churn')
    plt.show()
