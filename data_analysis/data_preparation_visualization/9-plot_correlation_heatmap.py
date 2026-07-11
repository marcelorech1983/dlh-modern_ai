#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """Compute pairwise correlations for continuous numeric
    features and display an annotated Seaborn heatmap mapped
    from vmin=-1 to vmax=1 using the coolwarm colormap."""
    plt.figure(figsize=(6, 5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
