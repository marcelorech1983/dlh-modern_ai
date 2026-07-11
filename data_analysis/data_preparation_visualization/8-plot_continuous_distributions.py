#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """Generate a two-column grid layout of histograms with a red KDE line (L)
    and boxplots (R) for each selected continuous numeric feature."""
    if columns_to_plot is None:
        # Keep only 'number' dtype columns (integers and floats)
        columns_to_plot = df.select_dtypes(include='number').columns.tolist()
    else:
        # Caller gave a specific list, so keep it as-is
        columns_to_plot = columns_to_plot

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(columns_to_plot):
        # Remove NaN values, histogram/KDE can't handle them
        data = df[col].dropna()

        # Left subplot: histogram (bars built from raw data)
        axes[i, 0].hist(data, bins=30, density=True, alpha=0.7,
                        edgecolor='black')
        # Build the KDE model (a "calculator" for the smooth curve)
        kde = stats.gaussian_kde(data)
        # 200 evenly spaced x-points to draw the curve smoothly
        x_range = np.linspace(data.min(), data.max(), 200)
        # Draw the KDE curve on top of the histogram, in red
        axes[i, 0].plot(x_range, kde(x_range), color='red', ls='--')
        # Set the titles for the histogram left subplot
        axes[i, 0].set_title(f"{col} Histogram + KDE")
        # Right subplot: horizontal boxplot (auto-computed stats)
        axes[i, 1].boxplot(data, vert=False)
        # Set the titles for the boxplot right subplot
        axes[i, 1].set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
