#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """Plot bar charts for categorical columns in a grid layout,
    showing the count of each category with x-axis labels
    rotated 45 degrees."""
    if columns_to_plot is None:
        # Keep only 'object' dtype columns (categorical/text)
        obj_cols = df.select_dtypes(include='object')
        # Drop the target column, then get plain list of names
        columns_to_plot = obj_cols.drop(columns='Churn').columns.tolist()
    else:
        # Caller gave a specific list, so keep it as-is
        columns_to_plot = columns_to_plot

    # Fix the grid at 3 plots per row (matches reference layout)
    n_cols = 3
    # Round up rows needed: e.g. 16 cols / 3 -> 6 rows
    n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # axes is a grid (n_rows x n_cols); flatten it into one flat
    # list so it lines up with our flat list of column names
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        # Count how many times each category appears
        counts = df[col].value_counts()

        # Draw the bar chart on this specific subplot slot
        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col)

        # Rotate the x-axis labels 45 degrees, as required
        axes[i].tick_params(axis='x', rotation=45)

    # Remove any leftover empty subplot slots (grid may have
    # more slots than columns, e.g. 18 slots for 16 columns)
    for j in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
