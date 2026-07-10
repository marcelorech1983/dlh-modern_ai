#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """This function def visualizes whit a plot
    missing values in a DataFrame"""
    plt.figure(figsize=(12, 8))

    missing = df.isna()
    row_idx, col_idx = np.where(missing)
    plt.scatter(row_idx, col_idx, marker='|')

    plt.yticks(range(len(df.columns)), df.axes[1])

    plt.tight_layout()
    plt.show()
    return None
