#!/usr/bin/env python3
"""This module provides functions for cleaning
and preprocessing dataset features."""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Generate and display a bar plot of the Churn column
    distribution using skyblue ('No') and salmon ('Yes') colors."""
    plt.figure(figsize=(12, 8))
    counts = df['Churn'].value_counts()
    x = counts.index
    y = counts.values
    bar_color = ["skyblue", "salmon"]
    plt.bar(x, y, color=bar_color)
    plt.ylabel("Count")
    plt.title("Churn distribution")
    return None
