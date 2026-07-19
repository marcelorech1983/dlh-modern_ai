#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means clustering configurations from 2 up to
    a maximum number of clusters, calculating inertia and silhouette
    scores to help identify the ideal number of clusters."""
    silhouette_score = metrics.silhouette_score
    cluster_numbers = []
    inertias = []
    silhouettes = []
    for k in range(2, max_clusters + 1):
        kmeans = K_Means(X, k, random_state)
        labels = kmeans.labels_
        cluster_numbers.append(k)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return cluster_numbers, inertias, silhouettes
