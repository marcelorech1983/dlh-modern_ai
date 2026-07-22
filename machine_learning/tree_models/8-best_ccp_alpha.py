#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best ccp_alpha: highest test accuracy first,
    then smallest train-test gap, then largest alpha."""

    # Find every index with the max test score
    max_test_scores = max(test_scores)
    position = []
    for i, score in enumerate(test_scores):
        if score == max_test_scores:
            position.append(i)
    # If tied, grab the smallest difference between training and test accuracy
    if len(position) == 1:
        best_index = position[0]
    else:
        diffs = []
        pos_diff = []
        for i in position:
            diff = train_scores[i] - test_scores[i]
            diffs.append(diff)
            pos_diff.append(i)

        min_diff = min(diffs)
        position2 = []
        for i, diff in enumerate(diffs):
            if diff == min_diff:
                position2.append(pos_diff[i])

        # If still tied, pick the largest ccp_alpha
        if len(position2) == 1:
            best_index = position2[0]
        else:
            alpha_candidates = []
            for i in position2:
                alpha_candidates.append(ccp_alphas[i])
            max_alpha = max(alpha_candidates)
            best_index = position2[alpha_candidates.index(max_alpha)]

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
