#!/usr/bin/env python3
"""Get the raw model scores (logits) at each <mask> position."""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Run one forward pass and return the logits for each <mask>."""
    # one forward pass, no gradients needed for inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # one score per vocabulary word, at each mask position
    mask_logits_list = []
    for index in mask_indices:
        mask_logits_list.append(logits[0, index])
    return mask_logits_list
