#!/usr/bin/env python3
"""Find where the <mask> tokens are in a tokenized sentence."""


def get_mask_index(inputs, tokenizer):
    """Return a list with the index of every <mask> token."""
    # token ids of the first (and only) sentence
    ids = inputs["input_ids"][0]

    # keep the positions where the id is <mask>
    mask_indices = []
    for i in range(len(ids)):
        if int(ids[i]) == tokenizer.mask_token_id:
            mask_indices.append(i)

    # no mask means there is nothing to predict
    if len(mask_indices) == 0:
        raise ValueError("No <mask> token found in the input!")
    return mask_indices
