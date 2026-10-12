#!/usr/bin/env python3
"""Turn mask logits back into readable vocabulary tokens."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """Return one list of token strings per mask, one per vocab id."""
    decoded_tokens = []
    for logits in mask_logits_list:
        # decode every id of the vocabulary to its word
        words = []
        for token_id in range(len(logits)):
            word = tokenizer.decode([token_id])
            # " improve" -> "improve"
            words.append(word.strip())
        decoded_tokens.append(words)
    return decoded_tokens
