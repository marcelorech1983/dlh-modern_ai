#!/usr/bin/env python3
"""Turn a sentence (or a list of sentences) into RoBERTa tokens."""
from transformers import RobertaTokenizer


def tokenize_text(model_name, sentence, padding=True):
    """Tokenize the sentence and return the tokenizer and the inputs."""
    # tokenizer has to match the model, not just any RoBERTa one
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # text -> token ids, as PyTorch tensors
    inputs = tokenizer(sentence, padding=padding, return_tensors="pt")
    return tokenizer, inputs
