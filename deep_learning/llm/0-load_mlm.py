#!/usr/bin/env python3
"""Load a pre-trained RoBERTa model for masked language modeling."""
from transformers import RobertaForMaskedLM


def load_mlm(model_name):
    """Load RoBERTa for MLM and return the model ready for inference."""
    # download (or read from cache) the pre-trained weights
    model = RobertaForMaskedLM.from_pretrained(model_name)
    return model
