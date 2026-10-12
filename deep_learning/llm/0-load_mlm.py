#!/usr/bin/env python3
"""Load a pre-trained RoBERTa model for masked language modeling."""
from transformers import RobertaForMaskedLM


def load_mlm(model_name):
    """Load RoBERTa for MLM and return the model ready for inference."""
    # grabs the weights, cached copy if we already have one
    model = RobertaForMaskedLM.from_pretrained(model_name)
    return model
