#!/usr/bin/env python3
"""Load a pre-trained RoBERTa model for masked language modeling."""
import transformers


def load_mlm(model_name):
    """Load RoBERTa for MLM and return the model ready for inference."""
    # grabs the weights, cached copy if we already have one
    model = transformers.RobertaForMaskedLM.from_pretrained(model_name)

    # evaluation mode: dropout off, we only predict
    model.eval()
    return model
