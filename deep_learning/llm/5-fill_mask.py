#!/usr/bin/env python3
"""Do masked language modeling in one step with a pipeline."""
import transformers


def fill_mask(model_name, top_k):
    """Create a fill-mask pipeline that returns top_k predictions."""
    # the pipeline does tokenize + model + decode for us
    fill = transformers.pipeline("fill-mask", model=model_name,
                                 top_k=top_k)
    return fill
