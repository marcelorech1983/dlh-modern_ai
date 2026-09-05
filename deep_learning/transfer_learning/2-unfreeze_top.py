#!/usr/bin/env python3
"""Transfer Learning for Computer Vision.
This module provides functions for feature extraction, model fine-tuning,
and building transfer learning pipelines using pretrained CNNs in Keras."""


def unfreeze_top_layers(model, n_layers):
    """Unfreezes the last n layers of a base model for fine-tuning."""
    for layer in model.layers[-n_layers:]:
        layer.trainable = True
    return None
