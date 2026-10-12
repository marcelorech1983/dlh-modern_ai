#!/usr/bin/env python3
"""Classify images with a pre-trained Vision Transformer (ViT)."""
from transformers import pipeline


def image_classifier(model):
    """Create an image-classification pipeline for the given model."""
    # the pipeline resizes the image and returns the top labels
    classifier = pipeline("image-classification", model=model)
    return classifier
