#!/usr/bin/env python3
"""Translate text between languages with a pre-trained model."""
from transformers import pipeline


def translate_text(model_name, src_lang=None, tgt_lang=None):
    """Create a translation pipeline for the given model."""
    # Helsinki-NLP models already know their language pair;
    # multilingual ones like m2m100 need src_lang/tgt_lang spelled out
    if src_lang is not None and tgt_lang is not None:
        translator = pipeline("translation", model=model_name,
                              src_lang=src_lang, tgt_lang=tgt_lang,
                              framework="pt")
    else:
        translator = pipeline("translation", model=model_name, framework="pt")
    return translator
