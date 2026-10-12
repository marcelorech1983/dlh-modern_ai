#!/usr/bin/env python3
"""Write a short caption for an image with the BLIP model."""
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def image_captioner(model, image_path, max_new_tokens):
    """Generate a caption for the image and return it as a string."""
    # processor bundles the image prep and the tokenizer together
    processor = BlipProcessor.from_pretrained(model)
    blip = BlipForConditionalGeneration.from_pretrained(model)

    # force RGB, since PNGs can carry a 4th (alpha) channel
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    output = blip.generate(**inputs, max_new_tokens=max_new_tokens)

    # token ids -> text, without [SEP] and other special tokens
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
