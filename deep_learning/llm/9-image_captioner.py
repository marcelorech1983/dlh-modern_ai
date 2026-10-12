#!/usr/bin/env python3
"""Write a short caption for an image with the BLIP model."""
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def image_captioner(model, image_path, max_new_tokens):
    """Generate a caption for the image and return it as a string."""
    # load the processor (image prep + tokenizer) and the model
    processor = BlipProcessor.from_pretrained(model)
    blip = BlipForConditionalGeneration.from_pretrained(model)

    # open the image, RGB because PNG files can have 4 channels
    image = Image.open(image_path).convert("RGB")

    # image -> PyTorch tensors
    inputs = processor(images=image, return_tensors="pt")

    # generate the caption token ids
    output = blip.generate(**inputs, max_new_tokens=max_new_tokens)

    # token ids -> text, without [SEP] and other special tokens
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
