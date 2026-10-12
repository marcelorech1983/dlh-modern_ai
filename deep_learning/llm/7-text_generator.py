#!/usr/bin/env python3
"""Generate text from a prompt with a decoder-only model (GPT-2)."""
import transformers


def create_text_generator(model_name, prompt, max_new_tokens, temperature,
                          repetition_penalty, no_repeat_ngram_size):
    """Create a text-generation pipeline and run it on the prompt."""
    generator = transformers.pipeline("text-generation",
                                      model=model_name)

    # GPT-2 has no pad token, so we use the end-of-sequence token instead
    pad_id = generator.tokenizer.eos_token_id
    generator.model.generation_config.pad_token_id = pad_id

    output = generator(prompt,
                       max_new_tokens=max_new_tokens,
                       temperature=temperature,
                       repetition_penalty=repetition_penalty,
                       no_repeat_ngram_size=no_repeat_ngram_size,
                       pad_token_id=pad_id)
    return generator, output
