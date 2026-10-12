# Large Language Models (LLMs)

This project uses pre-trained Hugging Face models to see how the three main LLM families work. First, it builds masked language modeling with RoBERTa step by step (load the model, tokenize, find the `<mask>` tokens, get the logits and decode them), then does the same in one line with a `fill-mask` pipeline. After that, it uses pipelines for translation (Helsinki-NLP and M2M100, encoder-decoder), text generation (GPT-2 and DistilGPT-2, decoder-only), image classification with a Vision Transformer (ViT), and image captioning with the BLIP vision-language model.

## Tasks

| # | Task | File | Status |
|---|---|---|---|
| 0 | (MLM) Load RoBERTa | `0-load_mlm.py` | Done |
| 1 | (MLM) Tokenization | `1-tokenization.py` | Done |
| 2 | (MLM) Get Mask Indices | `2-get_mask_index.py` | Done |
| 3 | (MLM) Compute Mask Logits | `3-compute_mask_logits.py` | Done |
| 4 | (MLM) Decode Mask Predictions | `4-decode_mask_predictions.py` | Done |
| 5 | (Encoder-Only) Fill Mask | `5-fill_mask.py` | Done |
| 6 | (Encoder-Decoder) Translation | `6-translate_text.py` | Done |
| 7 | (Decoder-Only) Text Generation | `7-text_generator.py` | Done |
| 8 | Performing Computer Vision Tasks | `8-image_classifier.py` | Done |
| 9 | VLM for Image Captioning | `9-image_captioner.py` | Done |

0 of 10 tasks complete.

## Dataset

- `images.rar` — two test images (`dog.jpg`, `animal.PNG`), used for tasks 8–9
- Pre-trained models from the Hugging Face Hub: `roberta-base` (tasks 0–5), `Helsinki-NLP/opus-mt-*` and `facebook/m2m100_418M` (task 6), `gpt2` and `distilgpt2` (task 7), `google/vit-base-patch16-224` (task 8), `Salesforce/blip-image-captioning-base` (task 9)

## Requirements

- Python 3.11, pycodestyle 2.14.0
- torch 2.8.0+cpu, transformers 4.56.1, sentence-transformers 5.1.0, tiktoken 0.11.0, sentencepiece 0.2.2, tabulate 0.9.0
- Every module and function documented
- Files start with `#!/usr/bin/env python3`, end with a newline, and are executable

## Author

Marcelo Rech — [github.com/marcelorech1983](https://github.com/marcelorech1983)
