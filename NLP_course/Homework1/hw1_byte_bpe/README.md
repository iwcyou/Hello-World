# Homework 1 reference implementation: Byte-level BPE

This directory contains a from-scratch implementation. It does not use
Hugging Face `tokenizers` or `transformers`.

## Reproduce

Run from this directory with Python 3.10+:

```bash
python -m unittest discover -s tests -v
python train_tokenizer.py --corpus-dir ../gene_related_papers_1000 \
  --output-dir outputs/tokenizer --vocab-size 1024
python evaluate_tokenizer.py --corpus-dir ../gene_related_papers_1000 \
  --tokenizer-dir outputs/tokenizer --output outputs/evaluation.json
python example_gpt2_usage.py
```

Training uses the first 900 lexicographically sorted filenames. Evaluation
uses the remaining 100. The fields are `article_title`, `article_abstract`,
and `article_text`, joined by newlines.

## Saved interface

```python
from byte_bpe_tokenizer import ByteBPETokenizer

tokenizer = ByteBPETokenizer.from_pretrained("outputs/tokenizer")
ids = tokenizer.encode("TP53 is a tumor suppressor.", add_bos=True, add_eos=True)
text = tokenizer.decode(ids)
batch = tokenizer(["TP53", "BRCA1"], padding=True, return_tensors="pt")
```

`tokenizer.vocab_size` can be passed to a later GPT-style model's embedding
and output layers. `pad_token_id`, `bos_token_id`, and `eos_token_id` are
stable and saved with the tokenizer.
