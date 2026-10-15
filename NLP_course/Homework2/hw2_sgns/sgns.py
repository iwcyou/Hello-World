"""Word-level preprocessing and a from-scratch PyTorch SGNS implementation."""
import ast
import re
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKEN_RE = re.compile(r"[^\W_]+(?:[-‐‑–][^\W_]+)*", re.UNICODE)


def tokenize(text):
    # Some source titles contain serialized XML dictionaries: retain text only.
    def unpack(match):
        try:
            return str(ast.literal_eval(match.group()).get('#text', ''))
        except (ValueError, SyntaxError):
            return ' '
    text = re.sub(r"\{[^{}]*\}", unpack, text)
    return [w for w in TOKEN_RE.findall(text.lower()) if any(c.isalpha() for c in w)]


def build_vocab(documents, min_count=5):
    counts = Counter(w for doc in documents for w in doc)
    words = sorted((w for w in counts if counts[w] >= min_count),
                   key=lambda w: (-counts[w], w))
    return words, [counts[w] for w in words]


def positive_pairs(ids, window):
    """Yield ordered pairs without crossing this document or discarded-word gaps."""
    for i, center in enumerate(ids):
        if center < 0:
            continue
        for j in range(max(0, i-window), min(len(ids), i+window+1)):
            if i != j and ids[j] >= 0:
                yield center, ids[j]


def pair_array(documents, window):
    # Vectorized equivalent of positive_pairs; never concatenate document sequences.
    blocks = []
    for seq in documents:
        seq = np.asarray(seq, dtype=np.int64)
        for offset in range(1, window+1):
            a, b = seq[:-offset], seq[offset:]
            keep = (a >= 0) & (b >= 0)
            if keep.any():
                pair = np.column_stack((a[keep], b[keep]))
                blocks.extend((pair, pair[:, ::-1]))
    if not blocks:
        raise ValueError('No training pairs: lower min_count or use more data.')
    return np.concatenate(blocks)


class NegativeSampler:
    def __init__(self, counts, device='cpu'):
        if len(counts) < 2 or min(counts) <= 0:
            raise ValueError('Need at least two positive-frequency words.')
        weights = torch.tensor(counts, dtype=torch.float32, device=device).pow(.75)
        self.probabilities = weights / weights.sum()

    def sample(self, contexts, k):
        # With replacement; redraw only collisions with the positive context.
        negatives = torch.multinomial(self.probabilities, contexts.numel()*k,
                                      replacement=True).reshape(-1, k)
        conflict = negatives.eq(contexts[:, None])
        while conflict.any():
            negatives[conflict] = torch.multinomial(
                self.probabilities, int(conflict.sum()), replacement=True)
            conflict = negatives.eq(contexts[:, None])
        return negatives


class SGNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -.5/dim, .5/dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, centers, contexts, negatives):
        v = self.center(centers)
        positive_logits = (v * self.context(contexts)).sum(-1)
        negative_logits = (v[:, None, :] * self.context(negatives)).sum(-1)
        # Binary logistic loss: positive label 1, negative labels 0.
        return -(F.logsigmoid(positive_logits)
                 + F.logsigmoid(-negative_logits).sum(-1)).mean()


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_vectors(directory):
    import json
    from pathlib import Path
    directory = Path(directory)
    vocab = json.loads((directory/'vocab.json').read_text())
    return vocab['words'], np.load(directory/'vectors.npy')
