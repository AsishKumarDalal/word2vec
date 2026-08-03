import random

import numpy as np
import torch
from torch.utils.data import Dataset


class NegativeSamplingDataset(Dataset):
    """
    Skip-gram-style (center, context) pairs for negative sampling.

    IMPORTANT: unlike CBOW/SkipGram datasets this one returns raw TOKEN IDS,
    NOT one-hot vectors - the Negative_Model uses nn.Embedding, so one-hot is
    unnecessary (and 100x cheaper on memory).

    Each __getitem__ returns:
        (center_id (), context_id ())
    """

    def __init__(self, texts, tokenizer, context_size=2, max_length=128,
                 max_pairs=None, seed=42):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.vocab_size = tokenizer.vocab_size

        # store (center_id, context_id) pairs
        self.pairs = []

        if max_pairs is not None:
            texts = list(texts)
            random.Random(seed).shuffle(texts)

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True,
                                      truncation=True, max_length=max_length)
            for i in range(context_size, len(tokens) - context_size):
                center = tokens[i]
                window = tokens[i - context_size:i] + tokens[i + 1:i + context_size + 1]
                for context in window:
                    self.pairs.append((center, context))
                    if max_pairs is not None and len(self.pairs) >= max_pairs:
                        return  # early stop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return (torch.tensor(center, dtype=torch.long),
                torch.tensor(context, dtype=torch.long))


class NegativeSampler:
    """
    Samples negative context words from the token unigram distribution raised
    to the power 0.75 (the standard word2vec trick - boosts rare words).

    P(w) ~ freq(w)^0.75
    """

    def __init__(self, tokenizer, texts, power=0.75, seed=0):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.rng = np.random.default_rng(seed)

        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for text in texts:
            for t in tokenizer.encode(text, add_special_tokens=True,
                                      truncation=True, max_length=1_000_000):
                counts[t] += 1.0
        counts[counts <= 0] = 1e-6  # never give a token zero probability

        probs = counts ** power
        probs /= probs.sum()
        self.probs = probs

    def sample(self, batch_size, k, exclude=None):
        """Draw (batch_size, k) negative ids; if `exclude` (B,) is given,
        any sample equal to the positive context word is re-drawn."""
        idx = self.rng.choice(self.vocab_size, size=(batch_size, k),
                              p=self.probs, replace=True)
        if exclude is not None:
            idx = idx.copy()
            mask = idx == exclude.numpy()[:, None]
            while mask.any():
                new = self.rng.choice(self.vocab_size, size=int(mask.sum()),
                                      p=self.probs, replace=True)
                idx[mask] = new
                mask = idx == exclude.numpy()[:, None]
        return torch.from_numpy(idx).long()
