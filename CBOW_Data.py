import numpy as np
import torch
from torch.utils.data import Dataset


class CBOWOneHotDataset(Dataset):
    def __init__(self, texts, tokenizer, context_size=4, max_length=128):
        """
        Converts token IDs to one-hot vectors for your Linear-based CBOW.
        One-hot vectors are built lazily in __getitem__ to save memory.
        """
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.vocab_size = tokenizer.vocab_size

        # Store (context_ids, target_id) pairs, not one-hot tensors (memory friendly)
        self.pairs = []

        for text in texts:
            # Tokenize to IDs (not one-hot yet)
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)

            # Create context-target pairs
            for i in range(context_size, len(tokens) - context_size):
                # Context: words around target (left and right)
                context = tokens[i - context_size:i] + tokens[i + 1:i + context_size + 1]
                target = tokens[i]
                self.pairs.append((context, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        # Convert context IDs to one-hot on the fly
        one_hot = np.zeros((len(context), self.vocab_size), dtype=np.float32)
        for i, token_id in enumerate(context):
            if token_id < self.vocab_size:
                one_hot[i, token_id] = 1.0
        return torch.from_numpy(one_hot), torch.tensor(target, dtype=torch.long)
