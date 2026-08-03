import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):
    """
    Skip-gram dataset.

    Unlike CBOW (many context words -> predict 1 center), skip-gram reverses it:
    for every center word we emit ONE (center, context) sample per word in its
    window (left AND right). The model then predicts the single context word
    from the center word's one-hot vector.

    Each __getitem__ returns:
        (center_one_hot (V,), context_token_id ())
    """

    def __init__(self, texts, tokenizer, context_size=2, max_length=128,
                 max_pairs=None, seed=42):
        """
        context_size : words on each side of the center word
        max_pairs    : early-stop once this many (center, context) pairs are
                       built (chunks are shuffled first so the sampled pairs
                       come from the whole corpus)
        """
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.vocab_size = tokenizer.vocab_size

        # Store (center_id, context_id) pairs, not one-hot tensors
        self.pairs = []

        if max_pairs is not None:
            texts = list(texts)
            random.Random(seed).shuffle(texts)

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True,
                                      truncation=True, max_length=max_length)
            for i in range(context_size, len(tokens) - context_size):
                center = tokens[i]
                # all context words in the window (left and right)
                window = tokens[i - context_size:i] + tokens[i + 1:i + context_size + 1]
                for context in window:
                    self.pairs.append((center, context))
                    if max_pairs is not None and len(self.pairs) >= max_pairs:
                        return  # early stop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        # lazily build the center word's one-hot vector
        one_hot = np.zeros(self.vocab_size, dtype=np.float32)
        if center < self.vocab_size:
            one_hot[center] = 1.0
        return torch.from_numpy(one_hot), torch.tensor(context, dtype=torch.long)


# ----------------------------------------------------------------------------
# Demo: build a tiny tokenizer + dataset and print samples so you can see
# exactly what is happening in the data.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from BPE_Tokenizer import BPETokenizer

    sample_text = [
        "the quick brown fox jumps over the lazy dog",
        "the dog and the fox are fast animals",
        "a lazy dog sleeps all day in the sun",
    ]

    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(sample_text)
    dataset = SkipGramDataset(sample_text, tokenizer, context_size=2)

    print("=" * 64)
    print("1) TOKENIZER: word/token -> id")
    print("=" * 64)
    for w, i in sorted(tokenizer.stoi.items(), key=lambda kv: kv[1]):
        if w not in ("<PAD>", "<UNK>"):
            print(f"   {w:<10s} -> {i}")

    print(f"\n   total (center, context) pairs built: {len(dataset)}")

    print("\n" + "=" * 64)
    print("2) FIRST 12 SAMPLES (center word -> one context word)")
    print("   (note: each center word appears once per context neighbor)")
    print("=" * 64)
    for i in range(min(12, len(dataset))):
        center, context = dataset.pairs[i]
        print(f"   sample {i:>2}: center='{tokenizer.itos[center]}' (id {center:>3}) "
              f"-> context='{tokenizer.itos[context]}' (id {context:>3})")

    print("\n" + "=" * 64)
    print("3) SHAPES RETURNED BY __getitem__ (what the model sees)")
    print("=" * 64)
    one_hot, target = dataset[0]
    center = int((one_hot != 0).nonzero().item())
    print(f"   center one-hot tensor: shape {tuple(one_hot.shape)}  "
          f"(1 non-zero at index {center} = '{tokenizer.itos[center]}')")
    print(f"   context target tensor : shape {tuple(target.shape)}  "
          f"value {target.item()} = '{tokenizer.itos[target.item()]}'")

    print("\n   Model flow: SKIP_Model(one_hot (1, V)) -> logits (1, V)")
    print("   CrossEntropyLoss(logits, context_target)")
