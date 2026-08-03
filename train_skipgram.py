"""
Full Skip-gram (Word2Vec) training script
=========================================
Same pipeline as train.py (BPE tokenizer, text8/sample corpus, tqdm bars,
similar words, analogies, plots) but for the SKIP-GRAM architecture:

    center word (one-hot)  ->  W (embedding lookup)  ->  W_dash  ->  scores
    loss: CrossEntropyLoss(logits, ONE context word)

Difference vs CBOW: the dataset emits one (center, context) sample per window
neighbor (SKIP_GRAM_Data.py), and the model takes a SINGLE word as input -
no context averaging.

Run in cloud:
    pip install torch tqdm matplotlib scikit-learn
    python train_skipgram.py
"""

import os
import re
import sys
import math
import zipfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # headless-safe (cloud)
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# Reuse all analysis/plot helpers from the CBOW script (they are model-agnostic)
from train import (load_corpus, get_embedding_matrix, normalize_rows,
                   similar_words, print_similar_words, show_analogy,
                   print_vectors, pick_query_words, plot_loss_curve,
                   plot_embeddings_2d, plot_embedding_heatmap,
                   plot_similarity_heatmap, plot_similarity_bars,
                   save_embeddings)

from tqdm.auto import tqdm as _tqdm_auto
import tqdm.std
NOTEBOOK_BARS = _tqdm_auto is not tqdm.std.tqdm

from BPE_Tokenizer import BPETokenizer
from SKIP_GRAM_Data import SkipGramDataset
from SKIP_GRAM_Model import SKIP_Model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG = {
    "emb_size": 100,            # embedding dimension
    "context_size": 4,          # words on each side of the center word
    "max_length": 256,
    "vocab_size": 10000,        # target BPE vocab size
    "batch_size": 256,
    "epochs": 15,               # skip-gram converges slower than CBOW
    "lr": 0.001,
    "seed": 42,
    "max_corpus_words": 150_000_000,  # cap corpus (text8 is ~17M words)
    "max_pairs": 2_000_000,     # cap (center, context) pairs (early-stop)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_url": "https://mattmahoney.net/dc/text8.zip",
    "out_dir": "outputs_skipgram",
}


def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    device = torch.device(cfg["device"])
    print("=" * 60)
    print(f"SKIP-GRAM Word2Vec training | device: {device}")
    print("=" * 60)

    # 1) Corpus (reuses train.py loader: corpus.txt > text8 > sample corpus)
    texts = load_corpus(cfg)
    print(f"Corpus: {len(texts)} text chunks")

    # 2) Tokenizer (BPE -> subword units, no OOV)
    tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
    tokenizer.train(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size} (BPE subword tokens)")

    # 3) Dataset / DataLoader (skip-gram: one (center, context) per neighbor)
    dataset = SkipGramDataset(texts, tokenizer,
                              context_size=cfg["context_size"],
                              max_length=cfg["max_length"],
                              max_pairs=cfg["max_pairs"],
                              seed=cfg["seed"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    print(f"Training (center, context) pairs: {len(dataset)}")

    # 4) Model
    model = SKIP_Model(vocab=tokenizer.vocab_size, emb=cfg["emb_size"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # 5) Training loop (same tqdm setup as train.py: one bar per epoch)
    is_tty = sys.stderr.isatty()
    losses = []
    print("\nTraining started ...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        batch_bar = _tqdm_auto(dataloader, desc=f"Epoch {epoch:>2}/{cfg['epochs']}",
                               leave=False, mininterval=0.5,
                               disable=not (NOTEBOOK_BARS or is_tty))
        for center, context in batch_bar:
            center, context = center.to(device), context.to(device)
            optimizer.zero_grad()
            logits = model(center)  # (B, V) one-hot center -> (B, V) scores
            loss = criterion(logits, context)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(center)
            count += len(center)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg = total / max(count, 1)
        losses.append(avg)
        print(f"   Epoch {epoch:>2}/{cfg['epochs']} | avg_loss = {avg:.4f}")

    # 6) Save model
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "emb_size": cfg["emb_size"],
        "config": cfg,
    }, "skipgram_model.pt")
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE | final loss: {losses[-1]:.4f}")
    print("Model saved -> skipgram_model.pt")
    print("=" * 60)

    # 7) Analysis (identical to train.py: input/output embedding average)
    E = get_embedding_matrix(model)
    E_n = normalize_rows(E)
    query_words = pick_query_words(tokenizer)
    results = {w: similar_words(tokenizer, E_n, w, topk=10) for w in query_words}
    results = {w: s for w, s in results.items() if s is not None}
    print_similar_words(results)
    print("\nWord2vec-style analogies (vector arithmetic):")
    show_analogy(tokenizer, E_n, "king", "man", "woman")
    show_analogy(tokenizer, E_n, "man", "king", "woman")
    show_analogy(tokenizer, E_n, "good", "bad", "word")
    print_vectors(tokenizer, E_n, query_words[:4])

    # 8) Plots (saved under outputs_skipgram so CBOW outputs are not clobbered)
    print("\nGenerating plots ...")
    plot_loss_curve(losses, cfg["out_dir"])
    plot_embeddings_2d(E_n, tokenizer, n_words=120, out_dir=cfg["out_dir"])
    plot_embedding_heatmap(E, tokenizer, query_words, out_dir=cfg["out_dir"])
    plot_similarity_heatmap(E_n, tokenizer, query_words, out_dir=cfg["out_dir"])
    plot_similarity_bars(results, out_dir=cfg["out_dir"])
    save_embeddings(E, tokenizer, cfg["out_dir"])

    print("\n" + "=" * 60)
    print("DONE! Outputs:")
    for f in sorted(os.listdir(cfg["out_dir"])):
        print(f"  {cfg['out_dir']}/{f}")
    print("  skipgram_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
