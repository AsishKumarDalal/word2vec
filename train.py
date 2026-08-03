"""
Full CBOW (Word2Vec) training script
====================================
- Builds a BPE tokenizer (subword units -> NO out-of-vocabulary problem) and a
  corpus (downloads text8 if possible, else uses a built-in sample corpus so
  it always runs in the cloud).
- Trains the Linear-based CBOW model with tqdm progress bars.
- After training:
    * prints most similar words (cosine similarity on learned embeddings)
    * prints raw vector representations
    * shows a word2vec-style analogy (king - man + woman ~= queen)
    * saves graphs: loss curve, 2D t-SNE/PCA projection, embedding heatmap,
      similarity heatmap, similarity bar charts
    * saves model + embeddings

Run in cloud:
    pip install torch tqdm matplotlib scikit-learn
    python train.py
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
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless-safe (cloud)
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from CBOW_model import CBOW
from CBOW_Data import CBOWOneHotDataset
from BPE_Tokenizer import BPETokenizer
from tqdm.auto import tqdm as _tqdm_auto   # widget bars in notebooks, text elsewhere
import tqdm.std

# True when tqdm.auto picked the ipywidgets (notebook) renderer => bars can
# be redrawn in place. Native Jupyter/Colab/Kaggle kernels choose it.
NOTEBOOK_BARS = _tqdm_auto is not tqdm.std.tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG = {
    "emb_size": 100,            # embedding dimension
    "context_size": 2,          # words on each side of the target
    "max_length": 256,
    "vocab_size": 10000,         # keep the 5000 most frequent words
    "batch_size": 64,
    "epochs": 12,
    "lr": 0.001,
    "seed": 42,
    "max_corpus_words": 1500000,  # cap corpus (text8) for memory/time
    "max_pairs": 200_000,         # cap number of (context, target) pairs
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_url": "https://mattmahoney.net/dc/text8.zip",
    "out_dir": "outputs",
}

SAMPLE_CORPUS = """
Machine learning is a field of computer science that gives computers the ability
to learn without being explicitly programmed. Word2vec is a popular technique for
learning word embeddings from raw text. Word embeddings map words to dense vectors
of real numbers in a high dimensional space. Words that appear in similar contexts
tend to have similar embeddings. The famous example is that the vector king minus
man plus woman is very close to the vector queen. The neural network learns
distributed representations of words during training. CBOW and skip gram are two
architectures used by word2vec. In the continuous bag of words model the network
predicts a target word from its surrounding context words. The king and the queen
ruled the kingdom with wisdom and grace. The man and the woman walked together
through the garden. The model learned that good words and bad words have opposite
meanings in the vector space. Similar words like cat and dog often share nearby
neighbors in the embedding space. Deep learning models trained on large text
corpora produce useful vector representations for many natural language processing
tasks. Language modeling requires a large vocabulary and huge amounts of training
data. Neural networks use gradient descent to update their weights during
backpropagation. Training a model on a small corpus still shows interesting
patterns between related words. The computer learned to group words about animals
together and words about food together. Vector arithmetic allows us to find
analogies between pairs of related words.
"""

CANDIDATE_WORDS = [
    "king", "queen", "man", "woman", "good", "bad", "word", "language",
    "learning", "neural", "model", "computer", "vector", "words", "text",
]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def download_text8(cfg):
    """Try to download the text8 corpus (plain text inside a zip)."""
    if os.path.exists("text8"):
        return True
    try:
        print(f"Downloading text8 from {cfg['data_url']} ...")
        zip_path = "text8.zip"
        urllib.request.urlretrieve(cfg["data_url"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("text8")
        os.remove(zip_path)
        print("Downloaded text8 corpus.")
        return True
    except Exception as e:
        print(f"Could not download text8 ({e}). Using built-in sample corpus instead.")
        return False


def load_corpus(cfg):
    if os.path.exists("corpus.txt"):
        print("Using local corpus.txt")
        text = open("corpus.txt", encoding="utf-8").read()
    elif download_text8(cfg):
        print("Using text8 corpus")
        text = open("text8", encoding="utf-8").read()
    else:
        print("Using built-in sample corpus")
        text = SAMPLE_CORPUS

    words = re.findall(r"[a-z0-9']+", text.lower())[:cfg["max_corpus_words"]]
    chunk = cfg["max_length"] // 2
    texts = [" ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)]
    return texts


# ----------------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------------
def get_embedding_matrix(model):
    """Average input + output embeddings (standard word2vec trick)."""
    # nn.Linear stores weights as (out_features, in_features), so transpose
    # W.weight: (emb, V) -> (V, emb); W_dash.weight is already (V, emb)
    W = model.W.weight.detach().cpu().numpy().T        # (V, E)
    Wd = model.W_dash.weight.detach().cpu().numpy()    # (V, E)
    return (W + Wd) / 2.0


def normalize_rows(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


# ----------------------------------------------------------------------------
# Similar-word analysis
# ----------------------------------------------------------------------------
def similar_words(tokenizer, E_n, word, topk=10):
    if word not in tokenizer.stoi:
        return None
    idx = tokenizer.stoi[word]
    v = E_n[idx]
    sims = E_n @ v
    sims[idx] = -np.inf
    order = np.argsort(sims)[::-1][:topk]
    return [(tokenizer.itos[int(i)], float(sims[i])) for i in order]


def print_similar_words(results, topk=10):
    print("\n" + "=" * 60)
    print("MOST SIMILAR WORDS (cosine similarity on learned embeddings)")
    print("=" * 60)
    for word, sims in results.items():
        line = ", ".join(f"{w} ({s:.3f})" for w, s in sims[:topk])
        print(f"  {word:<10s} -> {line}")


def show_analogy(tokenizer, E_n, a, b, c, k=5):
    if not all(w in tokenizer.stoi for w in (a, b, c)):
        print(f"  (skip analogy {a} - {b} + {c}: word(s) not in vocab)")
        return
    ia, ib, ic = tokenizer.stoi[a], tokenizer.stoi[b], tokenizer.stoi[c]
    target = E_n[ia] - E_n[ib] + E_n[ic]
    sims = E_n @ target
    sims[[ia, ib, ic]] = -np.inf
    order = np.argsort(sims)[::-1][:k]
    tops = [(tokenizer.itos[int(i)], float(sims[i])) for i in order]
    print(f"  {a} - {b} + {c}  ~=  {tops[0][0]} (sim {tops[0][1]:.3f}) | "
          f"top {k}: " + ", ".join(f"{w} ({s:.3f})" for w, s in tops))


def print_vectors(tokenizer, E_n, words, show_dims=16):
    print("\n" + "=" * 60)
    print("VECTOR REPRESENTATIONS (first {} dims of normalized embedding)".format(show_dims))
    print("=" * 60)
    for w in words:
        v = E_n[tokenizer.stoi[w]]
        vals = " ".join(f"{x:+.3f}" for x in v[:show_dims])
        print(f"  {w:<10s} | {vals} ... (dim={len(v)})\n")


def pick_query_words(tokenizer, n=6):
    out = []
    for w in CANDIDATE_WORDS:
        if w in tokenizer.stoi:
            out.append(w)
        else:
            # under BPE a word may be split; use its first subword token
            ids = tokenizer.encode(w)
            if ids and len(tokenizer.itos[ids[0]]) > 1:
                out.append(tokenizer.itos[ids[0]])
        if len(out) >= n:
            break
    for w in tokenizer.frequent_words:
        if len(out) >= n:
            break
        if w not in out:
            out.append(w)
    return out[:n]


def pca_2d(X):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_loss_curve(losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("CBOW Training Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    plt.close(fig)


def plot_embeddings_2d(E_n, tokenizer, n_words=120, out_dir="outputs"):
    words = tokenizer.frequent_words[:n_words]
    idxs = [tokenizer.stoi[w] for w in words if w in tokenizer.stoi]
    if len(idxs) < 5:
        print("  (too few words for a 2D plot)")
        return
    X = E_n[idxs]
    method = "pca"
    if SKLEARN_OK and len(idxs) >= 20:
        try:
            perp = min(30, max(5, (len(idxs) - 1) // 3))
            X2 = TSNE(n_components=2, perplexity=perp, random_state=42,
                      init="pca").fit_transform(X)
            method = "tsne"
        except Exception:
            X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X2 = pca_2d(X)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.scatter(X2[:, 0], X2[:, 1], s=30, alpha=0.6)
    for i, w in enumerate(words):
        ax.annotate(w, (X2[i, 0], X2[i, 1]), fontsize=9, alpha=0.85)
    ax.set_title(f"Word embeddings ({method.upper()} projection), top {len(idxs)} frequent words")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"embeddings_{method}.png"), dpi=150)
    plt.close(fig)


def plot_embedding_heatmap(E, tokenizer, words, out_dir="outputs", max_dims=64):
    idxs = [tokenizer.stoi[w] for w in words]
    M = E[idxs][:, :max_dims]
    vmax = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(14, max(4, 0.55 * len(words))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xticks(np.arange(0, max_dims, max(1, max_dims // 16)))
    ax.set_xlabel("embedding dimension")
    ax.set_title("Vector representation (first {} dims of averaged embeddings)".format(M.shape[1]))
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedding_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_similarity_heatmap(E_n, tokenizer, words, out_dir="outputs"):
    idxs = [tokenizer.stoi[w] for w in words]
    S = E_n[idxs] @ E_n[idxs].T
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cosine similarity between query words")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "similarity_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_similarity_bars(results, out_dir="outputs"):
    n = len(results)
    cols = 3
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax, (word, sims) in zip(axes.ravel(), results.items()):
        sims = sims[:10][::-1]  # ascending for barh
        ax.barh([s[0] for s in sims], [s[1] for s in sims], color="steelblue")
        ax.set_title(f"Words most similar to '{word}'", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis="x")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "similarity_bars.png"), dpi=150)
    plt.close(fig)


def save_embeddings(E, tokenizer, out_dir):
    np.save(os.path.join(out_dir, "embeddings.npy"), E)
    with open(os.path.join(out_dir, "embeddings.csv"), "w", encoding="utf-8") as f:
        f.write("word," + ",".join(f"d{i}" for i in range(E.shape[1])) + "\n")
        for i in range(E.shape[0]):
            f.write(tokenizer.itos[i] + "," + ",".join(f"{x:.6f}" for x in E[i]) + "\n")
    print(f"Embeddings saved -> {os.path.join(out_dir, 'embeddings.npy')} / .csv")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    device = torch.device(cfg["device"])
    print("=" * 60)
    print(f"CBOW Word2Vec training | device: {device}")
    print("=" * 60)

    # 1) Corpus
    texts = load_corpus(cfg)
    print(f"Corpus: {len(texts)} text chunks")

    # 2) Tokenizer (BPE -> subword units, no OOV)
    tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
    tokenizer.train(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size} (BPE subword tokens)")

    # 3) Dataset / DataLoader
    dataset = CBOWOneHotDataset(texts, tokenizer,
                                context_size=cfg["context_size"],
                                max_length=cfg["max_length"])
    dataset.pairs = dataset.pairs[:cfg["max_pairs"]]
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    print(f"Training pairs: {len(dataset)}")

    # 4) Model
    model = CBOW(vocab=tokenizer.vocab_size, emb_size=cfg["emb_size"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # 5) Training loop with tqdm progress bars.
    # One bar per epoch (no nested bars). Widget bars in Jupyter/Colab/Kaggle
    # redraw in place; on a real terminal the bar rewrites the same line; when
    # output is piped (no widgets, no TTY) the bar is disabled and we only
    # print one clean "avg_loss" line per epoch so logs don't spam.
    is_tty = sys.stderr.isatty()
    losses = []
    print("\nTraining started ...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        batch_bar = _tqdm_auto(dataloader, desc=f"Epoch {epoch:>2}/{cfg['epochs']}",
                               leave=False, mininterval=0.5,
                               disable=not (NOTEBOOK_BARS or is_tty))
        for ctx, tgt in batch_bar:
            ctx, tgt = ctx.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(ctx)
            loss = criterion(logits, tgt)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(tgt)
            count += len(tgt)
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
    }, "cbow_model.pt")
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE | final loss: {losses[-1]:.4f}")
    print("Model saved -> cbow_model.pt")
    print("=" * 60)

    # 7) Analysis
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

    # 8) Plots
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
    print("  cbow_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
