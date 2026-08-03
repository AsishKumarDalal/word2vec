# Word2Vec (CBOW) from scratch — PyTorch

A from-scratch implementation of the **Continuous Bag-of-Words (CBOW)** Word2Vec
model built with a **Linear stack (no embedding layer)**, a **custom BPE
subword tokenizer** (so there is **no out-of-vocabulary problem**), and a full
training pipeline with `tqdm` progress bars, similar-word analysis, analogies,
and visualization graphs.

Everything runs on either a single machine or a cloud VM / Colab. All tokenizer
code is **standard-library only** — no extra tokenizer package is required.

---

## Table of contents

1. [Project structure](#project-structure)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [The pipeline step by step](#the-pipeline-step-by-step)
7. [Outputs and visualizations](#outputs-and-visualizations)
8. [Using your own corpus](#using-your-own-corpus)
9. [Notes / troubleshooting](#notes--troubleshooting)
10. [License / references](#references)

---

## Project structure

```
word2vec/
├── train.py            # main pipeline: data -> tokenizer -> model -> train -> analyze -> plot
├── CBOW_model.py       # the CBOW neural network (two Linear layers)
├── CBOW_Data.py        # Dataset that builds (context, target) pairs as one-hot
├── BPE_Tokenizer.py    # BPE subword tokenizer (stdlib only, no OOV)
├── outputs/            # generated after training (gitignored)
│   ├── training_loss.png        # loss curve
│   ├── embeddings_tsne.png      # or embeddings_pca.png
│   ├── embedding_heatmap.png
│   ├── similarity_heatmap.png
│   ├── similarity_bars.png
│   ├── embeddings.npy / .csv
├── cbow_model.pt       # saved checkpoint (gitignored)
├── text8 / corpus.txt  # optional corpus files (gitignored)
└── .gitignore
```

---

## How it works

### CBOW architecture

CBOW predicts the **center word** from its **surrounding context words**.

```
context words (one-hot)          target word
        "the king ___ the kingdom"
              |                        input: one-hot
   context_one_hot : (ctx=4, vocab) ──►  word at position i
        │
        ▼
      W : Linear(vocab, emb)           # "lookup" weights
  context_embeddings : (4, emb)
        │
        ▼
  torch.mean(dim=1) ──► (emb,)        # bag-of-words average
        │
        ▼
      W_dash : Linear(emb, vocab)      # "output" weights
        │
        ▼
   logits : (vocab,)
   CrossEntropyLoss(logits, target_id)
```

Because the input is a **one-hot** vector, multiplying by the first Linear layer
is exactly the same as **looking up a row of the embedding matrix** (the classic
"embedding" operation). The weight matrix `W` *is* your word vectors (the
output layer `W_dash` gives a second, symmetrized version).

> **Important:** `nn.Linear` stores weights as `(out_features, in_features)`,
> so the *input* weight `W.weight` has shape `(emb, vocab)` and must be
> **transposed** to `(vocab, emb)` before averaging with `W_dash.weight`,
> which is already `(vocab, emb)`. See `get_embedding_matrix()`.

The final word vector is the **average of input and output embeddings**
(standard word2vec trick), then L2-normalized for cosine similarity:

```python
E = (model.W.weight.T + model.W_dash.weight) / 2
E_n = E / ||E|| along each row
```

### BPE tokenizer (no OOV)

`BPE_Tokenizer.py` learns subword merge rules on the corpus and encodes any word
— **even one never seen during training** — by splitting it into known subword
tokens, down to single characters if necessary. So, unlike a word-level
vocabulary, `<UNK>` is rarely used.

How it is trained:

1. Pre-tokenize the corpus into words (`[a-z0-9']+`, lowercased).
2. Build a frequency table.
3. Initial vocab = `<PAD>`, `<UNK>`, followed by all unique characters.
4. Iteratively merge the most frequent adjacent character/subword pair into a
   new token, until the target `vocab_size` is reached (or no pairs remain).
   Merges are only learned from the **most frequent words** (`max_unique_words`,
   `min_freq`) so training stays fast even on large texts.

How it encodes a word: a greedy pass that always merges the adjacent pair with
the **lowest merge rank** (i.e. the pair that was merged earliest during
training), guaranteeing it reproduces the exact segmentation for words seen in
the corpus and a sensible segmentation for words it has never seen.

`BPE_Tokenizer` mirrors the exact API expected by `CBOW_Data`:

```python
tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)
tokenizer.vocab_size
```

---

## Requirements

```
torch
tqdm
matplotlib
scikit-learn      (optional — skipped if not installed; PCA fallback built in)
```

Python **3.8+** recommended. No tokenizer package needed.

Install:

```bash
pip install torch tqdm matplotlib scikit-learn
```

---

## Quick start

```bash
python train.py
```

That's it. The script will:

1. Try to **download text8** (Matt Mahoney) automatically.
2. Fall back to a **built-in sample corpus** if the download fails (so it runs
   anywhere, even offline).
3. Train the CBOW model and print training progress via `tqdm`.
4. Print similar words, raw vectors, and a word2vec-style analogy.
5. Save plots + model + embeddings into `outputs/`.

---

## Configuration

All knobs live in the `CONFIG` dict at the top of `train.py`:

| Key                | Default | Description |
|--------------------|---------|-------------|
| `emb_size`         | `100`   | embedding dimension |
| `context_size`     | `2`     | words on each side of the target |
| `max_length`       | `256`   | max tokens per training chunk |
| `vocab_size`       | `10000`  | target BPE vocab size |
| `batch_size`       | `64`    | batches per step |
| `epochs`            | `12`    | number of training epochs |
| `lr`                | `0.001` | Adam learning rate |
| `seed`              | `42`    | reproducibility seed |
| `max_corpus_words`  | `150_000_000` | cap words used from the corpus (text8 is ~17M words) |
| `max_pairs`         | `2_000_000` | cap the number of training pairs (early-stop, sampled across the whole corpus) |
| `device`            | auto    | `cuda` if available, else `cpu` |
| `data_url`          | `mattmahoney.net/.../text8.zip` | corpus download URL |
| `out_dir`           | `outputs/` | where plots/embeddings go |

Example for a bigger/quieter run:

```python
CONFIG = {
    "emb_size": 128,
    "vocab_size": 8000,
    "epochs": 20,
}
```

---

## The pipeline step by step

1. **Load corpus**
   - try `corpus.txt` (local, if you place one)
   - else download `text8` from the web
   - else fall back to an internal sample corpus.
2. **Chunk** the text into roughly `max_length/2`-token chunks.
3. **Train BPE tokenizer** (`BPE_Tokenizer.py`) from the chunks.
4. **Build dataset** (`CBOW_Data.py`): for each position, take `context_size`
   words on each side as context and the word as target.
   One-hot vectors are created **lazily inside `__getitem__`** so memory stays
   flat even for large corpora.
5. **Train** the two-layer CBOW with `CrossEntropyLoss` + Adam, using nested
   `tqdm` bars (epoch on top, batches below, live loss in the right panel).
6. **Extract embeddings** : `E = (W.T + W_dash) / 2`, normalize for cosine.
7. **Analyze** :
   - printable table of the *k* most similar words to each query word;
   - print the raw normalized vectors of a few words;
   - vector analogies like `king - man + woman ≈ ?`.
8. **Visualize** (all saved under `outputs/`).

---

## Outputs and visualizations

| File (in `outputs/`)      | Contents |
|---------------------------|----------|
| `training_loss.png`       | average per-epoch loss curve |
| `embeddings_tsne.png` / `embeddings_pca.png` | 2D projection (t-SNE if available, otherwise SVD-PCA) of top frequent words |
| `embedding_heatmap.png`   | "image" of the vector representations: each row = a word, each column = an embedding dimension |
| `similarity_heatmap.png`  | cosine similarity between the query words themselves |
| `similarity_bars.png`     | horizontal bar charts: top-10 similar words per query |
| `embeddings.csv` / `.npy` | the full (vocab × emb) matrix for reuse |

On the console you'll also get:

```
MOST SIMILAR WORDS (cosine similarity on learned embeddings)
  king -> queen (0.87), kingdom (0.71), royal (0.62) ...

Word2vec-style analogies
  king - man + woman ~= queen (sim 0.42)
```

---

## Using a custom corpus

Drop a plain text file named `corpus.txt` in this directory, or edit
`CONFIG["data_url"]` to point at your own corpus. The script lowers all text and
tokenizes on word characters, so the pipeline needs no cleaning.

If you'd like to see the BPE fraction in action at a useful scale, prefer a
larger corpus (text8 gives the best demo); the included fallback sample is only
a smoke-test corpus and will produce mostly character-level tokens.

---

## Notes

- **`scikit-learn` is optional.** If it's missing, t-SNE plots fall back to a
  hand-rolled SVD-based PCA (`pca_2d`), so the script still runs.
- **Memory.** One-hot vectors for a full corpus are expensive. We cap the corpus
  with a configurable cap words (`max_corpus_words`) and one-hot is created
  lazily per batch, so it stays green.
- **Headless/cloud** runs are fine: `matplotlib` is forced to the `Agg`
  backend (files are saved, no GUI window needed).

### References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE, 2016)

---

## License

MIT (or as you choose) — this material is a study implementation for
educational purposes.