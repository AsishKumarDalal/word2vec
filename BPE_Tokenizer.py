"""
Byte-Pair-Encoding (BPE) tokenizer built on the Python standard library only.

Learns subword merge rules from the corpus, so there is NO out-of-vocabulary
(OOV) problem: any unseen word is greedily split into known subword units,
down to single characters if needed.

API matches what CBOWOneHotDataset expects:
    tokenizer.vocab_size
    tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=...)
and what train.py uses for analysis/plots:
    tokenizer.stoi / tokenizer.itos / tokenizer.frequent_words
"""

import heapq
import re
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=5000, max_unique_words=8000, min_freq=2):
        """
        vocab_size       : target vocabulary size (actual size may be slightly
                           smaller; training stops when no more pairs exist)
        max_unique_words : only the most frequent words are used for learning
                           merge rules (bounds training time)
        min_freq         : only words seen at least this many times are used
                           for merge learning
        """
        self.vocab_size_target = vocab_size
        self.max_unique_words = max_unique_words
        self.min_freq = min_freq

        # <PAD> = 0, <UNK> = 1, then characters, then merged subwords
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.merge_rank = {}   # (left_symbol, right_symbol) -> merge index (lower = merged earlier)
        self.frequent_words = []
        self.vocab_size = 2

    # ------------------------------------------------------------------ #
    # Pre-tokenization                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split(text):
        return re.findall(r"[a-z0-9']+", text.lower())

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, texts):
        # 1) word frequencies over the whole corpus
        word_freqs = Counter()
        for text in texts:
            for w in self._split(text):
                word_freqs[w] += 1

        # 2) initial alphabet from ALL words (so every char is in vocab)
        for w in word_freqs:
            for ch in w:
                if ch not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[ch] = idx
                    self.itos[idx] = ch

        # 3) keep the most frequent words only, to learn merge rules
        most = word_freqs.most_common(self.max_unique_words)
        word_freqs = {w: f for w, f in most if f >= self.min_freq}
        # 4) iterative merges (priority-queue based, lazy deletions)
        word_syms = {}                 # word -> current list of symbols
        pair_counts = defaultdict(int) # (a, b) -> weighted count
        pair_words = defaultdict(set)  # (a, b) -> words containing that pair
        heap = []                      # (-count, pair) lazy priority queue

        for w, f in word_freqs.items():
            syms = list(w)
            word_syms[w] = syms
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                if pair_counts[p] == 0:
                    heap.append((-f, p))
                pair_counts[p] += f
                pair_words[p].add(w)

        heapq.heapify(heap)
        num_merges = self.vocab_size_target - len(self.stoi)
        m = 0
        while m < num_merges:
            # pop the pair with the highest weighted count
            best = None
            while heap:
                neg_count, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -neg_count and -neg_count > 0:
                    best = p
                    break
            if best is None:
                break

            self.merge_rank[best] = m
            new_sym = best[0] + best[1]
            new_id = len(self.stoi)
            self.stoi[new_sym] = new_id
            self.itos[new_id] = new_sym
            m += 1

            # update only the words that contain the merged pair
            for w in list(pair_words[best]):
                f = word_freqs[w]
                syms = word_syms[w]

                # remove this word's old pair contributions
                for i in range(len(syms) - 1):
                    p = (syms[i], syms[i + 1])
                    pair_counts[p] -= f
                    heapq.heappush(heap, (-pair_counts[p], p))

                # merge non-overlapping occurrences left to right
                new_syms = []
                i = 0
                while i < len(syms):
                    if (i < len(syms) - 1 and syms[i] == best[0]
                            and syms[i + 1] == best[1]):
                        new_syms.append(new_sym)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                word_syms[w] = new_syms

                # add this word's new pair contributions
                for i in range(len(new_syms) - 1):
                    p = (new_syms[i], new_syms[i + 1])
                    pair_counts[p] += f
                    pair_words[p].add(w)
                    heapq.heappush(heap, (-pair_counts[p], p))

            # no word contains this pair anymore
            del pair_counts[best]
            pair_words.pop(best, None)

        self.vocab_size = len(self.stoi)

        # 5) token frequencies over the corpus (for the "frequent words" view).
        #    Use ALL words (not only the merge-training subset) so the list is
        #    never empty, even on tiny corpora.
        tok_counts = Counter()
        for w, f in most:
            for t in self.encode_word(w):
                tok_counts[t] += f
        self.frequent_words = [self.itos[t] for t, _ in tok_counts.most_common()
                               if t not in (0, 1)]

    # ------------------------------------------------------------------ #
    # Encoding                                                            #
    # ------------------------------------------------------------------ #
    def encode_word(self, word):
        """Encode a single word into subword token ids (never OOV)."""
        if word in self.stoi:
            return [self.stoi[word]]
        if not all(ch in self.stoi for ch in word):
            return [self.stoi["<UNK>"]]
        INF = float("inf")
        syms = list(word)
        while len(syms) > 1:
            # merge the adjacent pair with the LOWEST rank (= merged first)
            best_rank = INF
            best_i = 0
            for i in range(len(syms) - 1):
                r = self.merge_rank.get((syms[i], syms[i + 1]), INF)
                if r < best_rank:
                    best_rank = r
                    best_i = i
            if best_rank == INF:
                break
            syms = syms[:best_i] + [syms[best_i] + syms[best_i + 1]] + syms[best_i + 2:]
        return [self.stoi[s] for s in syms]

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = []
        for w in self._split(text):
            ids.extend(self.encode_word(w))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids = [self.stoi["<UNK>"]] + ids
        return ids
