import torch
import torch.nn as nn
import torch.nn.functional as F


class Negative_Model(nn.Module):
    """Skip-gram with negative sampling.

    emb1 : center / input embeddings   (V, E)   <- these are the word vectors
    emb2 : context / output embeddings (V, E)

    No softmax over the whole vocabulary - for each (center, context) pair we
    score ONE positive context word and K sampled negative words, using
    binary cross-entropy (sigmoid) as the objective.
    """

    def __init__(self, vocab, emb):
        super().__init__()
        self.emb1 = nn.Embedding(vocab, emb)
        self.emb2 = nn.Embedding(vocab, emb)

    def forward(self, center, context, negatives):
        """
        center   : (B,) token ids of center words
        context  : (B,) token ids of the positive context word
        negatives: (B, K) token ids of K negative samples per pair

        returns: pos_scores (B,), neg_scores (B, K) - raw dot products
        """
        c = self.emb1(center)                      # (B, E)
        pos = self.emb2(context)                   # (B, E)
        neg = self.emb2(negatives)                 # (B, K, E)

        pos_score = (c * pos).sum(dim=1)           # (B,)  dot(center, context)
        neg_scores = torch.bmm(neg, c.unsqueeze(-1)).squeeze(-1)  # (B, K)

        return pos_score, neg_scores

    def ns_loss(self, center, context, negatives):
        """Negative sampling loss:
        -log sigma(pos) - mean over negatives of log sigma(-neg)"""
        pos_score, neg_scores = self.forward(center, context, negatives)
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        return pos_loss + neg_loss
