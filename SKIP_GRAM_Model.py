import torch
import torch.nn as nn
import torch.nn.functional as F

class SKIP_Model(nn.Module):
    def __init__(self, vocab, emb):
        super().__init__()
        self.W = nn.Linear(vocab, emb)
        self.W_dash = nn.Linear(emb, vocab)

    def forward(self, x):
        x = self.W(x)
        x = self.W_dash(x)
        return x
    