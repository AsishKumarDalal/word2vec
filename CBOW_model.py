import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self,vocab,emb_size):
        super().__init__()
        self.W=nn.Linear(vocab,emb_size)

        self.W_dash=nn.Linear(emb_size,vocab)
    def forward(self,x):
        x=self.W(x)
        x = torch.mean(x, dim=1)
        out=self.W_dash(x)
        return out
    
