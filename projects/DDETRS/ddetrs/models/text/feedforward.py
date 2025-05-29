import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Feedforard_Network(nn.Module):
    def __init__(self, d_model=768, d_ffn=2048,
                 dropout=0.1
                 ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.gelu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        # self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout2(tgt2)

        return tgt