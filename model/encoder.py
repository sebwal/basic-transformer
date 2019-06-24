# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable

from model.multiheadattention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward

class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads):
        super(Encoder, self).__init__()
        
        self.encoderLayers = nn.ModuleList([EncoderLayer(n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(512)

    def forward(self, inputs, mask):
        x = inputs

        for layer in self.encoderLayers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mhattention = MultiHeadAttention(n_heads)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feedforward = PositionwiseFeedForward(512, 2048)

    def forward(self, inputs, mask):
        x = inputs 
        z = x
        x = self.mhattention(x, x, x, mask)
        x = self.dropout1(x)
        x = z + x
        x = self.norm1(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x) 
        return x