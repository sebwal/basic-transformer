# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable

from model.multiheadattention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward

class Decoder(nn.Module):
    def __init__(self, n_layers, n_heads):
        super(Decoder, self).__init__()
        
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(512)

    def forward(self, inputs, encoderOut, src_mask, tgt_mask):
        x = inputs

        for layer in self.decoderLayers:
            x = layer(x, encoderOut, src_mask, tgt_mask) 

        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mhattention1 = MultiHeadAttention(n_heads)
        self.mhattention2 = MultiHeadAttention(n_heads)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.feedforward = PositionwiseFeedForward(512, 2048)

    def forward(self, inputs, encoderOut, src_mask, tgt_mask):
        x = inputs
        z = x
        x = self.mhattention1(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = z + x        
        x = self.norm1(x)
        z = x
        x = self.mhattention2(x, encoderOut, encoderOut, src_mask) 
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = z + x
        x = self.norm3(x)
        return x