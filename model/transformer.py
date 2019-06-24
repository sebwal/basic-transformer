# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable
from model.encoder import Encoder
from model.decoder import Decoder
from model.embeddings import Embeddings
from model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n_layers=6, n_heads=8):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_heads)
        self.decoder = Decoder(n_layers, n_heads)
        self.src_embed = nn.Sequential(Embeddings(512, src_vocab), PositionalEncoding(512, 0.1))
        self.tgt_embed = nn.Sequential(Embeddings(512, tgt_vocab), PositionalEncoding(512, 0.1))
        self.proj = nn.Linear(512, tgt_vocab)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        x = self.encoder(self.src_embed(src), src_mask)
        x = self.decoder(self.tgt_embed(tgt), x, src_mask, tgt_mask)
        x = self.finalize_output(x)
        return x
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        encoderOut = self.encoder(self.src_embed(src), src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            tgt = Variable(ys)
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            out = self.decoder(self.tgt_embed(tgt), encoderOut, src_mask, tgt_mask) 
            prob = self.finalize_output(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys    
    
    def finalize_output(self, inputs): 
        x = F.log_softmax(self.proj(inputs), dim=-1)
        return x
    