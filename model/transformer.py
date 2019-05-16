from model.encoder import Encoder
from model.decoder import Decoder
from model.constants import TRANS_CONST

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, n_layers=TRANS_CONST['n_attention_layers'], n_attention_heads=TRANS_CONST['n_attention_heads']):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_attention_heads)
        self.decoder = Decoder(n_layers, n_attention_heads)
        self.embedding = nn.Embedding(TRANS_CONST['embedding_dic_size'], TRANS_CONST['embedded_vec_size'])
        # self.posEncoding = #TODO
        self.linear = nn.Linear(TRANS_CONST['linear_input'], TRANS_CONST['linear_output'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.doEmbedding(inputs)
        # x = self.posEncoding(x)
        x, encoderKV = self.encoder(x) 
        x = self.decoder(x, encoderKV) 
        x = self.linear(x) 
        x = self.softmax(x) 
        return x
    
    def doEmbedding(self, inputs):
        x = []
        for word in inputs:
            # get every word as embedding vector from the embedding matrix
            embedded = self.embedding(word)
            # append unsqueezed so they're rows, not columns 
            x.append(embedded.unsqueeze(0))
        # concat all embedded tensors to one 2D tensor containing all embedded inputs
        x = torch.cat(x)
        return x