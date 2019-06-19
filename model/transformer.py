from model.encoder import Encoder
from model.decoder import Decoder
from model.constants import TRANS_CONST, GLOBAL

import torch
import torch.nn as nn
import numpy

class Transformer(nn.Module):
    def __init__(self, n_layers=TRANS_CONST['n_attention_layers'], n_attention_heads=TRANS_CONST['n_attention_heads']):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_attention_heads)
        self.decoder = Decoder(n_layers, n_attention_heads)
        self.embedding = nn.Embedding(TRANS_CONST['embedding_dic_size'], TRANS_CONST['embedded_vec_size'])
        # self.posEncoding = #TODO
        self.linear = nn.Linear(TRANS_CONST['linear_input'], TRANS_CONST['linear_output'])
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, inputs=None):
        if inputs != None: 
            raise NotImplementedError

        import random
        inputs = []
        for _ in range(13): inputs.append(numpy.zeros(26)) # 26 is vocab size, should be constant; 13 is just a random amount of words in the sequence
        inputs = torch.Tensor(inputs)
        for i in inputs: i[random.randint(0, len(i) - 1)] = 1

        return self.forward(inputs.long())

    def forward(self, inputs):
        x = self.doEmbedding(inputs)
        x = self.encoder(x)
        x, weights = self.decoder(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x, weights

    def doEmbedding(self, inputs):
        x = inputs.nonzero()[:, 1] # this gets all indices of nonzero values from the inputs matrix
        x = self.embedding(x)
        # x = self.posEncoding(x)
        return x






