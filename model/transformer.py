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
        # TODO FIRST PRIO
            # get all inputs for embedding (real example for translation tasks etc, noise, decoder input) on the same format, which should be NxV
        #### ENCODING ####
        x = self.doEmbedding(inputs)
        # x = self.posEncoding(x)
        _, encoderKV = self.encoder(x) #TODO try running the encoder output trough 2 additional linear layers to make the KV matrices

        #### DECODING ####
        sos = numpy.zeros(GLOBAL['n_vocab'])
        sos[0] = 1
        x = torch.Tensor([sos])
        x_embedded = self.doEmbedding(x)
        while len(x) < TRANS_CONST['max_output_length']: #TODO add eos token
            ## Embedding
            # x_embedded = self.posEncoding(x_embedded) #TODO
            ## Decoding
            new_word = self.decoder(x_embedded, encoderKV)
            print(new_word)
            new_word = self.linear(new_word)
            new_word = self.softmax(new_word)
            # new_word = new_word.long()
            x = torch.cat([x, new_word], dim=0)

        return x

    def doEmbedding(self, inputs):
        inputs = inputs.nonzero()[:, 1] # this gets all indices of nonzero values from the inputs matrix
        return self.embedding(inputs)






